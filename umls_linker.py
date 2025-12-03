import json
from datasets import load_dataset
import spacy
from scispacy.linking import UmlsEntityLinker, UmlsKnowledgeBase
from tqdm import tqdm

# ------------------------------------------------------
# Load SciSpaCy + UMLS
# ------------------------------------------------------
print("Loading SciSpaCy...")
nlp = spacy.load("en_core_sci_lg")

print("Loading UMLS Entity Linker...")
linker = UmlsEntityLinker(resolve_abbreviations=True, max_entities_per_mention=1)
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True})

# For canonical names
kb = UmlsKnowledgeBase()


# ------------------------------------------------------
# Helper: Link text → best CUI (or None)
# ------------------------------------------------------
def get_best_cui(text):
    if not text or not isinstance(text, str):
        return None

    doc = nlp(text)

    if len(doc.ents) == 0:
        return None

    ent = doc.ents[0]        # take the main entity
    umls_ents = ent._.umls_ents

    if not umls_ents:
        return None

    cui = umls_ents[0][0]    # best candidate
    return cui


# ------------------------------------------------------
# Helper: Normalize one triple → UMLS triple
# ------------------------------------------------------
def normalize_triple(triple):
    """
    triple format:
      ("subject", "relation", "object")
    We return:
      {
         "subject_name": ...,
         "subject_cui": ...,
         "relation": ...,
         "object_name": ...,
         "object_cui": ...
      }
    Or return None if no CUI for either side.
    """

    if not triple or len(triple) != 3:
        return None

    subj, rel, obj = triple

    # get CUI
    subj_cui = get_best_cui(subj)
    obj_cui  = get_best_cui(obj)

    # skip triples whose entity has no CUI
    if subj_cui is None or obj_cui is None:
        return None

    # canonical names
    subj_name = kb.cui_to_entity[subj_cui].canonical_name
    obj_name  = kb.cui_to_entity[obj_cui].canonical_name

    # return normalized triple
    return {
        "subject_name": subj_name,
        "subject_cui": subj_cui,
        "relation": rel,
        "object_name": obj_name,
        "object_cui": obj_cui,
    }


# ------------------------------------------------------
# Map function for HF dataset
# ------------------------------------------------------
def convert_row(row):
    triples = row.get("triples", None)

    if not triples or len(triples) == 0:
        return {"umls_triples": []}

    normalized = []
    for t in triples:
        result = normalize_triple(t)
        if result is not None:
            normalized.append(result)

    return {"umls_triples": normalized}


# ------------------------------------------------------
# Run on your HF dataset
# ------------------------------------------------------
if __name__ == "__main__":
    print("Loading HF dataset...")
    ds = load_dataset("Vijayrathank/reddit-health-small")

    print("Converting triples → umls_triples...")
    ds_converted = ds.map(
        convert_row,
        batched=False,
        desc="Mapping dataset",
    )

    print("Saving new dataset...")
    ds_converted.save_to_disk("reddit-health-small-umls")


    print("Done! Saved to reddit-health-small-umls/")
