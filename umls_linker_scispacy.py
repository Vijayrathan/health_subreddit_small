import spacy
import scispacy
from scispacy.linking import EntityLinker

def get_unique_umls_cui(term):
    nlp = spacy.load("en_core_sci_lg")
    
    # We still keep resolve_abbreviations=True
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    
    doc = nlp(term)
    linker = nlp.get_pipe("scispacy_linker")
    
    results = []
    
    for entity in doc.ents:
        # Create a set to track CUIs we have already processed for THIS entity
        seen_cuis = set()
        
        for umls_ent in entity._.kb_ents:
            if entity._.kb_ents:
                umls_ent = entity._.kb_ents[0]
                cui = umls_ent[0]
                score = umls_ent[1]
                
                # --- THE FIX: check if we've seen this CUI before ---
                if cui in seen_cuis:
                    continue
                
                # If not, add it to the set and process it
                seen_cuis.add(cui)
                
                details = linker.kb.cui_to_entity[cui]
                results.append({
                    "Term": entity.text,
                    "CUI": cui,
                    "Name": details.canonical_name,
                    "Score": score
                })
            
    return results

# --- Test ---
term = "Reflux"
cui_data = get_unique_umls_cui(term)

print(cui_data)