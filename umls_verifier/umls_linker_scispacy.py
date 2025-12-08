import pandas as pd
import spacy
import scispacy
from scispacy.linking import EntityLinker
import ast
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "triples_output_100k.parquet"
OUTPUT_FILE = "triples_cui_mapped.parquet"
CONFIDENCE_THRESHOLD = 0.70

def load_scispacy():
    print("Loading SciSpacy model (en_core_sci_lg)...")
    # Ensure you have installed: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
    nlp = spacy.load("en_core_sci_lg")
    
    print("Adding UMLS Linker...")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, 
        "linker_name": "umls",
        "threshold": CONFIDENCE_THRESHOLD 
    })
    return nlp

def safe_literal_eval(val):
    """Safely parses the string representation of list of tuples."""
    try:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            # Basic sanity check to avoid evaluating random strings
            if val.startswith("[") and val.endswith("]"):
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
        return []
    except (ValueError, SyntaxError):
        return []

def main():
    # 1. Load Data
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 2. Parse Triples
    print("Parsing triple structures...")
    tqdm.pandas(desc="Parsing")
    df['parsed_triples'] = df['triples'].progress_apply(safe_literal_eval)
    
    # 3. Extract UNIQUE terms
    print("Extracting unique entities for batch processing...")
    unique_terms = set()
    for triples_list in df['parsed_triples']:
        # Ensure triples_list is actually a list before iterating
        if not isinstance(triples_list, list): continue
        
        for triple in triples_list:
            # --- FIX: Type Check to prevent Ellipsis error ---
            if not isinstance(triple, (tuple, list)): 
                continue
            
            if len(triple) == 3:
                subj, rel, obj = triple
                # Convert to string to be safe (in case of numbers)
                unique_terms.add(str(subj))
                unique_terms.add(str(obj))
    
    unique_terms_list = list(unique_terms)
    print(f"Found {len(unique_terms_list)} unique terms to link.")

    # 4. Batch Process with SciSpacy
    if not unique_terms_list:
        print("No terms found! Check input data.")
        return

    nlp = load_scispacy()
    linker = nlp.get_pipe("scispacy_linker")
    
    term_to_cui_map = {}
    
    print("Running Entity Linking on unique terms...")
    # Using nlp.pipe for speed
    for doc in tqdm(nlp.pipe(unique_terms_list, batch_size=500), total=len(unique_terms_list)):
        text = doc.text
        
        # We look for the best matching entity in the text
        if len(doc.ents) > 0:
            # We take the first entity found in the string
            entity = doc.ents[0]
            
            if len(entity._.kb_ents) > 0:
                # Get Top-1 Match
                cui, score = entity._.kb_ents[0]
                canonical_name = linker.kb.cui_to_entity[cui].canonical_name
                
                term_to_cui_map[text] = {
                    "cui": cui,
                    "name": canonical_name,
                    "score": float(score)
                }
            else:
                term_to_cui_map[text] = None
        else:
            term_to_cui_map[text] = None

    # 5. Map back to Dataframe
    print("Mapping terms back to triples...")
    
    def map_row(triples):
        if not isinstance(triples, list):
            return []
            
        mapped_triples = []
        for trip in triples:
            # --- FIX: Strict Type Checking ---
            if not isinstance(trip, (tuple, list)): 
                continue # Skip Ellipsis or garbage items
                
            if len(trip) != 3: 
                continue
                
            s, r, o = trip
            s = str(s)
            o = str(o)
            
            # Retrieve cached data
            s_data = term_to_cui_map.get(s)
            o_data = term_to_cui_map.get(o)
            
            # Construct mapped triple structure:
            # ( (Subj_Name, Subj_CUI), Relation, (Obj_Name, Obj_CUI) )
            new_s = (s, s_data['cui']) if s_data else (s, None)
            new_o = (o, o_data['cui']) if o_data else (o, None)
            
            mapped_triples.append((new_s, r, new_o))
        return mapped_triples

    tqdm.pandas(desc="Mapping Rows")
    df['cui_triples'] = df['parsed_triples'].progress_apply(map_row)

    # --- FIX STARTS HERE ---
    print("Serializing complex column to string to satisfy PyArrow...")
    # This converts the complex list-of-tuples into a string format 
    # to prevent "ArrowInvalid" schema errors.
    df['cui_triples'] = df['cui_triples'].astype(str)
    # --- FIX ENDS HERE ---

    # 6. Save
    print("Saving final results...")
    # Drop intermediate column
    df_final = df.drop(columns=['parsed_triples'])
    df_final.to_parquet(OUTPUT_FILE)
    print(f"Done! Saved to {OUTPUT_FILE}")

    
if __name__ == "__main__":
    main()