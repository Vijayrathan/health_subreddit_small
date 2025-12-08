import pandas as pd
import ast
import gc
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "triples_cui_mapped.parquet"
# Ensure this matches your actual extracted filename
SEMMED_FILE = "semmedVER43_2024_R_PREDICATION.csv" 
OUTPUT_FILE = "triples_semmed_verified.parquet"

RELATION_MAP = {
    "treats": {"TREATS", "PREVENTS", "DISRUPTS", "INHIBITS"},
    "prevents": {"PREVENTS", "INHIBITS", "DISRUPTS", "TREATS"},
    "causes": {"CAUSES", "PRODUCES", "PRECIPITATES", "STIMULATES"},
    "worsens": {"AGGRAVATES", "COMPLICATES", "WORSENS", "EXACERBATES", "STIMULATES", "AUGMENTS"},
    "improves": {"TREATS", "AMELIORATES", "PREVENTS"},
    "manifests_as": {"MANIFESTATION_OF", "DIAGNOSES"}, 
    "is_associated_with": {"ASSOCIATED_WITH", "COEXISTS_WITH"} 
}
GENERIC_RELATIONS = {"ASSOCIATED_WITH", "RELATED_TO", "COEXISTS_WITH"}

def get_relevant_cuis(df):
    """
    Extracts a set of all CUIs present in your input data.
    Used to filter SemMedDB so we don't load unnecessary data.
    """
    print("Extracting relevant CUIs from input data...")
    relevant_cuis = set()
    
    # parse if string
    if isinstance(df['cui_triples'].iloc[0], str):
        triples_col = df['cui_triples'].apply(ast.literal_eval)
    else:
        triples_col = df['cui_triples']

    for triples_list in tqdm(triples_col, desc="Scanning CUIs"):
        if not isinstance(triples_list, list): continue
        for triple in triples_list:
            # Triple format: ((SubjName, SubjCUI), Rel, (ObjName, ObjCUI))
            s_cui = triple[0][1]
            o_cui = triple[2][1]
            if s_cui: relevant_cuis.add(s_cui)
            if o_cui: relevant_cuis.add(o_cui)
            
    print(f"Found {len(relevant_cuis)} unique CUIs to verify.")
    return relevant_cuis

def load_filtered_semmeddb(path, relevant_cuis):
    """
    Loads SemMedDB using column INDICES and LATIN1 encoding.
    """
    print(f"Loading and filtering SemMedDB from {path}...")
    
    semmed_edges = {}
    chunksize = 1_000_000
    
    try:
        reader = pd.read_csv(
            path, 
            sep=',',              
            quotechar='"',        
            header=None,          
            # 3=PREDICATE, 4=SUBJECT_CUI, 8=OBJECT_CUI
            usecols=[3, 4, 8],    
            names=["PREDICATE", "SUBJECT_CUI", "OBJECT_CUI"], 
            dtype=str,
            chunksize=chunksize,
            on_bad_lines='skip',
            encoding='latin1'  # <--- FIXED: Handles special chars like 'ö' or 'é'
        )

        for chunk in tqdm(reader, desc="Filtering SemMedDB Chunks"):
            # Filter: Keep row IF Subject OR Object is in our relevant list
            mask = chunk['SUBJECT_CUI'].isin(relevant_cuis) & chunk['OBJECT_CUI'].isin(relevant_cuis)
            filtered_chunk = chunk[mask]
            
            if filtered_chunk.empty:
                continue

            for s, p, o in filtered_chunk[['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']].values:
                if pd.isna(s) or pd.isna(o) or pd.isna(p):
                    continue
                    
                key = (s, o)
                if key not in semmed_edges:
                    semmed_edges[key] = set()
                semmed_edges[key].add(p)
        
        print(f"Loaded {len(semmed_edges)} relevant edges from SemMedDB.")
        return semmed_edges

    except FileNotFoundError:
        print("ERROR: SemMedDB file not found.")
        return {}

def verify_claim(triple, semmed_edges):
    try:
        subj_node, relation, obj_node = triple
        s_cui, o_cui = subj_node[1], obj_node[1]
        
        if not s_cui or not o_cui: return "NO_CUI_MATCH"

        direct_predicates = semmed_edges.get((s_cui, o_cui), set())
        valid_semmed_targets = RELATION_MAP.get(relation.lower(), GENERIC_RELATIONS)

        # 1. Exact Semantic Match
        if not direct_predicates.isdisjoint(valid_semmed_targets):
            return "SUPPORTED"
            
        # 2. Ambiguous Match (Right CUI link, wrong verb)
        if direct_predicates:
            return "AMBIGUOUS_SUPPORT"

        # 3. Inverse Match (Subject/Object swapped)
        inverse_predicates = semmed_edges.get((o_cui, s_cui), set())
        if inverse_predicates:
             return "INVERSE_SUPPORT"

        return "UNVERIFIED"
    except Exception:
        return "ERROR"

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 1. Get List of CUIs we actually care about
    relevant_cuis = get_relevant_cuis(df)
    
    # 2. Load SemMedDB (Filtered)
    semmed_edges = load_filtered_semmeddb(SEMMED_FILE, relevant_cuis)
    
    if not semmed_edges:
        print("Warning: No edges loaded. Check file path or CUI format.")
        # Proceeding anyway will result in all UNVERIFIED, which is handled.

    # 3. Parsing Structure
    print("Parsing structure for verification...")
    if isinstance(df['cui_triples'].iloc[0], str):
        tqdm.pandas(desc="Parsing AST")
        df['cui_triples_struct'] = df['cui_triples'].progress_apply(ast.literal_eval)
    else:
        df['cui_triples_struct'] = df['cui_triples']

    # 4. Verification Loop
    def process_row_verification(triples_list):
        if not isinstance(triples_list, list): return []
        verified_output = []
        for trip in triples_list:
            status = verify_claim(trip, semmed_edges)
            flat_triple = {
                "subject": trip[0][0],
                "subject_cui": trip[0][1],
                "relation": trip[1],
                "object": trip[2][0],
                "object_cui": trip[2][1],
                "claim_status": status
            }
            verified_output.append(flat_triple)
        return verified_output

    print("Verifying claims...")
    tqdm.pandas(desc="Verifying")
    df['claim_verification'] = df['cui_triples_struct'].progress_apply(process_row_verification)

    df['claim_verification'] = df['claim_verification'].astype(str)
    
    cols_to_keep = ['cui_triples', 'claim_verification']
    df_final = df[cols_to_keep]
    
    df_final.to_parquet(OUTPUT_FILE)
    print(f"Done! Verified data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()