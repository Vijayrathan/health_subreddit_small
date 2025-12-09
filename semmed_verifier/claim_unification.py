import ast
import pandas as pd
def get_unified_status(verification_list):
    """
    Aggregates micro-claims into a trinary macro-label for the comment.
    Priority: SUPPORTED > AMBIGUOUS_SUPPORT > UNVERIFIED
    """
    verification_list= ast.literal_eval(verification_list)
    # 1. Handle empty/None
    if not isinstance(verification_list, list) or not verification_list:
        return "UNVERIFIED"
    
    # 2. Extract all statuses in this comment
    statuses = set()
    for item in verification_list:
        # Get status, default to unverified if missing
        s = item.get('claim_status', 'UNVERIFIED')
        statuses.add(s)
        
    # 3. Hierarchy Check
    
    # Priority 1: If ANY verified fact exists, the comment is Supported.
    if 'SUPPORTED' in statuses:
        return "SUPPORTED"
        
    # Priority 2: If no strict support, but ambiguous/inverse evidence exists.
    # We group INVERSE_SUPPORT here because it indicates a strong biological link 
    # (just the wrong direction), which fits "Ambiguous" better than "Unverified".
    if 'AMBIGUOUS_SUPPORT' in statuses or 'INVERSE_SUPPORT' in statuses:
        return "AMBIGUOUS_SUPPORT"
        
    # Priority 3: Fallback
    return "UNVERIFIED"

if __name__ == "__main__":
    df_semmed= pd.read_parquet("triples_semmed_verified.parquet")
    df_semmed['unified_claim_status'] = df_semmed['claim_verification'].apply(get_unified_status)
    df_semmed.to_parquet("triples_semmed_unified_claims.parquet")
