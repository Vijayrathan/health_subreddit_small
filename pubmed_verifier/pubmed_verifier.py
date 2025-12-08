import requests
import time
import ast
from datasets import load_from_disk
from tqdm import tqdm

class PubMedMeshVerifier:
    def __init__(self, email):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        # Use a session for connection pooling (faster/stable)
        self.session = requests.Session()

    def get_count(self, query):
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "email": self.email,
            "retmax": 0
        }
        try:
            r = self.session.get(self.base_url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            return int(data["esearchresult"]["count"])
        except Exception as e:
            return 0

    def verify_claim(self, subject, user_predicate, object_term):
        # 1. Subject IS the Therapy
        query_therapy = (
            f'("{subject}"[Title] AND "therapeutic use"[sh]) AND "{object_term}"[Title/Abstract]'
        )
        
        # 2. Subject IS the Cause/Harm
        query_harm = (
            f'("{subject}"[Title] AND ("adverse effects"[sh] OR "toxicity"[sh] OR "etiology"[sh] OR "chemically induced"[sh])) '
            f'AND "{object_term}"[Title/Abstract]'
        )

        count_therapy = self.get_count(query_therapy)
        time.sleep(0.34) # Rate limit
        count_harm = self.get_count(query_harm)
        
        total = count_therapy + count_harm

        if total == 0:
            return "ambiguous" # No data

        # Calculate Ratio (Therapy / Harm)
        # Handle division by zero safely by setting a high ceiling
        if count_harm == 0:
            ratio = 999.0 # Effectively infinite positive signal
        else:
            ratio = count_therapy / count_harm

        is_positive_claim = user_predicate.lower() in ["treats", "cures", "helps", "heals", "prevents"]

        # --- ADJUSTED THRESHOLDS ---
        if is_positive_claim:
            # User says "It Cures"
            if ratio >= 2.0: 
                return "true"
            elif ratio <= 0.5:
                return "false" # Evidence actually points to harm
            else:
                return "ambiguous" # Ratio between 0.5 and 2.0 (Controversial/Mixed)
        else:
            # User says "It Causes" (Negative intent)
            if ratio <= 0.5:
                return "true" # Low therapy / High harm = True for "Causes"
            elif ratio >= 2.0:
                return "false" # High therapy / Low harm = False for "Causes"
            else:
                return "ambiguous" 

# --- EXECUTION ---
MY_EMAIL = "your_email@example.com"
verifier = PubMedMeshVerifier(MY_EMAIL)

print("Loading dataset: triples_output_cleaned")
dataset = load_from_disk("triples_output_cleaned")

claims = []
total_triples = 0

pbar = tqdm(enumerate(dataset), total=len(dataset))
for idx, row in pbar:
    try:
        triples_list = ast.literal_eval(row["triples"])
    except:
        claims.append("na")
        continue
    
    if not isinstance(triples_list, list) or len(triples_list) == 0:
        claims.append("na")
        continue
    
    triple_verdicts = []
    for triple in triples_list:
        if len(triple) != 3: continue
        subject, predicate, object_term = triple
        
        # Filter: Skip generic triples that aren't medical claims
        # This saves API calls and reduces noise
        if predicate not in ["treats", "cures", "helps", "heals", "causes", "worsens", "induces"]:
            continue

        total_triples += 1
        verdict = verifier.verify_claim(subject, predicate, object_term)
        triple_verdicts.append(verdict)
        time.sleep(0.35) 
    
    # --- BETTER AGGREGATION LOGIC ---
    if not triple_verdicts:
        claims.append("na")
    elif "true" in triple_verdicts and "false" not in triple_verdicts:
        # If at least one medical triple is True and none are False -> TRUE
        claims.append("true")
    elif "false" in triple_verdicts:
        # If any medical triple is demonstrably False -> FALSE (Safety priority)
        claims.append("false")
    else:
        # All ambiguous, or mix of ambiguous/na
        claims.append("ambiguous")
    
    pbar.set_postfix({"triples": total_triples})

print("Saving...")
dataset = dataset.add_column("claim", claims)
dataset.save_to_disk("triples_output_cleaned_verified")
df=dataset.to_pandas()
df['claim'].value_counts()