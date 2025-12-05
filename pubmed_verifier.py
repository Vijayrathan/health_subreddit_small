import requests
import time

class PubMedMeshVerifier:
    def __init__(self, email):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    def get_count(self, query):
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "email": self.email,
            "retmax": 0
        }
        try:
            r = requests.get(self.base_url, params=params)
            r.raise_for_status()
            data = r.json()
            return int(data["esearchresult"]["count"])
        except:
            return 0

    def verify_claim(self, subject, user_predicate, object_term):
        print(f"--- Precision Verification: '{subject}' vs '{object_term}' ---")
        
        # --- QUERY CONSTRUCTION ---
        # We use [Title] for the subject to ensure it's the main topic.
        # We use [sh] (Subheading) to bind the role TO the subject.
        
        # 1. Hypothesis: Subject IS the Therapy
        # Logic: Find papers where "Subject" is the main topic AND is tagged as "Therapeutic Use"
        # AND "Object" appears anywhere in the abstract.
        query_therapy = (
            f'("{subject}"[Title] AND "therapeutic use"[sh]) AND "{object_term}"[Title/Abstract]'
        )
        
        # 2. Hypothesis: Subject IS the Cause/Harm
        # Logic: Find papers where "Subject" is the main topic AND is tagged as "Adverse Effects/Toxicity"
        # AND "Object" appears anywhere in the abstract.
        query_harm = (
            f'("{subject}"[Title] AND ("adverse effects"[sh] OR "toxicity"[sh] OR "etiology"[sh] OR "chemically induced"[sh])) '
            f'AND "{object_term}"[Title/Abstract]'
        )

        # --- EXECUTION ---
        count_therapy = self.get_count(query_therapy)
        time.sleep(0.35)
        count_harm = self.get_count(query_harm)
        
        total = count_therapy + count_harm
        
        print(f"  [+] 'Therapeutic' Tags:  {count_therapy}")
        print(f"  [-] 'Harm/Causality' Tags: {count_harm}")

        # --- VERDICT LOGIC ---
        if total == 0:
            print("  => VERDICT: UNKNOWN / NO INDEXED DATA")
            print("     (The subject might be too new or not yet indexed with MeSH)")
            print("\n")
            return

        # Calculate Ratio
        if count_harm == 0:
            print("  => VERDICT: TRUE (100% Positive Signal)")
            print("\n")
            return
            
        ratio = count_therapy / count_harm
        print(f"  [i] Therapeutic Ratio: {ratio:.4f}")

        is_positive_claim = user_predicate.lower() in ["treats", "cures", "helps", "heals"]

        if is_positive_claim:
            if ratio > 1.0:
                print("  => VERDICT: TRUE / SUPPORTED")
            elif ratio < 0.1:
                print("  => VERDICT: FALSE / DANGEROUS (Overwhelming evidence of harm)")
            else:
                print("  => VERDICT: CONTROVERSIAL (Evidence exists but harm is also significant)")
        else:
            # User claims "Causes" (Negative)
            if ratio < 0.2:
                print("  => VERDICT: TRUE / SUPPORTED")
            elif ratio > 1.0:
                print("  => VERDICT: FALSE (Actually acts as a therapy)")
            else:
                print("  => VERDICT: AMBIGUOUS")
        print("\n")

# --- EXECUTION ---
MY_EMAIL = "your_email@example.com"
verifier = PubMedMeshVerifier(MY_EMAIL)

# 1. Smoking -> Cancer
# This should now yield nearly ZERO therapeutic tags.
verifier.verify_claim("Smoking", "cures", "Cancer")

# 2. Aspirin -> Headache
# This should have high therapeutic tags.
verifier.verify_claim("Aspirin", "treats", "Headache")

# 3. Moringa -> Infertility
# This tests if Moringa is indexed as a drug or just a food.
verifier.verify_claim("Moringa", "helps", "Infertility")