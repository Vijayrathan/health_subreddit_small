import requests
import re

class UMLSHybridVerifier:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_uri = "https://uts-ws.nlm.nih.gov/rest"
        self.version = "current"
        
        # Mappings for logic
        self.positive_verbs = ["treats", "cures", "helps", "prevents", "heals", "ameliorates"]
        self.negative_verbs = ["causes", "worsens", "induces", "aggravates"]
        
        self.umls_positive_rels = ["may_treat", "treats", "prevents", "therapeutic_class_of"]
        self.umls_negative_rels = ["causes", "induces", "associated_with", "finding_site_of", "cause_of"]

    def get_top_cui(self, term):
        """
        Get the single most likely CUI.
        """
        endpoint = f"{self.base_uri}/search/{self.version}"
        params = {
            "string": term,
            "apiKey": self.api_key,
            "searchType": "words",
            "returnIdType": "concept"
        }
        try:
            r = requests.get(endpoint, params=params)
            r.raise_for_status()
            data = r.json()
            if data["result"]["results"]:
                # Return top result only for cleaner logic in this hybrid approach
                top = data["result"]["results"][0]
                return top["ui"], top["name"]
            return None, None
        except Exception:
            return None, None

    def check_relations(self, cui_s, cui_o):
        """
        Check structured API relations (The "Hard" link).
        """
        endpoint = f"{self.base_uri}/content/{self.version}/CUI/{cui_s}/relations"
        params = {"apiKey": self.api_key, "includeRelationLabels": "true"}
        
        try:
            r = requests.get(endpoint, params=params)
            if r.status_code != 200: return []
            
            data = r.json()
            found_rels = []
            for item in data["result"]:
                if cui_o in item["relatedId"]:
                    found_rels.append(item.get("relationLabel", "RO"))
            return found_rels
        except:
            return []

    def check_definitions(self, cui_s, object_term):
        """
        Check unstructured text definitions (The "Soft" link).
        Fetches definitions of Subject and looks for Object keyword.
        """
        endpoint = f"{self.base_uri}/content/{self.version}/CUI/{cui_s}/definitions"
        params = {"apiKey": self.api_key}
        
        try:
            r = requests.get(endpoint, params=params)
            if r.status_code != 200: return False, None
            
            data = r.json()
            definitions = [d["value"] for d in data["result"]]
            
            # Simple regex search for the object term inside the subject's definition
            # e.g., Search "Headache" inside definition of "Aspirin"
            for definition in definitions:
                if re.search(r'\b' + re.escape(object_term) + r'\b', definition, re.IGNORECASE):
                    return True, definition
            
            return False, None
        except:
            return False, None

    def verify(self, subject, predicate, object_term):
        print(f"--- Verifying: {subject} -> {predicate} -> {object_term} ---")
        
        # 1. Entity Linking
        cui_s, name_s = self.get_top_cui(subject)
        cui_o, name_o = self.get_top_cui(object_term)
        
        if not cui_s or not cui_o:
            print("  [!] Entities not found in UMLS.")
            return

        print(f"  Mapped: {name_s} ({cui_s}) / {name_o} ({cui_o})")

        # 2. Check Structured Relations (Metathesaurus)
        rels = self.check_relations(cui_s, cui_o)
        
        if rels:
            print(f"  [+] STRUCTURED MATCH: Found relations {rels}")
            print("  => VERDICT: PROVEN (via Knowledge Graph Relations)")
            print("\n")
            return

        # 3. Check Definitions (Text Mining)
        print("  [i] No structured relation found. Scanning text definitions...")
        found_in_def, text = self.check_definitions(cui_s, object_term)
        
        if found_in_def:
            print(f"  [+] TEXT MATCH: Found '{object_term}' in definition of '{subject}'.")
            print(f"      Snippet: \"...{text[:100]}...\"")
            print("  => VERDICT: LIKELY TRUE (via Definition Text)")
        else:
            # Try Inverse Definition (Look for Subject in Object's definition)
            found_in_inv, text_inv = self.check_definitions(cui_o, subject)
            if found_in_inv:
                print(f"  [+] TEXT MATCH: Found '{subject}' in definition of '{object_term}'.")
                print(f"      Snippet: \"...{text_inv[:100]}...\"")
                print("  => VERDICT: LIKELY TRUE (via Definition Text)")
            else:
                print("  [-] No evidence found in Relations or Definitions.")
                print("  => VERDICT: AMBIGUOUS / UNPROVEN")
        print("\n")

# --- EXECUTION ---
MY_API_KEY = "a7ecd867-5bdc-40a2-8eba-56f52041837c" 

verifier = UMLSHybridVerifier(MY_API_KEY)

# 1. This usually fails structured, but might pass via text definition
verifier.verify("Smoking", "causes", "Cancer")

# 2. This often works via definition ("used to treat headache")
verifier.verify("Aspirin", "treats", "Headache")

# 3. This will likely fail both, which is the CORRECT scientific result (it's unproven)
verifier.verify("Moringa", "helps", "Infertility")