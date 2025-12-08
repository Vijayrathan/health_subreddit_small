from datasets import load_dataset
import pandas as pd
import re

# Precise keywords that strongly indicate misinformation/fringe theories
# (removed common words like "doctor" to avoid false positives from r/science)
keywords = [
    "mms", "chlorine dioxide", "miracle mineral", "black salve", "turpentine",
    "ivermectin", "fenbendazole", "hydroxychloroquine", "bioweapon", 
    "depopulation", "clot shot", "shedding", "graphene oxide", "spike protein",
    "fake news", "plandemic", "scamdemic", "big pharma mafia", "soros"
]

# Compile regex for speed (uses your CPU efficiency)
pattern = re.compile(r'\b(' + '|'.join([re.escape(k) for k in keywords]) + r')\b', re.IGNORECASE)

def is_misinfo_candidate(example):
    # Only check body length and keywords (No subreddit check = Much Faster)
    body = example['body']
    if len(body) < 30: return False
    return bool(pattern.search(body))

# Stream
ds = load_dataset("fddemarco/pushshift-reddit-comments", split="train", streaming=True)

# Filter
filtered = ds.filter(is_misinfo_candidate)

# Collect 100k
print("Streaming... (This will be much faster now)")
data = list(filtered.take(100000))

# Save
df = pd.DataFrame(data)
df.to_csv("health_misinfo_speed_run.csv", index=False)
print(f"Done! Collected {len(df)} rows.")