import os
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" # Or your Qwen model
DATASET_NAME = "triples_output_100k.parquet"
OUTPUT_FILE = "reddit_health_ground_truth.parquet"

# --- The Aligned Judge Prompt ---
JUDGE_SYSTEM_PROMPT = """You are an expert medical fact-checker. Evaluate the health claims in the Reddit comment based on scientific consensus.

Classify the comment into exactly one of these categories (Use these EXACT terms):

1. **SUPPORTED**
   - Use this for strict, causal, or treatment claims that are scientifically established.
   - ALSO use this for personal anecdotes IF they illustrate a widely accepted biological fact (e.g., "I stopped eating sugar and lost weight" -> SUPPORTED, because caloric deficit causes weight loss).
   - Examples: "Insulin treats diabetes", "Running helped me lose weight".

2. **AMBIGUOUS_SUPPORT**
   - Use this for claims that are biologically linked but vague, or for product claims (e.g., specific shampoos, mattresses).
   - Examples: "This cream helped my acne", "Stress might be why I'm shedding".

3. **UNVERIFIED**
   - Use this for purely subjective opinions without biological basis, questions, fiction, or obviously false info.
   - Examples: "I hate doctors", "My dog is cute", "Zombie virus story".

**Output Format:**
Label | Explanation
"""

def format_judge_prompt(text):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{JUDGE_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nComment: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nVerdict:"

def has_triples(val):
    if val is None: return False
    s_val = str(val).strip()
    return s_val != "[]" and s_val != "" and s_val != "None"

def main():
    print(f"Loading dataset: {DATASET_NAME}...")
    
    # 1. Filter Data
    df = pd.read_parquet(DATASET_NAME)

    # Filter for rows that actually have triples to verify
    df_filtered = df[df['triples'].apply(has_triples)].copy()
    print(f"Generating Ground Truth for {len(df_filtered)} rows...")

    # 2. Initialize Model (H200 Config)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95, 
        max_model_len=8192,
        dtype="float16",
        quantization="awq",
        trust_remote_code=True
    )

    # 3. Prepare Prompts
    clean_texts = [str(t).replace("\n", " ") for t in df_filtered['body']]
    prompts = [format_judge_prompt(t) for t in clean_texts]

    # 4. Inference
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64, 
        stop=["<|eot_id|>", "\n\n"]
    )
    
    print("Running Inference...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Parse Labels
    labels = []
    verdicts = []
    
    print("Parsing verdicts...")
    for o in outputs:
        text_out = o.outputs[0].text.strip()
        verdicts.append(text_out)
        upper_text = text_out.upper().replace("*", "")
        # Strict Label Matching
        upper_text = text_out.upper()
        if "AMBIGUOUS_SUPPORT" in upper_text:
            labels.append("AMBIGUOUS_SUPPORT")
        elif "SUPPORTED" in upper_text:
            # Check ambiguous again just in case (e.g. "NOT SUPPORTED")
            if "NOT SUPPORTED" in upper_text:
                labels.append("UNVERIFIED")
            else:
                labels.append("SUPPORTED")
        else:
            labels.append("UNVERIFIED")

    # 6. Save
    df_filtered['ground_truth_label'] = labels
    df_filtered['ground_truth_explanation'] = verdicts

    # --- FINAL CLEANUP BEFORE SAVE ---
    # 1. Deduplicate columns (in case 'body' or 'id' appeared twice)
    df_filtered = df_filtered.loc[:, ~df_filtered.columns.duplicated()]
    
    # 2. Drop any new artifacts
    cols_to_drop_final = [c for c in df_filtered.columns if c.startswith("__")]
    if cols_to_drop_final:
        df_filtered = df_filtered.drop(columns=cols_to_drop_final)

    # 3. Reset index one last time
    df_filtered = df_filtered.reset_index(drop=True)
    
    print(f"Saving to {OUTPUT_FILE}...")
    # Using index=False is crucial to prevent PyArrow from creating the __index_level_0__ column again
    df_filtered.to_parquet(OUTPUT_FILE, index=False)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()