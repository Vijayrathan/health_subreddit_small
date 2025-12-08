import os
import gc
import re
from datasets import load_from_disk
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-8B"  # Updated to your specific model
DATASET_PATH = "reddit-health-86k" 
OUTPUT_FILE = "triples_output_100k.parquet"
INFERENCE_BATCH_SIZE = 20000 # Save every 20k rows to manage RAM

# --- Final Optimized Prompt ---
SYSTEM_PROMPT = """You are an expert medical information extraction system. Your task is to extract health-related knowledge triples from informal text for a clinical Knowledge Graph.

Goal: Extract (subject, relation, object) triples.

Rules:
1.  **Standardize Terminology:**
    * Convert slang or colloquialisms into standard medical/scientific terms if obvious.
    * Example: "high blood sugar" -> "Hyperglycemia"
    * Example: "belly fat" -> "Abdominal obesity"
    
2.  **Health Scope Only:**
    * **Subjects:** Medications, chemical compounds, foods, lifestyle factors (smoking, exercise), pathogens.
    * **Objects:** Diseases, symptoms, physiological processes, anatomy, side effects.

3.  **Atomic Entities:**
    * Keep entities short and atomic (1-3 words).
    * Avoid full sentences or "context" inside the entity string.

4.  **Verbs:** Use these specific relations if they apply: 
    * treats, causes, prevents, manifests_as, worsens, improves, is_associated_with

5.  **Format:** Output ONLY a valid Python list of tuples: [("Subject", "Relation", "Object"), ...]. Return [] if no claims exist.

6. **Strict Grounding:** Extract ONLY information explicitly stated in the text. Do NOT use your internal medical knowledge to diagnose or add symptoms not mentioned.
   - Bad: Text says "dog is shedding" -> Output: [("Shedding", "caused_by", "Hypothyroidism")]
   - Good: Text says "dog is shedding" -> Output: [] (if no cause is stated)

Examples:

Input: "My doctor said my pot belly is putting me at risk for sugar sickness."
Output: [("Abdominal obesity", "is_risk_factor_for", "Diabetes")]

Input: "I started taking magnesium glycinate and it totally wiped out my insomnia."
Output: [("Magnesium glycinate", "treats", "Insomnia")]

Input: "Drinking too much coffee makes me jittery."
Output: [("Coffee", "causes", "Tremor"), ("Coffee", "causes", "Anxiety")]

Input: "I hate the taste of kale."
Output: []

Input: """

def format_prompt(text):
    return f"{SYSTEM_PROMPT} \"{text}\"\nOutput:"

def post_process_text(text):
    """Clean the raw generation output."""
    # 1. Safety net: Remove Qwen 'thinking' tags if they ever appear
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    text = text.strip()
    
    # 2. Fix missing brackets due to stop token truncation
    if not text.endswith("]") and text.strip() != "":
         text += "]"
    if not text.startswith("["):
        text = "[]"
    return text

def main():
    print(f"Loading dataset from: {DATASET_PATH}...")
    ds = load_from_disk(DATASET_PATH)
    # ds=ds.select(range(100))
    print(f"Total rows to process: {len(ds)}")

    # 1. Format Prompts
    print("Formatting prompts...")
    clean_text = [str(t).replace("\n", " ") for t in tqdm(ds["body"], desc="Formatting")]
    prompts = [format_prompt(t) for t in clean_text]

    # 2. Initialize vLLM
    print(f"Initializing vLLM with {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        dtype="float16",
        trust_remote_code=True
    )

    # 3. Sampling Params (The Winning Config)
    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=128,
        stop=["<|endoftext|>", "<|im_end|>", "]", "\nInput:", "Input:", "\n\n"],
        repetition_penalty=1.05 
    )

    # 4. Run Inference in Batches
    print(f"Starting inference...")
    
    all_generated_texts = []
    total_batches = (len(prompts) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    
    for i in tqdm(range(0, len(prompts), INFERENCE_BATCH_SIZE), total=total_batches, desc="Inference Batches"):
        batch_prompts = prompts[i : i + INFERENCE_BATCH_SIZE]
        
        # Run vLLM
        batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        
        # Clean text
        for output in batch_outputs:
            all_generated_texts.append(post_process_text(output.outputs[0].text))
            
        # RAM Cleanup
        del batch_outputs
        gc.collect()

    # 5. Save Final Result
    print("Combining results and saving...")
    df = ds.to_pandas()
    df["triples"] = all_generated_texts
    
    df.to_parquet(OUTPUT_FILE)
    print(f"Success! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()