import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import sys
from datasets import load_from_disk

# 1. CONFIGURATION
MODEL_ID = "Qwen/Qwen3-8B"

OUTPUT_FILE = "triples_output"
BATCH_SIZE = 32  # Safe starting point for A100
current_batch = 0

def main():
    print(f"--- Starting Extraction on {torch.cuda.get_device_name(0)} ---")

    # 2. LOAD MODEL (A100 Optimized)
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,    # Native A100 format
        device_map="auto",
        attn_implementation="sdpa",    # Stable & Fast
    )
    model.eval() # Ensure model is in eval mode
    print(model.device)
    # 3. LOAD DATASET
    # Replace this with however you load your data (e.g., from disk or hub)
    # ds = load_dataset(INPUT_DATASET, split="train") 
    # For testing, let's assume 'ds' exists. If loading from a file:
    # ds = Dataset.from_file("path/to/your/data.arrow")
    
    

    ds = load_from_disk("reddit-health-cleaned")
    ds = Dataset.from_pandas(ds.to_pandas())
    print(f"Dataset Size: {len(ds)} rows")  # <--- ADD THIS
    if len(ds) == 0:
        print("ERROR: Dataset is empty. Check your loading path!")
        return

    # 2. VERIFY COLUMN NAME
    print(f"Columns: {ds.column_names}")    # <--- ADD THIS
    if "clean_text" not in ds.column_names:
        print("ERROR: Column 'clean_text' not found. Update the script to use the correct column name.")
        return
    # ----------------------------------------------------------------

    # 4. PROCESSING FUNCTION
    def batch_process(batch):
        global current_batch
        
        current_batch += 1
        
        # --- THE ONLY LOG YOU ASKED FOR ---
        print(f"Processing Batch: {current_batch}", flush=True)

        system_msg = """
You are an information extraction system designed to transform informal Reddit comments into factual knowledge triples.

Your task: Extract all factual or causal claim triples from the given text in the format:
(subject, relation, object)

Guidelines:
- Focus only on factual, scientific, or health-related claims (not opinions or emotions).
- Each triple must represent a single relationship that can, in principle, be verified.
- Use simple canonical verbs like: treats, causes, prevents, leads_to, associated_with, contains, increases, decreases, helps_with, used_for, etc.
- Avoid non-informative relations such as “is”, “have”, “make”, “thing”.
- Combine multi-word entities where appropriate (e.g., “vitamin D deficiency”, “blood pressure”).
- If no factual claim is found, return an empty list.
- Output strictly as tuple in this format:
[
  (subject, relation, object),
  ...
]
"""
        
        prompts = [
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSentence: {text}\n\nTriples:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            for text in batch["clean_text"]
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Slice off the prompt
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return {"triples": decoded}

    # 5. EXECUTE BATCH
    print("Starting Batch Inference...")

    # processed_ds = ds.map(
    # batch_process,
    # batched=True,
    # batch_size=BATCH_SIZE,
    # writer_batch_size=1000,
    # load_from_cache_file=False,
    # num_proc=1    # prevents multiprocessing deadlock
    # )
   
    triples = []
    for i in range(0, len(ds), BATCH_SIZE):
        batch = ds[i : i + BATCH_SIZE]
        out = batch_process(batch)
        triples.extend(out["triples"])

    print("Done! Writing dataset...")
    ds = ds.add_column("triples", triples)
    ds.save_to_disk(OUTPUT_FILE)

    
if __name__ == "__main__":
    main()