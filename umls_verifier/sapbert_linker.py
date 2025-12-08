import os
import sys
import torch
import faiss
import numpy as np
import spacy
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
# Registers the factory
from scispacy.linking import EntityLinker 

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BATCH_SIZE = 1024       # GPU Batch size
INDEX_CHUNK_SIZE = 50000 # RAM Batch size
TOP_K = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FAISS CONFIGURATION (COMPRESSION)
# IVF1024 = Faster search (splits space into 1024 clusters)
# PQ64 = Compresses vectors to 64 bytes each (Huge RAM saving)
FAISS_INDEX_STRING = "IVF1024,PQ64" 

print(f"Running on: {DEVICE}")

# ------------------------------------------------------------------
# 1. LOAD RESOURCES
# ------------------------------------------------------------------
def load_resources():
    print("\n--- STEP 1: LOADING RESOURCES ---")
    print("1. Loading SciSpaCy (for UMLS dictionary)...")
    try:
        nlp = spacy.load("en_core_sci_lg")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        linker = nlp.get_pipe("scispacy_linker")
        kb = linker.kb
        print(f"   ✅ UMLS KB Loaded. Total concepts: {len(kb.cui_to_entity)}")
    except Exception as e:
        print(f"   ❌ Failed to load SciSpaCy Linker: {e}")
        sys.exit(1)

    print("2. Loading SapBERT Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("   ✅ SapBERT Loaded.")

    return kb, tokenizer, model

# ------------------------------------------------------------------
# 2. HELPER: ENCODE BATCH
# ------------------------------------------------------------------
def encode_batch(texts, tokenizer, model):
    """Encodes a list of texts into numpy vectors."""
    try:
        encoded_input = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=64, 
            return_tensors='pt'
        ).to(DEVICE)

        with torch.no_grad():
            output = model(**encoded_input)
            cls_emb = output.last_hidden_state[:, 0, :]
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
            return cls_emb.cpu().numpy()
            
    except RuntimeError as e:
        # Fallback for individual bad strings if batch fails
        if "CUDA out of memory" in str(e):
            print("❌ CUDA OOM in batch encoding. Reduce BATCH_SIZE.")
            sys.exit(1)
        print(f"⚠️ Warning: Batch encoding failed ({e}). Retrying individually...")
        valid_vecs = []
        for t in texts:
            try:
                # Recursive call for single item
                v = encode_batch([t], tokenizer, model)
                valid_vecs.append(v)
            except Exception:
                # If a single string is corrupted/bad, skip it (fill with zeros)
                print(f"   Skipping bad string: {t[:20]}...")
                valid_vecs.append(np.zeros((1, 768), dtype=np.float32))
        return np.vstack(valid_vecs)

# ------------------------------------------------------------------
# 3. BUILD QUANTIZED INDEX
# ------------------------------------------------------------------
def build_index_quantized(kb, tokenizer, model):
    print("\n--- STEP 2: INDEXING UMLS (QUANTIZED) ---")
    
    # A. Extract strings
    print("Extracting names and aliases...")
    all_names = []
    all_cuis = []
    for cui, entity in tqdm(kb.cui_to_entity.items(), desc="Reading KB"):
        all_names.append(entity.canonical_name)
        all_cuis.append(cui)
        for alias in entity.aliases:
            all_names.append(alias)
            all_cuis.append(cui)

    total_items = len(all_names)
    print(f"Total phrases to index: {total_items}")

    # B. TRAIN THE INDEX
    # Quantized indexes (PQ) need to "learn" the distribution of vectors first.
    # We take a random sample of 25,000 items to train.
    print("Training FAISS Index (Sampling 25k items)...")
    
    # Take random sample
    train_size = min(25000, total_items)
    train_indices = np.random.choice(total_items, train_size, replace=False)
    train_texts = [all_names[i] for i in train_indices]
    
    # Encode training set
    train_vectors = []
    for i in tqdm(range(0, len(train_texts), BATCH_SIZE), desc="Encoding Train Set"):
        batch = train_texts[i : i + BATCH_SIZE]
        vecs = encode_batch(batch, tokenizer, model)
        train_vectors.append(vecs)
    train_matrix = np.vstack(train_vectors)

    # Initialize Index
    d = 768
    index = faiss.index_factory(d, FAISS_INDEX_STRING, faiss.METRIC_INNER_PRODUCT)
    
    # Train
    index.train(train_matrix)
    print("✅ Index Trained.")
    
    # Clear memory
    del train_matrix
    del train_vectors
    gc.collect()

    # C. ADD ALL VECTORS IN CHUNKS
    print("Indexing all items...")
    
    for i in tqdm(range(0, total_items, INDEX_CHUNK_SIZE), desc="Indexing Chunks"):
        chunk_texts = all_names[i : i + INDEX_CHUNK_SIZE]
        
        chunk_embeddings = []
        for j in range(0, len(chunk_texts), BATCH_SIZE):
            batch_texts = chunk_texts[j : j + BATCH_SIZE]
            vecs = encode_batch(batch_texts, tokenizer, model)
            chunk_embeddings.append(vecs)
        
        if not chunk_embeddings: continue
        chunk_matrix = np.vstack(chunk_embeddings)
        
        # Add to FAISS
        index.add(chunk_matrix)
        
        # Free memory
        del chunk_matrix
        del chunk_embeddings
        gc.collect()
        
    print(f"   ✅ Index built with {index.ntotal} vectors.")
    return index, all_names, all_cuis

# ------------------------------------------------------------------
# 4. PROCESS DATASET
# ------------------------------------------------------------------
def process_dataset(index, all_cuis, all_names, tokenizer, model, kb):
    print("\n--- STEP 3: LINKING DATASET ---")
    
    ds = load_dataset("Vijayrathank/reddit-health-small", split="train")
    new_rows = []
    
    # Increase nprobe for better accuracy in IVFPQ index
    index.nprobe = 10 

    def resolve_entity(text):
        if not text: return None, None
        
        vec = encode_batch([text], tokenizer, model) # returns (1, 768)
        
        distances, indices = index.search(vec, TOP_K)
        
        best_idx = indices[0][0]
        score = distances[0][0]
        
        # Note: Inner Product scores are roughly Cosine Sim. 
        # 0.65 is a safe threshold.
        if score < 0.65 or best_idx == -1: 
            return None, None
            
        cui = all_cuis[best_idx]
        canonical_name = kb.cui_to_entity[cui].canonical_name
        return cui, canonical_name

    for row in tqdm(ds, desc="Linking Triples"):
        triples = row.get("triples", [])
        normalized_triples = []
        
        if triples:
            for t in triples:
                if not t or len(t) != 3: continue
                subj, rel, obj = t
                
                s_cui, s_name = resolve_entity(subj)
                o_cui, o_name = resolve_entity(obj)
                
                if s_cui or o_cui:
                    normalized_triples.append({
                        "subject_name": s_name if s_name else subj,
                        "subject_cui": s_cui,
                        "relation": rel,
                        "object_name": o_name if o_name else obj,
                        "object_cui": o_cui
                    })
        
        out_row = row.copy()
        out_row['umls_triples'] = normalized_triples
        new_rows.append(out_row)
        
    return new_rows

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    kb, tokenizer, model = load_resources()
    
    # Use the QUANTIZED builder
    index, all_names, all_cuis = build_index_quantized(kb, tokenizer, model)
    
    processed_data = process_dataset(index, all_cuis, all_names, tokenizer, model, kb)
    
    print("\n--- STEP 4: SAVING ---")
    final_ds = Dataset.from_list(processed_data)
    save_path = "reddit-health-small-sapbert-full"
    final_ds.save_to_disk(save_path)
    print(f"✅ Saved to: {save_path}")