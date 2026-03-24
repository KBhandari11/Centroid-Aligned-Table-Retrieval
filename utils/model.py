from transformers import AutoModel
import torch
import numpy as np
from sentence_transformers import SparseEncoder

from sentence_transformers import SentenceTransformer



def get_model(accelerator,model_name="reasonir"):
    if "splade" in model_name:
         model = SparseEncoder("naver/splade-v3")
    elif "reasonir" in model_name:
        device = accelerator.device
        model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
        model.eval()
        #model = model.to("cuda")
        #model.to(device)
        #model = accelerator.prepare(model)
    elif "rank1" in model_name:
        device = accelerator.device
        model = AutoModel.from_pretrained("jhu-clsp/rank1-7b", torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
        model.eval()
    elif "bge" in model_name:
        model = SentenceTransformer("BAAI/bge-m3", model_kwargs={"dtype": torch.float16,'device_map': "auto"},device="cuda",)
    elif "mpnet" in model_name:
        model = SentenceTransformer("all-mpnet-base-v2",device="cuda")
    elif "jina" in model_name:
        model = SentenceTransformer("jinaai/jina-embeddings-v3", model_kwargs={"dtype": torch.float16,'device_map': "auto"}, trust_remote_code=True)
    elif "bm25" in model_name:
        model = None  # BM25Okapi is non-neural
    return model

def encode_texts_in_batches(model_name,model, texts, instruction="", batch_size=16):
    """
    Encode a list of texts in batches, keep all embeddings on CPU.
    Ensures output shape [N, D], dtype float32, and strips NaN/Inf.
    """
    cleaned_texts = []
    for t in texts:
        # Ensure string
        if not isinstance(t, str):
            t = str(t)
        # Avoid truly empty strings
        t = t.strip()
        if t == "":
            t = "[EMPTY]"
        cleaned_texts.append(t)

    all_embs = []
    for start in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[start:start + batch_size]
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch).to(model.device)
        with torch.inference_mode():
            if model_name == "reasonir":
                emb = model.encode(batch, instruction=instruction)
            elif model_name == "bge" or model_name =="mpnet":
                emb = model.encode(batch)
            elif model_name == "jina":
                emb = model.encode(batch, task="retrieval.passage", prompt_name="retrieval.passage")
            elif model_name == "splade":
                emb = model.encode_document(batch)
        # Convert to torch tensor if needed
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        # Ensure 2D: [B, D]
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        # Clean NaN / Inf
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

        #print("\t\t",start, emb.shape, flush=True)
        # Move to CPU and store
        all_embs.append(emb.cpu())

        torch.cuda.empty_cache()
        #if accelerator is not None:
        #    accelerator.wait_for_everyone()  # all ranks sync here
    # [N, D]
    return torch.cat(all_embs, dim=0)