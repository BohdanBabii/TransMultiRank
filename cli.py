# cli.py
from pathlib import Path
from itertools import islice
from cyclopts import App
from sentence_transformers import SentenceTransformer
import ir_datasets
import numpy as np
from tqdm import tqdm

app = App()

@app.command
def encode(
    dataset: str,               
    out_dir: Path,               
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased"
):
    """
    Encode documents from a dataset using a SentenceTransformer model.
    Args:
        dataset (str): The name of the dataset to load.
        out_dir (Path): The directory to save the encoded embeddings.
        model_name (str): The name of the SentenceTransformer model to use.
    """
    
    # 1) Daten laden
    ds = ir_datasets.load(dataset)   
    docs_iter = ds.docs_iter()
    docs_iter = islice(docs_iter, 250_000)

    # 2) Modell vorbereiten
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512

    # 3) Batch-Weise encoden
    batch_idx = 0
    while True:
        batch = list(islice(docs_iter, 50_000))
        if not batch:
            break
        doc_ids, texts = zip(*((d.doc_id, d.text) for d in batch))
        emb = model.encode(texts, batch_size=64, show_progress_bar=False)
        np.save(out_dir / f"embed_{batch_idx:03d}.npy",
                {"ids": doc_ids, "emb": emb},
                allow_pickle=True)
        batch_idx += 1
        tqdm.write(f"Saved batch {batch_idx} – {len(doc_ids)} docs")

if __name__ == "__main__":
    app() 
