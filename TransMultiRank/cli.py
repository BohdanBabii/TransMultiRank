from pathlib import Path
from itertools import islice
from cyclopts import App
from cyclopts import run
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import ir_datasets
import numpy as np
from tqdm.auto import tqdm

app = App()

def getIterator(dataset: str, kind: str):
    ds = ir_datasets.load(dataset)
    match kind:
        case "docs":
            return ds.docs_iter()
        case "queries":
            return ds.queries_iter()
        case "qrels":
            return ds.qrels_iter()
        case "scoreddocs":
            return ds.scoreddocs_iter()
        case _:
            raise ValueError(f" ds_kind must be one of docs|queries|qrels|scoreddocs, got {ds_kind}.")
    
    
def preprocess(text: str, stop_words: set[str], stemmer: SnowballStemmer) -> str:
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()] 
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.command
def create_embedding(
    dataset: str = "msmarco-passage-v2/trec-dl-2022",
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased",
    batch_size: int = 1_000,
    sample_size: int = 10_000,
    ds_kind: str = "docs",
    out_dir: Path = Path("batches")  
):
    """Generate embeddings for a dataset using a specified model.

    Parameters
    ----------
    dataset: str, optional
        Name of the dataset to use.
    model_name: str
        Name of the SentenceTransformer model to use.
    num_samples: int, optional
        Number of samples to process from the dataset.
    batch_size: int, optional
        Batch size for processing samples.
    sample_size: int, optional
        Total number of samples to process from the dataset.
    ds_kind: str, optional
        Kind of dataset to process. Can be 'docs', 'queries', 'qrels', or 'scoreddocs'.
    out_dir: Path, optional
        Directory where the output embeddings will be saved.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    
    if ds_kind not in {"docs", "queries", "qrels", "screddocs"}:
        raise ValueError(f" ds_kind must be one of docs|queries|qrels|scoreddocs, got {ds_kind}.")
    
    model = SentenceTransformer(model_name)
    ds_iter = getIterator(dataset, ds_kind)

    print(
        f"model: {model_name}\n"
        f"dataset: {dataset}\n" 
        f"kind: {ds_kind}\n"
        f"batch_size: {batch_size}\n"
        f"sample_size: {sample_size}\n"
        f"output_dir: {out_dir}\n"
    )

    iterator = islice(ds_iter, sample_size)
    progress = tqdm(total=sample_size, unit=ds_kind.rstrip("s"))

    # ToDos: 
    #   - Add support for queries, qrels, and scoreddocs
    #   - Add support for multiprocessing
    #   - Add support for different models
    #   - Add support for different preprocessing methods
    for batch_idx, chunk in enumerate(iter(lambda:list(islice(iterator, batch_size)), [])):
        processed = {item.doc_id: model.encode(preprocess(item.text,  stop_words, stemmer)) for item in chunk}
        np.save(out_dir / f"{ds_kind}_{batch_idx}.npy", processed, allow_pickle=True)
        progress.update(len(chunk))

    progress.close()
    print(f"Embeddings saved to {out_dir}")

@app.default
def main():
    print("No command specified. Use --help for usage information.")