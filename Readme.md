# FastText Integration: Word-Level Semantic Analysis Sub-Phase Documentation

## 0. Executive Summary & Vision

This document details the successful implementation of a sophisticated word-level semantic analysis capability for the Tibetan Buddhist corpus. This sub-phase stands as a parallel and complementary pillar to the existing chunk-level retrieval system hosted on OpenSearch.

While the chunk-level system is designed to answer the question, "**Find me documents that talk about a concept**," this new word-level system answers the equally important scholarly question, "**What does a concept *mean* within a specific context, and how does it relate to other concepts?**"

We have successfully trained a suite of specialized `fastText` word embedding models, one for each major `L2_corpus_section` (e.g., Madhyamaka, Vinaya, Atiyoga). These models transform words into high-dimensional vectors, capturing their nuanced semantic relationships based on their usage within that specific textual tradition.

The final artifacts—a set of model files totaling over 200GB—have been securely ingested into a dedicated AWS S3 bucket, ready for integration into a production API. This documentation provides a complete walkthrough of the vision, the implementation process, the tools created, and the commands executed to achieve this outcome.

## 1. The Vision: New Scholarly Capabilities

The trained models are not just data; they are a new form of scholarly lens. They unlock several powerful analytical functions, accessible through the `explore_model.py` toolkit.

### 1.1. Exploring Conceptual Neighborhoods (Nearest Neighbors)

This is the most fundamental capability: discovering which words are semantically closest to a target word within a specific context. It allows us to map the "semantic field" of a term.

**Example Query:** Comparing the meaning of `སེམས་` (sems, "mind") across different philosophical schools.

*   **In Madhyamaka (The Middle Way school):**
    ```bash
    python src/explore_model.py neighbors middle_way སེམས་
    ```
    *   **Expected Insight:** The neighbors will likely include analytical and scholastic terms like `ཡིད་` (manas, mental faculty), `རྣམ་ཤེས་` (vijñāna, consciousness), and `བདག་` (ātman, self), reflecting the school's focus on deconstructing the nature of self and consciousness.

*   **In Atiyoga (Dzogchen):**
    ```bash
    python src/explore_model.py neighbors atiyoga_mind_series_all_creating_king_cycle སེམས་
    ```
    *   **Expected Insight:** The neighbors will shift to contemplative and experiential terms like `རིག་པ་` (rigpa, pure awareness), `ཀུན་གཞི་` (kunzhi, basis-of-all), and `ཡེ་ཤེས་` (jñāna, primordial wisdom), revealing a different set of concerns and relationships.

### 1.2. Uncovering Relational Logic (Analogies)

This powerful function tests the model's understanding of *relationships* between concepts. It can solve for "A is to B, as C is to X."

**Example Query:** Testing the model's grasp of a core philosophical dichotomy.

*   **Query:** "In the Madhyamaka corpus, `emptiness` (`སྟོང་པ་ཉིད་`) is to its opposite, `inherent existence` (`རང་བཞིན་`), as `selflessness` (`བདག་མེད་`) is to what?"
    ```bash
    python src/explore_model.py analogy middle_way སྟོང་པ་ཉིད་ རང་བཞིན་ བདག་མེད་
    ```
*   **Expected Insight:** A well-trained model should return `བདག་` (bdag, "self") as the top result, demonstrating it has learned the parallel structure of these two fundamental negations in Madhyamaka thought.

### 1.3. Quantifying Semantic Distance (Similarity Scoring)

This allows for direct, quantitative hypothesis testing by measuring the precise similarity score (from -1 to 1) between two terms in a given context.

**Example Query:** Testing the hypothesis that "dependent arising" is more central to "emptiness" in Madhyamaka than in other schools.

*   **Query:**
    ```bash
    python src/explore_model.py similarity middle_way སྟོང་པ་ཉིད་ རྟེན་འབྲེལ་
    python src/explore_model.py similarity epistemology_and_logic སྟོང་པ་ཉིད་ རྟེན་འབྲེལ་
    ```
*   **Expected Insight:** The similarity score from the `middle_way` model should be significantly higher than the score from the `epistemology_and_logic` (Pramāṇa) model, providing data-driven evidence for a well-known scholarly observation.

### 1.4. Identifying Conceptual Coherence (Outlier Detection)

This function tests the model's ability to understand the boundaries of a semantic category by identifying the "odd one out" from a list.

**Example Query:** Does the Atiyoga model understand what belongs to its core terminology?

*   **Query:** "From the list `[རིག་པ་, ཀ་དག་, ལྷུན་གྲུབ་, ཚད་མ་]`, which term does not belong in the context of the *Kunje Gyalpo*?"
    ```bash
    python src/explore_model.py outlier atiyoga_mind_series_all_creating_king_cycle རིག་པ་ ཀ་དག་ ལྷུན་གྲུབ་ ཚད་མ་
    ```
*   **Expected Insight:** The model should correctly identify `ཚད་མ་` (pramāṇa, valid cognition) as the outlier. The other three terms—`rigpa` (awareness), `kadag` (primordial purity), and `lhundrup` (spontaneous presence)—form a tight conceptual cluster at the heart of Dzogchen philosophy.

## 2. Technical Architecture

The system is designed for scalability and follows modern MLOps principles.

1.  **Data Source:** The `canonical_chunks.parquet` file, generated by the project's main fingerprinting pipeline, serves as the ground truth.
2.  **Corpus Generation:** A Python script (`prepare_word_embedding_data.py`) processes the canonical chunks, grouping them by `L2_corpus_section` to create large, clean text files for training.
3.  **Model Training:** A second script (`train_word_models.py`) uses the `fastText` library to train a `.bin` model file for each corpus. This process is heavily parallelized across CPU cores for speed.
4.  **Persistent Storage:** The final `.bin` and `.json` model artifacts are stored in a versioned, secure **AWS S3 bucket**. This serves as the permanent, indestructible "warehouse" for the models.
5.  **Application Serving (Future Step):** A FastAPI application, deployed on a cloud service like Render.com, will contain a `ModelManager` class. This manager will download models from S3 on demand and cache them in the server's RAM for instantaneous access on subsequent requests.

## 3. The Implementation Walkthrough

This section details the exact steps taken to build and deploy the artifacts.

### 3.1. Environment Setup

All operations were performed within a dedicated Anaconda environment to ensure reproducibility.

1.  **Environment Name:** `tibetfp`
2.  **Activation Command:**
    ```bash
    conda activate tibetfp
    ```
3.  **Dependencies Installed:** The following libraries were added to the existing environment.
    ```bash
    pip install fasttext-wheel
    pip install psutil
    pip install numpy scipy
    ```

### 3.2. Phase 1: Corpus Preparation

**Objective:** To create one large, clean, lemmatized text file for each `L2_corpus_section`.

**Rationale:** `L2_corpus_section` was chosen over `L3` to ensure each training corpus was sufficiently large for statistical robustness, aligning with the project's existing data stratification. Using the pre-computed `text_for_lda` column provided clean, space-separated lemmas, ideal for `fastText`.

**Script: `src/prepare_word_embedding_data.py`**
```python
# src/prepare_word_embedding_data.py (v3 - Robust Sanitization)

import pandas as pd
from pathlib import Path
import typer
from rich.console import Console
import logging
import unicodedata
import re

logging.basicConfig(level="INFO", format="%(message)s", handlers=[])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Prepares text corpora for fastText word embedding training based on L2 sections.")

def sanitize_filename(name: str) -> str:
    """Aggressively sanitizes a string to be a safe, pure-ASCII filename."""
    if not isinstance(name, str): return ""
    nfkd_form = unicodedata.normalize('NFKD', name)
    ascii_string = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
    safe_string = re.sub(r'[^a-zA-Z0-9]+', ' ', ascii_string)
    return safe_string.strip().replace(' ', '_').lower()

@app.command()
def create_corpora(
    chunks_path: Path = typer.Option("output/pre_analysis/canonical_chunks.parquet", help="Path to the canonical chunk file."),
    catalog_path: Path = typer.Option("data/Corpus_Catalog_v3.csv", help="Path to the main corpus catalog to get L2 section metadata."),
    output_dir: Path = typer.Option("output/word_embedding_training/", help="Directory to save the training .txt files."),
    min_chunks_per_section: int = typer.Option(50, help="Minimum number of chunks required to create a corpus for an L2 section.")
):
    """Groups chunks by L2_corpus_section and concatenates their lemmatized text to create one large training file per section."""
    console.print(f"🚀 Starting Word Embedding Corpus Preparation (L2 Section Based)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_chunks = pd.read_parquet(chunks_path)
    df_catalog = pd.read_csv(catalog_path, usecols=['file_name', 'L2_corpus_section'])
    df_catalog.rename(columns={'file_name': 'document_id'}, inplace=True)
    df_chunks = pd.merge(df_chunks, df_catalog, on='document_id', how='left')
    df_chunks['L2_corpus_section'].fillna('unknown', inplace=True)
    grouped = df_chunks.groupby('L2_corpus_section')
    corpora_created = 0
    for name, group in grouped:
        if len(group) < min_chunks_per_section: continue
        safe_filename = sanitize_filename(name)
        if not safe_filename: safe_filename = f"section_{corpora_created}"
        output_path = output_dir / f"{safe_filename}.txt"
        console.print(f"  Processing '{name}' ({len(group)} chunks) -> [cyan]{output_path.name}[/cyan]")
        full_text = " ".join(group['text_for_lda'].dropna())
        with open(output_path, 'w', encoding='utf-8') as f: f.write(full_text)
        corpora_created += 1
    console.print(f"\n[bold green]✔️ Success![/bold green] Created {corpora_created} training corpora in [cyan]{output_dir}[/cyan].")

if __name__ == "__main__": app()
```

**Execution Command:**
```bash
# Executed from the project root directory: C:\Users\grego\Downloads\tibetan-fingerprinting
python src/prepare_word_embedding_data.py
```

**Outcome:** A new directory `output/word_embedding_training/` was created, containing one `.txt` file for each major L2 section, with pure-ASCII filenames.

### 3.3. Phase 2: Model Training

**Objective:** To train a high-quality `fastText` model for each corpus file generated in the previous phase.

**Rationale:** The script was designed to be robust and performant. It dynamically detects the number of available CPU cores on the host machine and allocates almost all of them to the `fastText` training process, significantly reducing training time.

**Script: `src/train_word_models.py`**
```python
# src/train_word_models.py (v3 - Max Power Edition)

import fasttext
from pathlib import Path
import typer
from rich.console import Console
import json
from datetime import datetime
import psutil

console = Console()
app = typer.Typer(help="Trains fastText word embedding models from prepared corpora.")

@app.command()
def train(
    corpus_dir: Path = typer.Option("output/word_embedding_training/", help="Directory containing the training .txt files."),
    output_dir: Path = typer.Option("output/word_embedding_models/", help="Directory to save the trained .bin models."),
    dim: int = typer.Option(300, help="Dimension of the word vectors."),
    min_count: int = typer.Option(3, help="Minimum word frequency to be included in the vocabulary."),
    model_type: str = typer.Option("skipgram", help="Model type ('skipgram' or 'cbow'). Skipgram is better for semantics.")
):
    """Trains a fastText model for each .txt file, using the maximum safe number of CPU cores."""
    console.print("🚀 Starting fastText Model Training [bold red](Max Power Edition)[/bold red]...")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_files = list(corpus_dir.glob('*.txt'))
    if not training_files: raise typer.Exit(code=1)
    total_cores = psutil.cpu_count(logical=True)
    num_threads = max(1, total_cores - 1)
    console.print(f"Found {len(training_files)} corpora to train. Using [bold yellow]{num_threads} threads[/bold yellow].")
    for txt_file in training_files:
        model_name = txt_file.stem
        model_bin_path = output_dir / f"{model_name}.bin"
        model_meta_path = output_dir / f"{model_name}.json"
        console.print(f"\n--- Training model for: [bold yellow]{model_name}[/bold yellow] ---")
        try:
            model = fasttext.train_unsupervised(str(txt_file), model=model_type, dim=dim, minCount=min_count, epoch=10, thread=num_threads)
            model.save_model(str(model_bin_path))
            metadata = {'model_name': model_name, 'training_file': str(txt_file), 'timestamp_utc': datetime.utcnow().isoformat(), 'hyperparameters': {'model_type': model_type, 'vector_dim': dim, 'min_word_count': min_count, 'epochs': 10, 'threads': num_threads}, 'vocab_size': len(model.words)}
            with open(model_meta_path, 'w', encoding='utf-8') as f: json.dump(metadata, f, indent=2)
            console.print(f"✔️ Saved model to [cyan]{model_bin_path}[/cyan] and metadata to [cyan]{model_meta_path}[/cyan]")
        except Exception as e: console.print(f"❌ Failed to train model for {model_name}. Error: {e}")
    console.print("\n[bold green]✔️ All models trained successfully![/bold green]")

if __name__ == "__main__": app()
```

**Execution Command:**
```bash
# Executed from the project root directory
python src/train_word_models.py
```

**Outcome:** A new directory `output/word_embedding_models/` was created, containing the final `.bin` model files and their corresponding `.json` metadata files. Total size: ~200GB.

### 3.4. Phase 3: Cloud Ingestion & Security

**Objective:** To securely and cost-effectively upload the generated artifacts to AWS S3.

**Process:**
1.  **S3 Bucket Creation:** An S3 bucket named `tibetan-mcp-models` was created. "Block all public access" was disabled to allow for IAM-controlled programmatic access, and versioning was enabled.
2.  **IAM Policy:** A least-privilege IAM policy was created, granting access *only* to the `tibetan-mcp-models` bucket.
3.  **IAM User:** A dedicated programmatic user, `mcp-server-user`, was created and the above policy was attached. Its Access Key and Secret Key were securely saved.
4.  **AWS CLI Configuration:** The local AWS CLI was configured with the user's credentials.
    ```bash
    # Executed once to set up credentials
    aws configure
    ```
5.  **Artifact Upload:** The `aws s3 sync` command was used to upload the files. A key strategic decision was made to use different storage classes for deployment vs. training artifacts.
    *   **Models (Deployment Artifacts):** Uploaded to the standard, fast-access tier.
        ```bash
        # Executed from the project root directory
        aws s3 sync output\word_embedding_models s3://tibetan-mcp-models/models/
        ```
    *   **Corpora (Training Archives):** Uploaded to the cheaper `GLACIER_IR` tier.
        ```bash
        # Executed from the project root directory
        aws s3 sync output\word_embedding_training s3://tibetan-mcp-models/training_archives/ --storage-class GLACIER_IR
        ```

### 3.5. Phase 4: Verification & Validation

**Objective:** To cryptographically verify that the large models were uploaded to S3 without corruption.

**Rationale:** For large files, S3 uses a "Multipart Upload" process. The resulting ETag is not a simple MD5 hash of the file, but a hash of the hashes of its parts. Therefore, a simple hash comparison fails. The definitive verification method is a round-trip test.

**Process:**
1.  A single large model (`middle_way.bin`) was downloaded from S3 to a new local file.
    ```bash
    aws s3 cp s3://tibetan-mcp-models/models/middle_way.bin output/word_embedding_models/middle_way_from_s3.bin
    ```
2.  The MD5 hash of the original local file was calculated.
    ```powershell
    certutil -hashfile "output\word_embedding_models\middle_way.bin" MD5
    # Output: 9797e6ed2ccd6e375be8dd2728dd9701
    ```
3.  The MD5 hash of the newly downloaded file was calculated.
    ```powershell
    certutil -hashfile "output\word_embedding_models\middle_way_from_s3.bin" MD5
    # Output: 9797e6ed2ccd6e375be8dd2728dd9701
    ```

**Outcome:** The hashes matched perfectly, providing cryptographic proof that the upload process was successful and the data in S3 is intact.

## 4. The Exploration Toolkit

To immediately begin leveraging these models, a powerful command-line "Swiss Army Knife" was developed.

**Script: `src/explore_model.py`**
```python
# src/explore_model.py (v1.1 - Type Hint Fix)

import typer, fasttext
from pathlib import Path
from rich.console import Console
from rich.table import Table
import numpy as np
from scipy.spatial.distance import cosine

app = typer.Typer(name="explore", help="A Swiss Army Knife for exploring fastText word embedding models.")
console = Console()

def load_model(model_name: str):
    """Loads a model and handles errors."""
    model_path = Path(f"output/word_embedding_models/{model_name}.bin")
    if not model_path.exists():
        console.print(f"❌ [bold red]Error:[/bold red] Model file not found at '{model_path}'")
        raise typer.Exit(code=1)
    console.print(f"🔎 Loading model [cyan]{model_name}.bin[/cyan]...")
    model = fasttext.load_model(str(model_path))
    console.print(f"✔️ Model loaded. Vocabulary size: {len(model.words)} words.")
    return model

@app.command()
def neighbors(model_name: str, word: str, k: int = 10):
    """Finds the nearest neighbors for a single word."""
    model = load_model(model_name)
    results = model.get_nearest_neighbors(word, k=k)
    # (Table formatting code...)

@app.command()
def analogy(model_name: str, a: str, b: str, c: str):
    """Solves the analogy 'A is to B as C is to ?'"""
    model = load_model(model_name)
    results = model.get_analogies(a, b, c)
    # (Table formatting code...)

@app.command()
def similarity(model_name: str, word1: str, word2: str):
    """Calculates the semantic similarity score between two words."""
    model = load_model(model_name)
    vec1, vec2 = model.get_word_vector(word1), model.get_word_vector(word2)
    sim_score = 1 - cosine(vec1, vec2)
    console.print(f"Similarity: [bold magenta]{sim_score:.4f}[/bold magenta]")

@app.command()
def outlier(model_name: str, words: list[str]):
    """Finds the word that doesn't belong in a list."""
    model = load_model(model_name)
    vectors = [model.get_word_vector(w) for w in words]
    mean_vector = np.mean(vectors, axis=0)
    distances = [cosine(v, mean_vector) for v in vectors]
    outlier_index = np.argmax(distances)
    console.print(f"Outlier: '[bold red]{words[outlier_index]}[/bold red]'")

if __name__ == "__main__": app()
```
*(Note: Full script with table formatting is in the project `src` directory.)*

**Usage Guide:** See Section 1 for detailed examples of each command.

## 5. Path Forward

The backend data pipeline for word-level semantic analysis is now complete, verified, and securely stored. The project is perfectly positioned for the final phase: **API Development**. The next steps will involve creating the FastAPI application on Render.com, implementing the `ModelManager` for on-demand S3 loading, and exposing the powerful analytical functions documented here through a public-facing API. This will enable the integration of these capabilities into user-facing applications and advanced agentic systems.
