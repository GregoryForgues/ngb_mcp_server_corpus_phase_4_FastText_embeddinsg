# src/prepare_word_embedding_data.py (v3 - Robust Sanitization)

import pandas as pd
from pathlib import Path
import typer
from rich.console import Console
import logging
import unicodedata  # <-- NEW: Import for advanced string normalization
import re          # <-- NEW: Import for regular expressions

# Setup basic logging and a rich console for nice output
logging.basicConfig(level="INFO", format="%(message)s", handlers=[])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Prepares text corpora for fastText word embedding training based on L2 sections.")

def sanitize_filename(name: str) -> str:
    """
    Aggressively sanitizes a string to be a safe, pure-ASCII filename.
    - Transliterates diacritics (e.g., 'ū' -> 'u').
    - Replaces non-alphanumeric characters with underscores.
    """
    if not isinstance(name, str):
        return ""
    
    # 1. Normalize Unicode string, separating characters from diacritics (e.g., 'ū' -> 'u' + '̄')
    nfkd_form = unicodedata.normalize('NFKD', name)
    
    # 2. Encode to ASCII, ignoring the diacritics, then decode back to a clean string
    ascii_string = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
    
    # 3. Replace any remaining non-alphanumeric characters with a space
    safe_string = re.sub(r'[^a-zA-Z0-9]+', ' ', ascii_string)
    
    # 4. Replace spaces with underscores and make lowercase
    return safe_string.strip().replace(' ', '_').lower()


@app.command()
def create_corpora(
    chunks_path: Path = typer.Option(
        "output/pre_analysis/canonical_chunks.parquet", 
        help="Path to the canonical chunk file."
    ),
    catalog_path: Path = typer.Option(
        "data/Corpus_Catalog_v3.csv",
        help="Path to the main corpus catalog to get L2 section metadata."
    ),
    output_dir: Path = typer.Option(
        "output/word_embedding_training/", 
        help="Directory to save the training .txt files."
    ),
    min_chunks_per_section: int = typer.Option(
        50, 
        help="Minimum number of chunks required to create a corpus for an L2 section."
    )
):
    """
    Groups chunks by L2_corpus_section and concatenates their lemmatized text
    to create one large training file per section.
    """
    console.print(f"🚀 Starting Word Embedding Corpus Preparation (L2 Section Based)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df_chunks = pd.read_parquet(chunks_path)
        console.print(f"✔️ Loaded {len(df_chunks)} chunks from [cyan]{chunks_path}[/cyan].")
    except FileNotFoundError:
        console.print(f"❌ ERROR: Input file not found at {chunks_path}. Please run the pre-processing step first.")
        raise typer.Exit(code=1)

    try:
        df_catalog = pd.read_csv(catalog_path, usecols=['file_name', 'L2_corpus_section'])
        df_catalog.rename(columns={'file_name': 'document_id'}, inplace=True)
        df_chunks = pd.merge(df_chunks, df_catalog, on='document_id', how='left')
        df_chunks['L2_corpus_section'].fillna('unknown', inplace=True)
        console.print(f"✔️ Merged L2 section metadata from [cyan]{catalog_path}[/cyan].")
    except FileNotFoundError:
        console.print(f"❌ ERROR: Corpus Catalog not found at '{catalog_path}'. Cannot determine L2 sections.")
        raise typer.Exit(code=1)

    console.print(f"Grouping chunks by 'L2_corpus_section'...")
    grouped = df_chunks.groupby('L2_corpus_section')

    corpora_created = 0
    for name, group in grouped:
        if len(group) < min_chunks_per_section:
            console.print(f"  Skipping section '{name}': only {len(group)} chunks (min is {min_chunks_per_section}).")
            continue

        # --- THE KEY CHANGE: Use the new robust sanitization function ---
        safe_filename = sanitize_filename(name)
        if not safe_filename:
            safe_filename = f"section_{corpora_created}"
        
        output_path = output_dir / f"{safe_filename}.txt"

        console.print(f"  Processing '{name}' ({len(group)} chunks) -> [cyan]{output_path.name}[/cyan]")
        
        full_text = " ".join(group['text_for_lda'].dropna())

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        corpora_created += 1

    console.print(f"\n[bold green]✔️ Success![/bold green] Created {corpora_created} training corpora in [cyan]{output_dir}[/cyan].")

if __name__ == "__main__":
    app()