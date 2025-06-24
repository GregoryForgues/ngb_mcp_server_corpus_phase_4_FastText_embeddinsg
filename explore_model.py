# src/explore_model.py (v1.1 - Type Hint Fix)

import typer
import fasttext
from pathlib import Path
from rich.console import Console
from rich.table import Table
import numpy as np
from scipy.spatial.distance import cosine

# --- Setup ---
app = typer.Typer(name="explore", help="A Swiss Army Knife for exploring fastText word embedding models.")
console = Console()

# --- Helper Functions ---
def load_model(model_name: str): # <-- THE FIX: Removed the '-> fasttext._FastText' part
    """Loads a model and handles errors."""
    model_path = Path(f"output/word_embedding_models/{model_name}.bin")
    if not model_path.exists():
        console.print(f"❌ [bold red]Error:[/bold red] Model file not found at '{model_path}'")
        raise typer.Exit(code=1)
    
    console.print(f"🔎 Loading model [cyan]{model_name}.bin[/cyan]...")
    try:
        model = fasttext.load_model(str(model_path))
        console.print(f"✔️ Model loaded. Vocabulary size: {len(model.words)} words.")
        return model
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] Failed to load model. Error: {e}")
        raise typer.Exit(code=1)

# --- Sub-commands ---

@app.command()
def neighbors(
    model_name: str = typer.Argument(..., help="Model name (e.g., 'madhyamaka')."),
    word: str = typer.Argument(..., help="The Tibetan word to query."),
    k: int = typer.Option(10, "-k", help="Number of neighbors to show.")
):
    """Finds the nearest neighbors for a single word."""
    model = load_model(model_name)
    if word not in model.words:
        console.print(f"❌ [bold red]Error:[/bold red] Word '{word}' not found in this model's vocabulary.")
        raise typer.Exit(code=1)

    results = model.get_nearest_neighbors(word, k=k)
    table = Table(title=f"Top {k} Nearest Neighbors for '{word}' in '{model_name}'")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Similarity", style="magenta", justify="center")
    table.add_column("Neighbor Word", style="cyan", justify="left")
    for i, (score, neighbor) in enumerate(results):
        table.add_row(str(i + 1), f"{score:.4f}", neighbor)
    console.print(table)

@app.command()
def analogy(
    model_name: str = typer.Argument(..., help="Model name."),
    a: str = typer.Argument(..., help="Word A (e.g., སྟོང་པ་ཉིད་)"),
    b: str = typer.Argument(..., help="Word B (e.g., རང་བཞིན་)"),
    c: str = typer.Argument(..., help="Word C (e.g., བདག་མེད་)")
):
    """Solves the analogy 'A is to B as C is to ?'"""
    model = load_model(model_name)
    results = model.get_analogies(a, b, c)
    
    table = Table(title=f"Analogy Results for '{a}' → '{b}' :: '{c}' → ?")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Similarity", style="magenta", justify="center")
    table.add_column("Result Word", style="cyan", justify="left")
    for i, (score, word) in enumerate(results):
        table.add_row(str(i + 1), f"{score:.4f}", word)
    console.print(table)

@app.command()
def similarity(
    model_name: str = typer.Argument(..., help="Model name."),
    word1: str = typer.Argument(..., help="The first word."),
    word2: str = typer.Argument(..., help="The second word.")
):
    """Calculates the semantic similarity score between two words."""
    model = load_model(model_name)
    vec1 = model.get_word_vector(word1)
    vec2 = model.get_word_vector(word2)
    # Cosine similarity is 1 - cosine distance
    sim_score = 1 - cosine(vec1, vec2)
    console.print(f"Similarity between '[cyan]{word1}[/cyan]' and '[cyan]{word2}[/cyan]' in '{model_name}': [bold magenta]{sim_score:.4f}[/bold magenta]")

@app.command()
def outlier(
    model_name: str = typer.Argument(..., help="Model name."),
    words: list[str] = typer.Argument(..., help="A list of words to test (e.g., རིག་པ་ ཀ་དག་ ཚད་མ་).")
):
    """Finds the word that doesn't belong in a list."""
    model = load_model(model_name)
    vectors = [model.get_word_vector(w) for w in words]
    mean_vector = np.mean(vectors, axis=0)
    
    distances = [cosine(v, mean_vector) for v in vectors]
    outlier_index = np.argmax(distances)
    
    console.print(f"In the list {words}, the word '[bold red]{words[outlier_index]}[/bold red]' is the most likely outlier.")

if __name__ == "__main__":
    app()