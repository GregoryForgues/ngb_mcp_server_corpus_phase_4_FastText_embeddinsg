# src/train_word_models.py (v3 - Max Power Edition)

import fasttext
from pathlib import Path
import typer
from rich.console import Console
import json
from datetime import datetime
import psutil  # <-- NEW: Import the system utilities library

console = Console()
app = typer.Typer(help="Trains fastText word embedding models from prepared corpora.")

@app.command()
def train(
    corpus_dir: Path = typer.Option(
        "output/word_embedding_training/", 
        help="Directory containing the training .txt files."
    ),
    output_dir: Path = typer.Option(
        "output/word_embedding_models/", 
        help="Directory to save the trained .bin models."
    ),
    dim: int = typer.Option(300, help="Dimension of the word vectors."),
    min_count: int = typer.Option(3, help="Minimum word frequency to be included in the vocabulary."),
    model_type: str = typer.Option("skipgram", help="Model type ('skipgram' or 'cbow'). Skipgram is better for semantics.")
):
    """
    Finds all .txt files in the corpus directory and trains a fastText model for each,
    using the maximum safe number of CPU cores to accelerate the process.
    """
    console.print("🚀 Starting fastText Model Training [bold red](Max Power Edition)[/bold red]...")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_files = list(corpus_dir.glob('*.txt'))

    if not training_files:
        console.print(f"❌ ERROR: No training files found in {corpus_dir}. Run corpus preparation first.")
        raise typer.Exit(code=1)

    # --- THE KEY CHANGE: Dynamically set the number of threads ---
    # Get the number of logical CPU cores (includes hyper-threading)
    total_cores = psutil.cpu_count(logical=True)
    # Use most cores, but leave one free for system responsiveness. Ensure at least 1.
    num_threads = max(1, total_cores - 1)
    
    console.print(f"Found {len(training_files)} corpora to train.")
    console.print(f"System has {total_cores} CPU cores. Using [bold yellow]{num_threads} threads[/bold yellow] for training.")

    for txt_file in training_files:
        model_name = txt_file.stem
        model_bin_path = output_dir / f"{model_name}.bin"
        model_meta_path = output_dir / f"{model_name}.json"
        
        console.print(f"\n--- Training model for: [bold yellow]{model_name}[/bold yellow] ---")
        
        try:
            # The core training command, now supercharged with more threads!
            model = fasttext.train_unsupervised(
                str(txt_file),
                model=model_type,
                dim=dim,
                minCount=min_count,
                epoch=10,
                thread=num_threads  # <-- Pass the dynamically calculated number of threads
            )
            
            model.save_model(str(model_bin_path))
            console.print(f"✔️ Saved model to [cyan]{model_bin_path}[/cyan]")

            # Save metadata for reproducibility
            metadata = {
                'model_name': model_name,
                'training_file': str(txt_file),
                'timestamp_utc': datetime.utcnow().isoformat(),
                'hyperparameters': {
                    'model_type': model_type, 'vector_dim': dim, 'min_word_count': min_count, 'epochs': 10, 'threads': num_threads
                },
                'vocab_size': len(model.words),
            }
            with open(model_meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            console.print(f"✔️ Saved metadata to [cyan]{model_meta_path}[/cyan]")

        except Exception as e:
            console.print(f"❌ Failed to train model for {model_name}. Error: {e}")

    console.print("\n[bold green]✔️ All models trained successfully![/bold green]")

if __name__ == "__main__":
    app()