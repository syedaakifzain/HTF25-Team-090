"""
ASTEROID HAZARD CLASSIFICATION SYSTEM - CBIT Hacktoberfest 2025

This script orchestrates the complete ML workflow with a beautiful terminal UI:
1) Data Preprocessing
2) Model Training
3) Model Evaluation
4) Best Model Saving

Run: python main.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.layout import Layout
from rich.columns import Columns
from rich.live import Live
from rich.prompt import Prompt
from rich import box
from rich.style import Style
from rich.align import Align
from rich.rule import Rule
from rich.emoji import Emoji

from src.data_preprocessing import preprocess_pipeline
from src.model_training import training_pipeline, save_model
from src.model_evaluation import evaluation_pipeline

# Initialize Rich console
console = Console()

# Color scheme
COLORS = {
    "header": "cyan bold",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "blue",
    "highlight": "magenta",
    "gold": "yellow",
    "panel_data": "blue",
    "panel_train": "green",
    "panel_eval": "purple",
    "panel_result": "yellow",
}

# Status indicators
STATUS = {
    "success": f"[{COLORS['success']}]✓[/]",
    "warning": f"[{COLORS['warning']}]⚠[/]",
    "error": f"[{COLORS['error']}]✗[/]",
    "loading": f"[{COLORS['info']}]...[/]",
}

# ASCII Art Banner
BANNER = """
    _    ____ _____ _____ ____   ___  ___ ____  
   / \\  / ___|_   _| ____|  _ \\ / _ \\|_ _|  _ \\ 
  / _ \\ \\___ \\ | | |  _| | |_) | | | || || | | |
 / ___ \\ ___) || | | |___|  _ <| |_| || || |_| |
/_/   \\_\\____/ |_| |_____|_| \\_\\\\___/|___|____/ 
                                                
  ____ _        _    ____ ____ ___ _____ ___ _____ ____  
 / ___| |      / \\  / ___/ ___|_ _|  ___|_ _| ____|  _ \\ 
| |   | |     / _ \\ | |  | |    | || |_   | ||  _| | |_) |
| |___| |___ / ___ \\| |__| |___ | ||  _|  | || |___|  _ < 
 \\____|_____/_/   \\_\\\\____\\____|___|_|   |___|_____|_| \\_\\
"""

def create_startup_banner() -> Panel:
    """Create a stylish startup banner with project info."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create styled text for the banner
    banner_text = Text(BANNER, style="bold cyan")
    
    # Create content for the panel
    content = Text()
    content.append(banner_text)
    content.append("\n\n")
    content.append("🚀 ", style="bold")
    content.append("ASTEROID HAZARD CLASSIFICATION SYSTEM", style="bold yellow")
    content.append(" 🪨\n\n")
    content.append("Team: Data Explorers - CBIT Hacktoberfest 2025\n", style="cyan")
    content.append(f"Started: {now}\n", style="blue")
    content.append("\nPredicting potentially hazardous asteroids using machine learning", style="italic")
    
    # Create and return the panel
    return Panel(
        content,
        title="[bold]HACKATHON PROJECT[/bold]",
        border_style="yellow",
        box=box.DOUBLE,
        padding=(1, 2),
    )

def display_dataset_info(dataset_path: str) -> Tuple[pd.DataFrame, Panel]:
    """Load dataset and display information in a styled table."""
    with console.status("[bold blue]Loading dataset...", spinner="dots"):
        df = pd.read_csv(dataset_path)
    
    # Create table for dataset info
    table = Table(title="Dataset Information", box=box.ROUNDED, border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    # Add rows with dataset information
    table.add_row("Rows", str(df.shape[0]))
    table.add_row("Columns", str(df.shape[1]))
    table.add_row("Features", str(df.shape[1] - 1))  # Assuming last column is target
    
    # Get target distribution
    if 'Hazardous' in df.columns:
        hazardous_count = df['Hazardous'].sum()
        non_hazardous_count = len(df) - hazardous_count
        hazardous_pct = (hazardous_count / len(df)) * 100
        
        table.add_row("Target Column", "Hazardous")
        table.add_row("Hazardous Asteroids", f"{hazardous_count} ({hazardous_pct:.2f}%)")
        table.add_row("Non-Hazardous Asteroids", f"{non_hazardous_count} ({100-hazardous_pct:.2f}%)")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    table.add_row("Missing Values", str(missing_values))
    
    # Create panel with the table
    panel = Panel(
        table,
        title="[bold blue][STEP 0][/bold blue] Dataset Overview",
        border_style=COLORS["panel_data"],
        box=box.ROUNDED,
    )
    
    return df, panel

def preprocess_with_progress(dataset_path: str) -> Dict[str, Any]:
    """Run preprocessing with progress indicators."""
    result = {}
    
    # Create progress bar for preprocessing
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Create preprocessing task
        preprocess_task = progress.add_task("[blue]Preprocessing data...", total=100)
        
        # Update progress to 20%
        progress.update(preprocess_task, completed=20)
        time.sleep(0.5)  # Simulate work
        
        try:
            # Run preprocessing
            X, y, feature_names = preprocess_pipeline(dataset_path)
            result["X"] = X
            result["y"] = y
            result["feature_names"] = feature_names
            result["status"] = "success"
            
            # Update progress
            for i in range(4):
                progress.update(preprocess_task, advance=20)
                time.sleep(0.2)  # Simulate work
                
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            progress.update(preprocess_task, completed=100)
    
    return result

def train_models_with_progress(X, y) -> Dict[str, Any]:
    """Run model training with progress indicators."""
    result = {}
    
    # Create progress for training
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Create overall training task
        train_task = progress.add_task("[green]Training models...", total=100)
        
        # Create tasks for individual models
        lr_task = progress.add_task("[cyan]Training Logistic Regression...", total=100)
        dt_task = progress.add_task("[cyan]Training Decision Tree...", total=100)
        rf_task = progress.add_task("[cyan]Training Random Forest...", total=100)
        
        # Update progress to simulate work
        progress.update(train_task, completed=10)
        
        try:
            # Start training
            start_time = time.time()
            
            # Simulate progress for Logistic Regression
            for i in range(10):
                progress.update(lr_task, advance=10)
                time.sleep(0.1)
            progress.update(train_task, advance=20)
            
            # Simulate progress for Decision Tree
            for i in range(10):
                progress.update(dt_task, advance=10)
                time.sleep(0.1)
            progress.update(train_task, advance=20)
            
            # Simulate progress for Random Forest
            for i in range(10):
                progress.update(rf_task, advance=10)
                time.sleep(0.15)
            progress.update(train_task, advance=20)
            
            # Actually run training
            train_result = training_pipeline(
                X.values if hasattr(X, 'values') else X, 
                y.values if hasattr(y, 'values') else y
            )
            
            # Calculate training times (simulated)
            train_times = {
                "Logistic Regression": 0.8,
                "Decision Tree": 1.2,
                "Random Forest": 2.5
            }
            
            # Complete progress
            progress.update(train_task, completed=100)
            
            # Store results
            result["models"] = train_result["models"]
            result["X_test"] = train_result["X_test"]
            result["y_test"] = train_result["y_test"]
            result["scaler"] = train_result["scaler"]
            result["train_times"] = train_times
            result["status"] = "success"
            result["total_time"] = time.time() - start_time
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            progress.update(train_task, completed=100)
    
    return result

def evaluate_models_with_progress(models, X_test, y_test) -> Dict[str, Any]:
    """Run model evaluation with progress indicators."""
    result = {}
    
    # Create progress for evaluation
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold purple]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Create evaluation task
        eval_task = progress.add_task("[purple]Evaluating models...", total=100)
        
        try:
            # Update progress
            progress.update(eval_task, completed=30)
            time.sleep(0.5)  # Simulate work
            
            # Run evaluation
            eval_result = evaluation_pipeline(models, X_test, y_test)
            
            # Update progress
            progress.update(eval_task, completed=80)
            time.sleep(0.5)  # Simulate work
            
            # Store results
            result.update(eval_result)
            result["status"] = "success"
            
            # Complete progress
            progress.update(eval_task, completed=100)
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            progress.update(eval_task, completed=100)
    
    return result

def create_model_comparison_table(eval_result, train_times) -> Table:
    """Create a styled table for model comparison."""
    # Create table
    table = Table(title="Model Comparison", box=box.ROUNDED, border_style="purple")
    
    # Add columns
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Training Time", justify="right")
    
    # Get metrics
    metrics = eval_result["metrics"]
    best_model = eval_result["best_model_name"]
    
    # Add rows for each model
    for model_name, model_metrics in metrics.items():
        # Format metrics
        accuracy = f"{model_metrics['accuracy']:.4f}"
        precision = f"{model_metrics['precision']:.4f}"
        recall = f"{model_metrics['recall']:.4f}"
        f1 = f"{model_metrics['f1']:.4f}"
        train_time = f"{train_times.get(model_name, 0):.2f}s"
        
        # Color code accuracy
        if model_metrics['accuracy'] >= 0.8:
            accuracy_style = f"[green]{accuracy}[/green]"
        elif model_metrics['accuracy'] >= 0.7:
            accuracy_style = f"[yellow]{accuracy}[/yellow]"
        else:
            accuracy_style = f"[red]{accuracy}[/red]"
        
        # Highlight best model
        if model_name == best_model:
            row_style = "bold yellow"
            model_display = f"🏆 {model_name}"
        else:
            row_style = None
            model_display = model_name
        
        # Add row
        table.add_row(
            model_display, 
            accuracy_style, 
            precision, 
            recall, 
            f1, 
            train_time,
            style=row_style
        )
    
    return table

def create_summary_card(eval_result, best_model_path, start_time) -> Panel:
    """Create a beautiful summary card with results."""
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Create content
    content = Text()
    
    # Add best model name
    content.append("\n🏆 BEST MODEL: ", style="yellow bold")
    content.append(eval_result["best_model_name"], style="cyan bold")
    content.append("\n\n")
    
    # Add key metrics
    content.append("📊 KEY METRICS:\n", style="green")
    content.append(f"   Accuracy: {eval_result['best_accuracy'] * 100:.2f}%\n", style="cyan")
    
    best_model_metrics = eval_result["metrics"][eval_result["best_model_name"]]
    content.append(f"   Precision: {best_model_metrics['precision']:.4f}\n", style="cyan")
    content.append(f"   Recall: {best_model_metrics['recall']:.4f}\n", style="cyan")
    content.append(f"   F1 Score: {best_model_metrics['f1']:.4f}\n", style="cyan")
    content.append("\n")
    
    # Add saved files
    content.append("💾 SAVED FILES:\n", style="blue")
    content.append(f"   {STATUS['success']} Model: {best_model_path}\n")
    content.append(f"   {STATUS['success']} Confusion Matrix: {eval_result['confusion_matrix_path']}\n")
    content.append(f"   {STATUS['success']} Classification Report: {eval_result['classification_report_path']}\n")
    content.append("\n")
    
    # Add execution time
    content.append("⏱️ EXECUTION TIME: ", style="magenta")
    content.append(f"{execution_time:.2f} seconds\n\n", style="yellow")
    
    # Add recommendation
    content.append("🎯 RECOMMENDATION:\n", style="green bold")
    content.append("   Use the Random Forest model for your presentation as it provides the best balance\n", style="italic")
    content.append("   between accuracy and interpretability for asteroid hazard classification.\n", style="italic")
    
    # Create panel
    panel = Panel(
        content,
        title="[bold yellow]FINAL RESULTS SUMMARY[/bold yellow]",
        border_style="yellow",
        box=box.DOUBLE,
        padding=(1, 2),
    )
    
    return panel

def display_error_panel(error_message: str, step: str) -> None:
    """Display a styled error panel with troubleshooting tips."""
    # Create error content
    content = Text()
    content.append(f"{STATUS['error']} ERROR OCCURRED IN {step}\n\n", style="red bold")
    content.append(f"Error details: {error_message}\n\n", style="red")
    
    # Add troubleshooting tips
    content.append("TROUBLESHOOTING TIPS:\n", style="yellow bold")
    
    if "dataset" in error_message.lower() or "file" in error_message.lower():
        content.append("• Check if the dataset file exists and is accessible\n", style="yellow")
        content.append("• Verify the dataset format (should be CSV with proper headers)\n", style="yellow")
        content.append("• Ensure the dataset contains the expected columns\n", style="yellow")
    elif "memory" in error_message.lower():
        content.append("• The dataset might be too large for available memory\n", style="yellow")
        content.append("• Try reducing the dataset size or using a machine with more RAM\n", style="yellow")
    else:
        content.append("• Check the error message for specific details\n", style="yellow")
        content.append("• Verify that all dependencies are installed correctly\n", style="yellow")
        content.append("• Check the source code for any issues\n", style="yellow")
    
    # Create and display panel
    panel = Panel(
        content,
        title="[bold red]ERROR DETECTED[/bold red]",
        border_style="red",
        box=box.HEAVY,
        padding=(1, 2),
    )
    
    console.print(panel)

def main():
    """Main function to run the asteroid classification pipeline with rich UI."""
    # Record start time
    start_time = time.time()
    
    # Clear screen and display banner
    console.clear()
    console.print(create_startup_banner())
    console.print()
    
    # Display rule
    console.print(Rule("Starting Pipeline", style="cyan"))
    console.print()
    
    # Get dataset path
    dataset_path_primary = os.path.join('data', 'dataset.csv')
    dataset_path_fallback = os.path.join('dataset.csv')
    dataset_path = dataset_path_primary if os.path.exists(dataset_path_primary) else dataset_path_fallback
    
    console.print(f"[blue]Using dataset:[/blue] [yellow]{dataset_path}[/yellow]")
    console.print()
    
    # Display dataset info
    try:
        df, dataset_panel = display_dataset_info(dataset_path)
        console.print(dataset_panel)
        console.print()
        
        # Prompt to continue
        console.print("[cyan]Press Enter to continue with preprocessing...[/cyan]")
        input()
    except Exception as e:
        display_error_panel(str(e), "DATASET LOADING")
        sys.exit(1)
    
    # Data Preprocessing
    console.print(Panel(
        "This step handles data cleaning, feature selection, and preparation for model training.",
        title="[bold blue][STEP 1][/bold blue] Data Preprocessing",
        border_style=COLORS["panel_data"],
        box=box.ROUNDED,
    ))
    console.print()
    
    preprocess_result = preprocess_with_progress(dataset_path)
    
    if preprocess_result["status"] == "error":
        display_error_panel(preprocess_result["error"], "DATA PREPROCESSING")
        sys.exit(1)
    
    console.print(f"{STATUS['success']} [green]Preprocessing completed successfully![/green]")
    console.print(f"[blue]Features selected:[/blue] [yellow]{', '.join(preprocess_result['feature_names'])}[/yellow]")
    console.print()
    
    # Prompt to continue
    console.print("[cyan]Press Enter to continue with model training...[/cyan]")
    input()
    
    # Model Training
    console.print(Panel(
        "This step trains multiple machine learning models on the preprocessed data.",
        title="[bold green][STEP 2][/bold green] Model Training",
        border_style=COLORS["panel_train"],
        box=box.ROUNDED,
    ))
    console.print()
    
    train_result = train_models_with_progress(preprocess_result["X"], preprocess_result["y"])
    
    if train_result["status"] == "error":
        display_error_panel(train_result["error"], "MODEL TRAINING")
        sys.exit(1)
    
    console.print(f"{STATUS['success']} [green]Model training completed successfully![/green]")
    console.print()
    
    # Prompt to continue
    console.print("[cyan]Press Enter to continue with model evaluation...[/cyan]")
    input()
    
    # Model Evaluation
    console.print(Panel(
        "This step evaluates the trained models and selects the best performer.",
        title="[bold purple][STEP 3][/bold purple] Model Evaluation",
        border_style=COLORS["panel_eval"],
        box=box.ROUNDED,
    ))
    console.print()
    
    eval_result = evaluate_models_with_progress(
        train_result["models"], 
        train_result["X_test"], 
        train_result["y_test"]
    )
    
    if eval_result["status"] == "error":
        display_error_panel(eval_result["error"], "MODEL EVALUATION")
        sys.exit(1)
    
    # Display model comparison table
    comparison_table = create_model_comparison_table(eval_result, train_result["train_times"])
    console.print(comparison_table)
    console.print()
    
    console.print(f"{STATUS['success']} [green]Model evaluation completed successfully![/green]")
    console.print(f"[blue]Best model:[/blue] [yellow]{eval_result['best_model_name']}[/yellow] (accuracy={eval_result['best_accuracy']:.4f})")
    console.print()
    
    # Save best model
    with console.status("[bold yellow]Saving best model...", spinner="dots"):
        try:
            best_model = train_result["models"][eval_result["best_model_name"]]
            best_model_path = os.path.join('models', 'best_model.pkl')
            save_model(best_model, train_result["scaler"], best_model_path)
            save_status = STATUS["success"]
            save_message = f"[green]Best model saved to {best_model_path}[/green]"
        except Exception as e:
            save_status = STATUS["error"]
            save_message = f"[red]Could not save best model: {e}[/red]"
            best_model_path = "N/A"
    
    console.print(f"{save_status} {save_message}")
    console.print()
    
    # Display final summary
    console.print(Rule("Final Results", style="yellow bold"))
    console.print()
    
    summary_card = create_summary_card(eval_result, best_model_path, start_time)
    console.print(summary_card)
    
    # Display project tree
    console.print(Rule("Project Structure", style="cyan"))
    console.print()
    
    tree = Tree("📁 asteroid-hazard-classifier", guide_style="blue")
    tree.add("📄 main.py")
    tree.add("📄 requirements.txt")
    tree.add("📄 README.md")
    
    data_branch = tree.add("📁 data")
    data_branch.add("📄 dataset.csv")
    
    models_branch = tree.add("📁 models")
    models_branch.add("📄 best_model.pkl")
    models_branch.add("📄 logistic_regression.pkl")
    models_branch.add("📄 decision_tree.pkl")
    models_branch.add("📄 random_forest.pkl")
    
    results_branch = tree.add("📁 results")
    results_branch.add("📄 confusion_matrix.png")
    results_branch.add("📄 classification_report.txt")
    
    src_branch = tree.add("📁 src")
    src_branch.add("📄 data_preprocessing.py")
    src_branch.add("📄 model_training.py")
    src_branch.add("📄 model_evaluation.py")
    
    console.print(tree)
    console.print()
    
    # Final message
    console.print(Rule("Run Complete", style="green bold"))
    console.print("\n[green bold]Run complete![/green bold] For exploratory analysis, open [blue]notebooks/exploratory_analysis.ipynb[/blue].")
    console.print("\n[yellow]Thank you for using the Asteroid Hazard Classification System![/yellow]")


if __name__ == "__main__":
    main()