"""
Command-line interface for 1p-spatial-pipeline.

Usage:
    1p-spatial-pipeline info                    # Show pipeline overview
    1p-spatial-pipeline init                    # Generate config template
    1p-spatial-pipeline nidq --config ...       # Visualize NIDQ channels (Pipeline C)
    1p-spatial-pipeline merge --config ...      # Merge all data (Pipeline D)
    1p-spatial-pipeline extract --config ...    # Extract active cells (Pipeline E)
    1p-spatial-pipeline analyze --config ...    # Place field analysis (Pipeline F)
    1p-spatial-pipeline run-all --config ...    # Run full pipeline (D -> E -> F)

Note: Pipelines A (DeepLabCut) and B (CNMF-E) must be run externally via the
provided Jupyter notebooks before using this CLI.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .config import (
    load_config,
    validate_config,
    generate_config_template,
    print_config_summary,
    ConfigError,
)

console = Console()


PIPELINE_DIAGRAM = """
[bold cyan]1p-spatial-pipeline: One-Photon Spatial Coding Analysis[/bold cyan]

[dim]External (run separately via notebooks):[/dim]
  [yellow]Pipeline A[/yellow]: DeepLabCut (behavior extraction from webcam)
           Notebook: notebooks/SingleMouseTracking_via_DeepLabCut.ipynb
           Output:   Position CSV (frame, x, y, likelihood)

  [yellow]Pipeline B[/yellow]: CNMF-E + OASIS (calcium extraction from miniscope)
           Notebook: notebooks/cnmfE_pipeline.ipynb (run on Colab/conda)
           Output:   MAT file with C, S, spatial components

[dim]Integrated (this CLI):[/dim]
  [yellow]Pipeline C[/yellow]: NIDQ Signal Processing
           Input:  NIDQ .bin/.meta files
           Output: Sync/trigger/reward channel assignments

[bold green]Data Flow:[/bold green]
                    [yellow]A[/yellow] (CSV)
                       |
                       v
  [yellow]B[/yellow] (MAT) -----> [cyan]D[/cyan] (Merge) -----> [cyan]E[/cyan] (Extract) -----> [cyan]F[/cyan] (Analyze)
                       ^                     |                      |
                       |                     v                      v
                    [yellow]C[/yellow] (NIDQ)         Downstream MAT        Place field plots
                                         (active cells)           + results MAT

[dim]Commands:[/dim]
  [green]info[/green]     Show this pipeline overview
  [green]init[/green]     Generate configuration template
  [green]nidq[/green]     Visualize NIDQ channels (Pipeline C)
  [green]merge[/green]    Merge all data sources (Pipeline D)
  [green]extract[/green]  Extract active cells (Pipeline E)
  [green]analyze[/green]  Run place field analysis (Pipeline F)
  [green]run-all[/green]  Run full pipeline (D -> E -> F)
"""


@click.group()
@click.version_option(version=__version__, prog_name="1p-spatial-pipeline")
def main():
    """
    One-photon spatial coding analysis pipeline.

    A user-friendly CLI for processing v4 miniscope recordings with
    behavior tracking and NIDQ synchronization signals.
    """
    pass


@main.command()
def info():
    """Show pipeline overview and data flow diagram."""
    console.print(Panel(PIPELINE_DIAGRAM, title="1p-spatial-pipeline", border_style="blue"))


@main.command()
@click.option(
    "--experiment", "-e",
    default="MY01_20251231",
    help="Experiment name for the configuration"
)
@click.option(
    "--output", "-o",
    default="config.yaml",
    help="Output path for configuration file"
)
def init(experiment: str, output: str):
    """Generate a configuration template file."""
    output_path = Path(output)

    if output_path.exists():
        if not click.confirm(f"File {output} already exists. Overwrite?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    template = generate_config_template(experiment, str(output_path))

    console.print(f"[green]Configuration template created:[/green] {output_path}")
    console.print("\n[dim]Next steps:[/dim]")
    console.print("  1. Edit the configuration file with your paths and parameters")
    console.print("  2. Run [cyan]1p-spatial-pipeline nidq --config config.yaml[/cyan] to determine channel assignments")
    console.print("  3. Run [cyan]1p-spatial-pipeline merge --config config.yaml[/cyan] to merge data")
    console.print("  4. Run [cyan]1p-spatial-pipeline extract --config config.yaml[/cyan] to extract active cells")
    console.print("  5. Run [cyan]1p-spatial-pipeline analyze --config config.yaml[/cyan] for place field analysis")


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file"
)
@click.option(
    "--preview-seconds", "-p",
    default=10.0,
    help="Duration to preview (seconds)"
)
def nidq(config: str, preview_seconds: float):
    """
    Visualize NIDQ channels to determine sync/trigger/reward assignments.

    This command reads the NIDQ binary file and displays each channel's
    signal profile to help you identify which bit corresponds to:
    - sync (miniscope frame sync)
    - trigger (webcam trigger)
    - reward_left / reward_right
    """
    try:
        cfg = load_config(config)
        validate_config(cfg, ['C'])
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)

    console.print("[cyan]Loading NIDQ data...[/cyan]")

    from .pipeline_c.nidq_processor import visualize_nidq_channels

    try:
        visualize_nidq_channels(cfg, preview_seconds)
    except Exception as e:
        console.print(f"[red]Error processing NIDQ data:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file"
)
@click.option(
    "--cnmfe-mat", "-m",
    type=click.Path(exists=True),
    help="CNMF-E MAT file (overrides cnmfe.mat_file in config)"
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompts"
)
def merge(config: str, cnmfe_mat: Optional[str], yes: bool):
    """
    Merge data from Pipelines A, B, and C (Pipeline D).

    Combines:
    - Position data from DeepLabCut CSV
    - Calcium data from CNMF-E MAT file
    - Synchronization signals from NIDQ

    Output: Merged MAT file ready for downstream analysis.
    """
    try:
        cfg = load_config(config)
        validate_config(cfg, ['D'])
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)

    # Show configuration summary
    console.print("\n[bold]Pipeline D: Data Merge[/bold]\n")
    print_config_summary(cfg)

    if not yes:
        if not click.confirm("\nProceed with these settings?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    from .pipeline_d.merger import merge_all_data

    # Use CLI flag if provided, otherwise fall back to config
    mat_path = cnmfe_mat or cfg.get('cnmfe', {}).get('mat_file')

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Merging data...", total=None)
            output_path = merge_all_data(cfg, mat_path)
            progress.update(task, completed=True)

        console.print(f"\n[green]Merge complete![/green]")
        console.print(f"Output saved to: [cyan]{output_path}[/cyan]")

    except Exception as e:
        console.print(f"[red]Error during merge:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file"
)
@click.option(
    "--input-mat", "-i",
    type=click.Path(exists=True),
    help="Input MAT file from Pipeline D (if not using default)"
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["C", "S", "both"]),
    default="both",
    help="Activity mode: C (calcium traces), S (deconvolved spikes), or both"
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompts"
)
def extract(config: str, input_mat: Optional[str], mode: str, yes: bool):
    """
    Extract active cells and prepare for downstream analysis (Pipeline E).

    Performs:
    - Good cell filtering based on CNMF-E evaluation
    - Duplicate cell removal (spatial proximity + correlation)
    - Global active cell detection
    - Trial-by-trial data extraction

    Output: Downstream MAT file with active cells and trial-aligned data.
    """
    try:
        cfg = load_config(config)
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)

    console.print("\n[bold]Pipeline E: Active Cell Extraction[/bold]\n")
    console.print(f"Mode: [cyan]{mode}[/cyan]")

    if not yes:
        if not click.confirm("\nProceed with these settings?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    from .pipeline_e.active_cells import extract_active_cells

    modes = ["C", "S"] if mode == "both" else [mode]

    for m in modes:
        console.print(f"\n[cyan]Processing mode: {m}[/cyan]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Extracting active cells ({m} mode)...", total=None)
                output_path = extract_active_cells(cfg, input_mat, m)
                progress.update(task, completed=True)

            console.print(f"[green]Complete![/green] Output: [cyan]{output_path}[/cyan]")

        except Exception as e:
            console.print(f"[red]Error during extraction ({m} mode):[/red] {e}")
            if mode != "both":
                sys.exit(1)


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file"
)
@click.option(
    "--input-mat", "-i",
    type=click.Path(exists=True),
    help="Input MAT file from Pipeline E (if not using default)"
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["C", "S", "both"]),
    default="both",
    help="Activity mode: C (calcium traces), S (deconvolved spikes), or both"
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompts"
)
def analyze(config: str, input_mat: Optional[str], mode: str, yes: bool):
    """
    Run place field analysis (Pipeline F).

    Performs:
    - Movement and spatial filtering
    - Transient detection
    - Place field calculation with Gaussian smoothing
    - Place cell identification via MI shuffle test
    - Direction-specific analysis (L/R)

    Output: Place field plots and results MAT file.
    """
    try:
        cfg = load_config(config)
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)

    console.print("\n[bold]Pipeline F: Place Field Analysis[/bold]\n")
    console.print(f"Mode: [cyan]{mode}[/cyan]")
    console.print(f"Shuffle iterations: [cyan]{cfg['analysis']['n_shuffles']}[/cyan]")
    console.print(f"P-value threshold: [cyan]{cfg['analysis']['p_threshold']}[/cyan]")

    if not yes:
        if not click.confirm("\nProceed with these settings?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    from .pipeline_f.place_fields import analyze_place_fields

    modes = ["C", "S"] if mode == "both" else [mode]

    for m in modes:
        console.print(f"\n[cyan]Analyzing mode: {m}[/cyan]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Analyzing place fields ({m} mode)...", total=None)
                output_dir = analyze_place_fields(cfg, input_mat, m)
                progress.update(task, completed=True)

            console.print(f"[green]Complete![/green] Output directory: [cyan]{output_dir}[/cyan]")

        except Exception as e:
            console.print(f"[red]Error during analysis ({m} mode):[/red] {e}")
            if mode != "both":
                sys.exit(1)


@main.command("run-all")
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file"
)
@click.option(
    "--cnmfe-mat", "-m",
    type=click.Path(exists=True),
    help="CNMF-E MAT file (overrides cnmfe.mat_file in config)"
)
@click.option(
    "--mode",
    type=click.Choice(["C", "S", "both"]),
    default="both",
    help="Activity mode: C (calcium traces), S (deconvolved spikes), or both"
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompts"
)
def run_all(config: str, cnmfe_mat: Optional[str], mode: str, yes: bool):
    """
    Run full pipeline: Merge (D) -> Extract (E) -> Analyze (F).

    This command runs all integrated pipeline steps in sequence.
    """
    try:
        cfg = load_config(config)
        validate_config(cfg, ['D'])
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)

    console.print("\n[bold]Full Pipeline Execution[/bold]")
    console.print(f"Pipelines: D (Merge) -> E (Extract) -> F (Analyze)")
    console.print(f"Mode: [cyan]{mode}[/cyan]\n")

    print_config_summary(cfg)

    if not yes:
        if not click.confirm("\nProceed with full pipeline?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    # Pipeline D: Merge
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]PIPELINE D: DATA MERGE[/bold cyan]")
    console.print("=" * 60)

    from .pipeline_d.merger import merge_all_data

    # Use CLI flag if provided, otherwise fall back to config
    mat_path = cnmfe_mat or cfg.get('cnmfe', {}).get('mat_file')

    try:
        merged_mat = merge_all_data(cfg, mat_path)
        console.print(f"[green]Merge complete:[/green] {merged_mat}")
    except Exception as e:
        console.print(f"[red]Error during merge:[/red] {e}")
        sys.exit(1)

    # Pipeline E: Extract
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]PIPELINE E: ACTIVE CELL EXTRACTION[/bold cyan]")
    console.print("=" * 60)

    from .pipeline_e.active_cells import extract_active_cells

    modes = ["C", "S"] if mode == "both" else [mode]
    downstream_mats = {}

    for m in modes:
        console.print(f"\n[cyan]Processing mode: {m}[/cyan]")
        try:
            downstream_mat = extract_active_cells(cfg, str(merged_mat), m)
            downstream_mats[m] = downstream_mat
            console.print(f"[green]Extraction complete:[/green] {downstream_mat}")
        except Exception as e:
            console.print(f"[red]Error during extraction ({m} mode):[/red] {e}")

    # Pipeline F: Analyze
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]PIPELINE F: PLACE FIELD ANALYSIS[/bold cyan]")
    console.print("=" * 60)

    from .pipeline_f.place_fields import analyze_place_fields

    for m, mat_path in downstream_mats.items():
        console.print(f"\n[cyan]Analyzing mode: {m}[/cyan]")
        try:
            output_dir = analyze_place_fields(cfg, str(mat_path), m)
            console.print(f"[green]Analysis complete:[/green] {output_dir}")
        except Exception as e:
            console.print(f"[red]Error during analysis ({m} mode):[/red] {e}")

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]PIPELINE COMPLETE[/bold green]")
    console.print("=" * 60)
    console.print(f"\nOutput directory: [cyan]{cfg['experiment']['output_dir']}[/cyan]")


if __name__ == "__main__":
    main()
