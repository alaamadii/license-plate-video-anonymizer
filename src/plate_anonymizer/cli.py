"""Command-line interface."""

from pathlib import Path

import typer

app = typer.Typer(
    name="plate-anonymizer",
    help="Recall-first license-plate video anonymization.",
    no_args_is_help=True,
)


@app.command()
def anonymize(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
    device: str = typer.Option("cpu", help="cpu, cuda, or cuda:N"),
) -> None:
    """Anonymize license plates in a video (pipeline added in Phase 2)."""
    typer.echo(
        f"Pipeline foundation ready: input={input_path}, output={output_path}, device={device}"
    )


@app.command()
def evaluate() -> None:
    """Evaluate predictions against annotations (implemented in a later phase)."""
    typer.echo("Evaluation foundation ready; metrics pipeline is not implemented yet.")


@app.command()
def benchmark() -> None:
    """Benchmark processing throughput (implemented in a later phase)."""
    typer.echo("Benchmark foundation ready; benchmark runner is not implemented yet.")


if __name__ == "__main__":
    app()
