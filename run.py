"""Windows-friendly launcher: run.cmd INPUT [--output OUTPUT] [--redaction blur]."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and hide vehicle license plates.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--redaction", choices=["solid", "blur", "pixelate"], default="solid")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = args.input.resolve()
    output = (
        args.output.resolve() if args.output else root / "output" / f"{source.stem}_redacted.mp4"
    )
    python = Path(sys.executable)
    return subprocess.call(
        [
            str(python),
            "-m",
            "plate_anonymizer.cli",
            "anonymize",
            "--input",
            str(source),
            "--output",
            str(output),
            "--model",
            str(root / "models" / "plate_detector.pt"),
            "--device",
            "cpu",
            "--image-size",
            "1280",
            "--redaction",
            args.redaction,
            "--preserve-audio",
        ],
        cwd=root,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


if __name__ == "__main__":
    sys.exit(main())
