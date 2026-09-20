# Setup and reproducibility

## Portable installation

Use Python 3.11 or newer in an isolated virtual environment inside the repository:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
# .venv\Scripts\activate.bat
# Linux/macOS:
# source .venv/bin/activate
python -m pip install -e ".[dev,yolo,media]"
python scripts/download_model.py
```

If PowerShell activation is disabled, call .venv\Scripts\python.exe directly;
no system execution-policy change is needed. run.cmd uses the repository venv,
then the sibling venv used during local development, then Python on PATH.
On other systems use the plate-anonymizer CLI or python run.py.

YOLO is optional for evaluation/tests. Install only .[dev] for CI and .[media] for
bundled FFmpeg. CUDA additionally requires a compatible PyTorch/CUDA installation;
the Windows CPU snapshot is not a CUDA environment lock.

## Recorded environment

constraints/windows-cpu-py313.txt records exact installed versions of the project's
runtime/test dependency closure from the local Windows/Python 3.13 environment.
It is a platform-specific constraints snapshot, not a universal lock. A clean
Windows/Python 3.13 environment was installed with the dev extra and these
constraints: all 28 tests, Ruff, the evaluation example and pip check passed.
The full YOLO/media extras were exercised in the existing development environment,
not reinstalled in the clean test environment. Use the snapshot with:

```bash
python -m pip install -c constraints/windows-cpu-py313.txt -e ".[dev,yolo,media]"
```

For a production handover, build and test a dedicated environment on the chosen
CUDA platform and freeze its resolved dependencies and container digest. Record
the model checksum and exact processing command with each experiment. Do not
freeze unrelated packages from a general workstation.

## Container

```bash
docker build -t plate-anonymizer .
docker run --rm -v /absolute/workspace:/data plate-anonymizer anonymize --input /data/input.mp4 --output /data/output.mp4 --model /data/plate_detector.pt --redaction solid --preserve-audio
```

The image installs detector/media extras; models and footage must be mounted.
The Dockerfile is provided but has not been built locally. It is not a pinned
CUDA production image. Image context excludes local media, weights and secrets.

## GitHub handoff

Track source, tests, examples, documentation, model provenance and constraints.
Do not add input/, output/, model binaries, virtual environments or local failure
crops. Synthetic examples are safe format fixtures, not accuracy evidence.
Run Ruff, pytest and the README evaluation example before pushing.
