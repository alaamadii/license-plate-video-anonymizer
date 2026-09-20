"""Download the tested public plate checkpoint; never upload local footage."""

import hashlib
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
SHA256 = "2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822"


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "models" / "plate_detector.pt"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
            print(f"Verified: {destination}")
            return
        raise RuntimeError(f"Existing model differs; move it aside first: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "plate-anonymizer"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("Downloaded model checksum changed; review upstream before using it.")
    destination.write_bytes(data)
    print(f"Downloaded and verified: {destination}")


if __name__ == "__main__":
    main()
