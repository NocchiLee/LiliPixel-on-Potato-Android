"""Download the three upstream assets to the directory containing runner.py."""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

BASE = Path(__file__).resolve().parent
DIT = "https://huggingface.co/Pinkstack/LiliPixel-V1-190M/resolve/5e89a11268b86627c399bb070047155e627ad555/"
ENC = "https://huggingface.co/MongoDB/voyage-4-nano/resolve/587d8d6dcaec029a9779c2098d6af020e4a07c01/"
ASSETS = [
    ("dit.st", DIT + "transformer/diffusion_pytorch_model.safetensors", "14c8390efb2fd9309655dc6d49fe2f1dcb3db7ec09360fb41ba0d12719b844f8"),
    ("enc.st", ENC + "model.safetensors", "3dae0c63c81dcab79ac213af331940a1bf2b8a53ec8646be878552890291ad30"),
    ("tokenizer.json", ENC + "tokenizer.json", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
]

def fetch(name, url, expected):
    target = BASE / name
    if target.exists():
        with target.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == expected:
                print(name, "verified")
                return
        raise SystemExit(f"{name} has a different digest. Move it aside before downloading.")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=BASE, suffix=".part", delete=False) as f:
            temp = Path(f.name)
            digest = hashlib.sha256()
            with urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {name}")
        os.replace(temp, target)
        print(name, "verified")
    finally:
        if temp:
            temp.unlink(missing_ok=True)

if __name__ == "__main__":
    for asset in ASSETS:
        fetch(*asset)
