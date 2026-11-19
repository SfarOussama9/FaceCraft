"""Download the text datasets and explain how to get CelebA.

    python scripts/download_data.py --out C:/Users/<you>/datasets/facecraft      (local)
    python scripts/download_data.py --out /kaggle/working/datasets               (Kaggle)

  - Face2Text v2.1  : downloaded automatically from Zenodo (1.4 MB).
  - CelebA-Dialog   : fine-grained labels (combined_annotation.txt, 4 MB) from Google Drive. Google often
                      blocks scripted downloads ("too many users"); if so, download it in a browser.
  - CelebA          : 1.4 GB. Local: download the Kaggle zip and unzip into <out>/celeba.
                      Kaggle: attach the dataset "jessicali9530/celeba-dataset" to the notebook.
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

FACE2TEXT_URL = "https://zenodo.org/api/records/10973388/files/face2text_v2.1.zip/content"
DIALOG_DRIVE_ID = "1wZcVEjJ5LwP1Ciuc3j_RFw9Vcusj4UEU"
DIALOG_BROWSER_URL = f"https://drive.google.com/file/d/{DIALOG_DRIVE_ID}/view"
CELEBA_KAGGLE = "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"


def get_face2text(out: Path) -> None:
    dest = out / "face2text"
    if (dest / "clean_train_2.1.json").exists():
        print(f"[face2text] already present in {dest}")
        return
    print("[face2text] downloading v2.1 from Zenodo ...")
    with urllib.request.urlopen(FACE2TEXT_URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    print(f"[face2text] OK -> {dest}")


def get_dialog(out: Path) -> None:
    dest = out / "dialog" / "combined_annotation.txt"
    if dest.exists():
        print(f"[dialog] already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
        gdown.download(id=DIALOG_DRIVE_ID, output=str(dest), quiet=True)
        if not dest.exists() or dest.stat().st_size < 1_000_000:
            raise RuntimeError("download incomplete")
        print(f"[dialog] OK -> {dest}")
    except Exception as e:                       # Google Drive quota, no gdown, no internet...
        print(f"[dialog] automatic download failed ({e}).")
        print(f"         Download it in a browser: {DIALOG_BROWSER_URL}")
        print(f"         and save it as: {dest}")
        print("         (The pipeline also runs without it: Dialog captions are simply skipped.)")


def check_celeba(out: Path) -> None:
    root = out / "celeba"
    imgs = root / "img_align_celeba" / "img_align_celeba"
    if (root / "list_attr_celeba.csv").exists() and imgs.exists():
        print(f"[celeba] OK -> {root}")
    else:
        print(f"[celeba] not found in {root}.")
        print(f"         Local: download the zip from {CELEBA_KAGGLE} and unzip it into {root}")
        print("         Kaggle: 'Add Input' -> search 'celeba-dataset' (jessicali9530) -> it appears in /kaggle/input/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dataset folder (keep it OUTSIDE OneDrive/Dropbox)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    get_face2text(out)
    get_dialog(out)
    check_celeba(out)
