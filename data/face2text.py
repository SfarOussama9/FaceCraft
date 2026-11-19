"""Face2Text v2.1: descriptions of CelebA faces written by real people (Tanti et al., 2022).

Each JSON file is a list of {"response_id", "filename", "description"}. "filename" is the CelebA
image name, so the text joins directly onto CelebA. Some images have several descriptions.
We ignore Face2Text's own train/dev/test files' meaning: the project re-splits everything by the
official CelebA partition, so we simply merge all files here.
"""
import json
from collections import defaultdict


def load_face2text(paths: list) -> dict:
    """Return {image_id: [description, ...]} merged from all given JSON files; empty texts are dropped."""
    captions = defaultdict(list)
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                text = (row.get("description") or "").strip()
                if text:
                    captions[row["filename"]].append(text)
    return dict(captions)
