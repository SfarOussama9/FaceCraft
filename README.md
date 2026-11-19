# FaceCraft

**Text-to-face generation with a diffusion model and a Transformer text encoder trained from scratch.**
CS495 Deep Learning · Fall 2026 · MedTech / SMU.

Team: Oussama Sfar, Ines Turki, Feriel Charrad, Yasmine Jedidi.

> Type *"a smiling young woman with blond hair and glasses"* → our model generates a matching face, starting from pure noise.

## Status
- [x] Deliverable 1 — Project proposal
- [ ] Deliverable 2 — Dataset & preprocessing notebook + first baseline results (due 22 Nov)
- [ ] Deliverable 3 — Final code, report, presentation & demo (21–23 Dec)

## Setup
```bash
pip install -r requirements.txt
```
Data download and preprocessing instructions: see `notebooks/01_data_and_preprocessing.ipynb` (coming for Deliverable 2).

## Repository layout
| Folder | Contents |
|---|---|
| `data/` | dataset download/prep, caption engine, PyTorch datasets |
| `models/` | U-Net, text encoder, diffusion process, GAN, attribute classifier |
| `training/` | training scripts |
| `evaluation/` | FID/KID, attribute accuracy, compositional test, attention maps |
| `demo/` | Gradio app |
| `notebooks/` | deliverable and results notebooks |
| `configs/` | one YAML config per experiment |
| `docs/` | design document and plans |

## Datasets
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset): 202,599 faces, 40 attributes
- [CelebA-Dialog](https://github.com/ziqihuangg/CelebA-Dialog): fine-grained attribute levels
- [Face2Text v2](https://zenodo.org/records/6583553): human-written face descriptions

All datasets are used under their non-commercial research licenses. Generated faces are synthetic.
