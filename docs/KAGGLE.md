# Running FaceCraft on Kaggle

Kaggle gives each account ~30 free GPU hours per week (T4 x2 or P100). Sessions stop after 12 h,
so every training script saves `last.pt` regularly and **resumes automatically** when re-run.

## One-time setup
1. **Upload the CelebA-Dialog labels** as a private dataset:
   Kaggle → *Datasets* → *New Dataset* → upload `combined_annotation.txt` → name it **`facecraft-dialog`**.
2. **Get the code on Kaggle** (pick one):
   - GitHub: `!git clone https://github.com/<user>/FaceCraft.git` (private repos need a token in Kaggle *Secrets*), or
   - Zip the repo (without `datasets/`, `outputs/`) and upload it as a dataset `facecraft-code`.

## In a Kaggle notebook (Settings: GPU T4 or P100, Internet ON)
1. *Add Input* → search **`celeba-dataset`** (by jessicali9530) → add. Also add **`facecraft-dialog`**.
2. Cells:
```bash
!git clone https://github.com/<user>/FaceCraft.git   # or: !cp -r /kaggle/input/facecraft-code/FaceCraft .
%cd FaceCraft
!pip install -q -r requirements.txt
!python scripts/download_data.py --out /kaggle/working/datasets        # Face2Text (Zenodo)
!python -m data.preprocess --config configs/data_kaggle.yaml --workers 4   # 64x64 cache, ~5-10 min
```
3. Training (each script resumes from `outputs/<run>/last.pt` if it exists):
```bash
!sed -i 's#configs/data.yaml#configs/data_kaggle.yaml#' configs/classifier.yaml configs/gan.yaml
!python -m training.train_classifier --config configs/classifier.yaml
!python -m training.train_gan --config configs/gan.yaml
```
4. For long runs use **Save Version → Save & Run All (Commit)**: it runs in the background even if you
   close the browser. Afterwards download `/kaggle/working/FaceCraft/outputs/` from the *Output* tab.
5. To continue a stopped run in a new session: add the previous version's output as an input, copy its
   `outputs/` folder into the repo, and run the same command again.

## The Deliverable 2 notebook
Put the downloaded `outputs/` folder next to the repo, open `notebooks/01_data_and_preprocessing.ipynb`
and run all cells (`RUN_TRAINING = False` loads the saved results instead of retraining).
