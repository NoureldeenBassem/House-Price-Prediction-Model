# Machine Learning Project — Ames House Price Prediction

**Author: Noureldin Bassem Mohamed**

## Files

- `Ames_Housing_Analysis.ipynb` — full end-to-end notebook (EDA → models → evaluation)
- `AmesHousing.csv` — Ames Housing dataset (2,930 houses, 82 columns)
- `train_models.py` — standalone model comparison script
- `streamlit_app.py` — interactive Streamlit web app
- `requirements.txt` — Python libraries
- `.venv` — local virtual environment

## Run The App

```bash
cd "/Users/fariskishtah/Downloads/Machine Learning Project"
source .venv/bin/activate
streamlit run streamlit_app.py
```

## Run The Model Comparison

```bash
cd "/Users/fariskishtah/Downloads/Machine Learning Project"
source .venv/bin/activate
python train_models.py
```

## Open The Notebook

```bash
cd "/Users/fariskishtah/Downloads/Machine Learning Project"
source .venv/bin/activate
jupyter notebook
```

## Pipeline Overview

The pipeline follows the notebook exactly:

1. **Load** the Ames Housing dataset.
2. **Clean** the data:
   - Drop columns with > 40 % missing values (Alley, Pool QC, Fence, etc.).
   - Fill missing numerical values with the **median** (robust to outliers).
   - Fill missing categorical values with **`'None'`** (feature doesn't exist).
3. **One-hot encode** all categorical columns (`drop_first=True` to avoid multicollinearity).
4. **Select features** — keep only those with an absolute correlation > 0.5 with `SalePrice`
   (currently 14 features including one-hot encoded dummies).
5. **Split** — 80 % train / 20 % test (`random_state=42`).
6. **Scale X** with `StandardScaler` (fit on train, transform both).
7. **Train 6 models** and evaluate with Test R², MAE, and MSE:

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline — scaled X |
| Polynomial Degree 2 | Quadratic terms on scaled X |
| Ridge Regression (α=10) | L2 regularization on Poly2 features |
| Lasso Regression (α=10) | L1 regularization + automatic feature selection on Poly2 |
| SVR (rbf, C=100, ε=0.1) | Non-linear; **both X and y are scaled**; predictions inverse-transformed |
| Random Forest | 100 trees; no scaling needed; best overall performer |

8. **Deploy** the best model in the Streamlit app with an interactive prediction form.

## Key Findings (from the Notebook)

- **Overall Quality** (corr = 0.80) and **Living Area** (corr = 0.71) are the top price drivers.
- Most houses are priced between **$100,000 and $250,000** (right-skewed distribution).
- **Random Forest** achieves the highest Test R² and lowest MAE among all models.
- Polynomial Degree 3 **overfits** badly — Degree 2 with Ridge/Lasso is the best linear approach.
- SVR requires scaling the target variable (`SalePrice`) separately before training.
