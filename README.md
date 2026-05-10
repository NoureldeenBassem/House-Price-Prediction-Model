# Machine Learning Project

This folder contains the complete house price machine learning project.

## Files

- `House_Price_Analysis_Final_Noureldin_Bassem.ipynb`: original notebook
- `AmesHousing.csv`: dataset
- `train_models.py`: simple model comparison script
- `streamlit_app.py`: deployed Streamlit app
- `requirements.txt`: Python libraries
- `.venv`: local virtual environment

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

## Simple Explanation

The project loads the Ames housing dataset, selects the most important numeric
features, fills missing values using the median, trains several regression
models, compares them using Test R2 and MAE, then deploys the best model in
Streamlit.
