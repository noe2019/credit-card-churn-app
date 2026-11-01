# Credit Card Churn Prediction

Identifies credit-card customers likely to close their account so retention teams can act early, with SHAP explanations.

**Live app:** https://noe-cc-churn-49c734693e6c.herokuapp.com/
**Portfolio:** https://noe2019.github.io/DA-Portfolio-Website/

## Features
- LightGBM classifier
- Global SHAP importance (TreeSHAP)
- Per-customer explanation (SHAP contributions)
- At-risk customer list export

## Tech stack
Python, LightGBM, SHAP (TreeSHAP), Streamlit

## Data
All data in this app is **simulated** (generated in `data.py` with a fixed random seed) to reproduce the structure of the original project without exposing real or confidential records.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Heroku (stack `heroku-24`, Python 3.12) using the `Procfile` in this repository.

---
Author: Noé Carème Fouotsa Manfouo
