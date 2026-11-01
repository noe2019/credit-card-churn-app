import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data import simulate

st.set_page_config(page_title="Credit Card Churn Prediction", page_icon="💳", layout="wide")
st.title("💳 Credit Card Churn Prediction")
st.caption("LightGBM classifier with SHAP (TreeSHAP) explanations to target proactive retention · simulated customers")
CAT = ["gender", "income_category", "card_category"]


@st.cache_resource
def train():
    df = simulate()
    X = df.drop(columns=["customer_id", "churned"]).copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    y = df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0).fit(Xtr, ytr)
    proba_te = model.predict_proba(Xte)[:, 1]
    contrib = model.predict(X, pred_contrib=True)  # TreeSHAP values (+ bias column)
    shap = pd.DataFrame(contrib[:, :-1], columns=X.columns)
    df = df.assign(churn_risk=model.predict_proba(X)[:, 1])
    return df, X, model, shap, contrib[:, -1], yte, proba_te


df, X, model, shap, base, yte, proba_te = train()

with st.sidebar:
    st.header("Filters")
    cards = st.multiselect("Card category", sorted(df.card_category.unique()), default=sorted(df.card_category.unique()))
    thr = st.slider("High-risk threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown("[Portfolio](https://noe2019.github.io/DA-Portfolio-Website/)")

view = df[df.card_category.isin(cards)]
k = st.columns(5)
k[0].metric("Customers", f"{len(view):,}")
k[1].metric("Observed churn rate", f"{view.churned.mean():.1%}")
k[2].metric("High-risk customers", f"{(view.churn_risk >= thr).sum():,}")
k[3].metric("ROC-AUC (test)", f"{roc_auc_score(yte, proba_te):.3f}")
k[4].metric("F1 (test)", f"{f1_score(yte, proba_te >= 0.5):.3f}")

t1, t2, t3 = st.tabs(["Global drivers (SHAP)", "Explain a customer", "At-risk list"])
with t1:
    imp = shap.abs().mean().sort_values()
    l, r = st.columns(2)
    l.plotly_chart(px.bar(imp, orientation="h", title="Mean |SHAP value|", labels={"value": "Impact on log-odds", "index": ""})
                   .update_layout(showlegend=False), use_container_width=True)
    feat = r.selectbox("Dependence plot for", imp.index[::-1])
    s = np.random.default_rng(0).choice(len(df), 2000, replace=False)
    xs = X[feat].iloc[s].astype(str) if feat in CAT else X[feat].iloc[s]
    r.plotly_chart(px.scatter(x=xs, y=shap[feat].iloc[s], opacity=.4, title="SHAP dependence",
                              labels={"x": feat, "y": f"SHAP value of {feat}"}), use_container_width=True)
with t2:
    top = view.sort_values("churn_risk", ascending=False).customer_id.head(200).tolist()
    cid = st.selectbox("Customer (top 200 by risk)", top)
    i = df.index[df.customer_id == cid][0]
    st.metric("Churn probability", f"{df.churn_risk[i]:.1%}")
    contrib = shap.iloc[i].sort_values(key=abs, ascending=False).head(10)[::-1]
    labels = [f"{c} = {X.iloc[i][c]}" for c in contrib.index]
    st.plotly_chart(px.bar(x=contrib.values, y=labels, orientation="h", color=contrib.values > 0,
                           color_discrete_map={True: "#C0392B", False: "#2E75B6"},
                           labels={"x": "SHAP contribution (log-odds)", "y": ""}, title="Why this customer may churn")
                    .update_layout(showlegend=False), use_container_width=True)
    st.caption("Red bars push the risk up, blue bars push it down. Base value (log-odds): %.2f" % base[i])
with t3:
    risky = view[view.churn_risk >= thr].sort_values("churn_risk", ascending=False)
    st.dataframe(risky[["customer_id", "card_category", "income_category", "months_inactive_12m", "contacts_12m",
                        "trans_count_12m", "churn_risk"]].style.format({"churn_risk": "{:.1%}"}), use_container_width=True, height=420)
    st.download_button("Download at-risk list (CSV)", risky.to_csv(index=False), "at_risk_customers.csv", "text/csv")
