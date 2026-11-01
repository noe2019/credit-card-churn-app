"""Simulated credit-card customer base (BankChurners-like)."""
import numpy as np
import pandas as pd

CARDS = ["Blue", "Silver", "Gold", "Platinum"]
INCOME = ["< $40K", "$40K–60K", "$60K–80K", "$80K–120K", "$120K +"]


def simulate(n=8000, seed=21):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "age": r.normal(46, 8, n).clip(26, 73).round(),
        "gender": r.choice(["F", "M"], n),
        "dependents": r.integers(0, 6, n),
        "income_category": r.choice(INCOME, n, p=[.35, .18, .14, .15, .18]),
        "card_category": r.choice(CARDS, n, p=[.9, .06, .03, .01]),
        "months_on_book": r.normal(36, 8, n).clip(13, 56).round(),
        "products_held": r.integers(1, 7, n),
        "months_inactive_12m": r.poisson(2.3, n).clip(0, 6),
        "contacts_12m": r.poisson(2.4, n).clip(0, 6),
        "credit_limit": r.lognormal(8.9, 0.75, n).clip(1400, 35000).round(),
        "revolving_balance": r.uniform(0, 2500, n).round(),
        "trans_amount_12m": r.lognormal(8.2, 0.5, n).clip(500, 18000).round(),
        "trans_count_12m": r.normal(65, 22, n).clip(10, 140).round(),
        "trans_count_change_q4_q1": r.normal(0.71, 0.22, n).clip(0, 3).round(3),
    })
    df["utilization_ratio"] = (df.revolving_balance / df.credit_limit).clip(0, 1).round(3)
    logit = (-1.6 + 0.45 * df.months_inactive_12m + 0.45 * df.contacts_12m - 0.35 * df.products_held
             - 0.035 * (df.trans_count_12m - 65) - 2.8 * (df.trans_count_change_q4_q1 - 0.7)
             - 1.2 * (df.revolving_balance > 500) - 0.00008 * (df.trans_amount_12m - 4000))
    df["churned"] = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
