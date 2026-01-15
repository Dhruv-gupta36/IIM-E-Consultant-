# %%
import pandas as pd
data = pd.DataFrame({
    "land_assets": [0.8, 0.6, 0.9, 0.3],
    "crop_sales": [0.7, 0.5, 0.8, 0.2],
    "input_repayments": [0.9, 0.4, 0.85, 0.1],
    "bill_discipline": [0.6, 0.3, 0.7, 0.2],
    # target: repaid loan (1=yes, 0=no)
    "repaid": [1, 0, 1, 0] 
})
from sklearn.linear_model import Ridge
X_wealth = data[[
    "land_assets", "crop_sales",
    "input_repayments", "bill_discipline"]]
y_wealth = data["repaid"]
wealth_model = Ridge(alpha=1.0)
wealth_model.fit(X_wealth, y_wealth)
from sklearn.linear_model import LogisticRegression
punctuality_model = LogisticRegression()
punctuality_model.fit(X_wealth, y_wealth)
from sklearn.ensemble import GradientBoostingClassifier
trust_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3)
trust_model.fit(X_wealth, y_wealth)
new_user = pd.DataFrame({
    "land_assets": [0.75],
    "crop_sales": [0.65],
    "input_repayments": [0.8],
    "bill_discipline": [0.6]
})
wealth_score = wealth_model.predict(new_user)[0]
punctuality_score = punctuality_model.predict_proba(new_user)[0][1]
trust_score = trust_model.predict_proba(new_user)[0][1]
EcoScore = 500 * (
    0.35 * wealth_score +
    0.40 * punctuality_score +
    0.25 * trust_score
)
EcoScore = round(EcoScore, 2)
print("EcoScore:", EcoScore)
def score_band(score):
    if score < 250:
        return "Poor"
    elif score < 350:
        return "Fair"
    elif score < 450:
        return "Good"
    else:
        return "Prime"
print("Category:", score_band(EcoScore))


