import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


# ── Data loading & cleaning (mirrors the notebook pipeline) ──────────────────

def load_data():
    data = pd.read_csv("AmesHousing.csv")
    df = data.copy()

    # Drop columns with more than 40 % missing values
    threshold = 0.40 * len(df)
    cols_to_drop = df.columns[df.isna().sum() > threshold]
    df = df.drop(columns=cols_to_drop)

    # Fill missing numerical values with the median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing categorical values with 'None' (feature doesn't exist)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("None")

    # One-hot encode categorical features
    df_final = pd.get_dummies(df, drop_first=True)

    # Select features with correlation > 0.5 with SalePrice
    corr_matrix = df_final.corr(numeric_only=True)
    top_corr = corr_matrix["SalePrice"].abs().sort_values(ascending=False)
    top_features = top_corr[top_corr > 0.5].index.tolist()

    return df_final[top_features]


# ── Split ────────────────────────────────────────────────────────────────────

df = load_data()
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

FEATURES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scale X ──────────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Polynomial features (degree 2) ───────────────────────────────────────────

poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train_scaled)
X_test_poly2  = poly2.transform(X_test_scaled)

# ── Scale y separately for SVR ───────────────────────────────────────────────

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# ── Train all models ─────────────────────────────────────────────────────────

rows = []

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
rows.append({
    "Model": "Linear Regression",
    "Train R2": r2_score(y_train, lr.predict(X_train_scaled)),
    "Test R2":  r2_score(y_test,  lr.predict(X_test_scaled)),
    "MAE": mean_absolute_error(y_test, lr.predict(X_test_scaled)),
    "MSE": mean_squared_error(y_test,  lr.predict(X_test_scaled)),
})

# 2. Polynomial Degree 2
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly2, y_train)
rows.append({
    "Model": "Polynomial Degree 2",
    "Train R2": r2_score(y_train, poly_lr.predict(X_train_poly2)),
    "Test R2":  r2_score(y_test,  poly_lr.predict(X_test_poly2)),
    "MAE": mean_absolute_error(y_test, poly_lr.predict(X_test_poly2)),
    "MSE": mean_squared_error(y_test,  poly_lr.predict(X_test_poly2)),
})

# 3. Ridge Regression (alpha=10, Poly2)
ridge = Ridge(alpha=10)
ridge.fit(X_train_poly2, y_train)
rows.append({
    "Model": "Ridge Regression (α=10)",
    "Train R2": r2_score(y_train, ridge.predict(X_train_poly2)),
    "Test R2":  r2_score(y_test,  ridge.predict(X_test_poly2)),
    "MAE": mean_absolute_error(y_test, ridge.predict(X_test_poly2)),
    "MSE": mean_squared_error(y_test,  ridge.predict(X_test_poly2)),
})

# 4. Lasso Regression (alpha=10, Poly2)
lasso = Lasso(alpha=10, max_iter=10000)
lasso.fit(X_train_poly2, y_train)
rows.append({
    "Model": "Lasso Regression (α=10)",
    "Train R2": r2_score(y_train, lasso.predict(X_train_poly2)),
    "Test R2":  r2_score(y_test,  lasso.predict(X_test_poly2)),
    "MAE": mean_absolute_error(y_test, lasso.predict(X_test_poly2)),
    "MSE": mean_squared_error(y_test,  lasso.predict(X_test_poly2)),
})

# 5. SVR (rbf) — scale y, predict, inverse-transform
svr = SVR(kernel="rbf", C=100, epsilon=0.1)
svr.fit(X_train_scaled, y_train_scaled)
y_pred_svr_test  = y_scaler.inverse_transform(svr.predict(X_test_scaled).reshape(-1, 1)).ravel()
y_pred_svr_train = y_scaler.inverse_transform(svr.predict(X_train_scaled).reshape(-1, 1)).ravel()
rows.append({
    "Model": "SVR (rbf)",
    "Train R2": r2_score(y_train, y_pred_svr_train),
    "Test R2":  r2_score(y_test,  y_pred_svr_test),
    "MAE": mean_absolute_error(y_test, y_pred_svr_test),
    "MSE": mean_squared_error(y_test,  y_pred_svr_test),
})

# 6. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rows.append({
    "Model": "Random Forest",
    "Train R2": r2_score(y_train, rf.predict(X_train)),
    "Test R2":  r2_score(y_test,  rf.predict(X_test)),
    "MAE": mean_absolute_error(y_test, rf.predict(X_test)),
    "MSE": mean_squared_error(y_test,  rf.predict(X_test)),
})

# ── Report ───────────────────────────────────────────────────────────────────

results = pd.DataFrame(rows).sort_values("Test R2", ascending=False)
best_model = results.iloc[0]["Model"]

print(results.round(3).to_string(index=False))
print()
print(f"Best model to deploy: {best_model}")
print(f"Features used ({len(FEATURES)}): {FEATURES}")
