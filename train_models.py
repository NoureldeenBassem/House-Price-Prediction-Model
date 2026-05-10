import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


FEATURES = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Garage Area",
    "Total Bsmt SF",
    "1st Flr SF",
    "Year Built",
    "Full Bath",
    "Year Remod/Add",
    "Garage Yr Blt",
    "Mas Vnr Area",
]


def load_data():
    df = pd.read_csv("AmesHousing.csv")
    df = df[FEATURES + ["SalePrice"]]
    df = df.fillna(df.median(numeric_only=True))
    return df


def make_models():
    return {
        "Linear Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Polynomial Degree 2": Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            ("model", Ridge(alpha=10)),
        ]),
        "Lasso Regression": Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            ("model", Lasso(alpha=10, max_iter=10000)),
        ]),
        "SVR": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, epsilon=5000)),
        ]),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }


df = load_data()
X = df[FEATURES]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = make_models()
rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rows.append({
        "Model": name,
        "Test R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
    })

results = pd.DataFrame(rows).sort_values("Test R2", ascending=False)
best_model = results.iloc[0]["Model"]

print(results.round(3).to_string(index=False))
print()
print(f"Best model to deploy: {best_model}")
