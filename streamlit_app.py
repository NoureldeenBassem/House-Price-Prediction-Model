import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


st.set_page_config(page_title="House Price Predictor", layout="wide")

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


@st.cache_data
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


@st.cache_resource
def train_models():
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
    return models, results


df = load_data()
models, results = train_models()
best_model_name = results.iloc[0]["Model"]

st.title("Ames House Price Predictor")
st.write("Compare models, choose the best one, then predict a house price.")

col1, col2, col3 = st.columns(3)
col1.metric("Best Model", best_model_name)
col2.metric("Best Test R2", f"{results.iloc[0]['Test R2']:.3f}")
col3.metric("Best MAE", f"${results.iloc[0]['MAE']:,.0f}")

st.subheader("Model Comparison")
show_results = results.copy()
show_results["Test R2"] = show_results["Test R2"].round(3)
show_results["MAE"] = show_results["MAE"].round(0).astype(int)
show_results["MSE"] = show_results["MSE"].round(0).astype(int)
st.dataframe(show_results, use_container_width=True, hide_index=True)

st.subheader("Graphs")

tab1, tab2, tab3 = st.tabs(["Model Scores", "Quality vs Price", "Area vs Price"])

with tab1:
    st.bar_chart(results, x="Model", y="Test R2")

with tab2:
    quality_prices = df.groupby("Overall Qual")["SalePrice"].median()
    st.bar_chart(quality_prices)

with tab3:
    st.scatter_chart(df, x="Gr Liv Area", y="SalePrice")

st.subheader("Predict Sale Price")

left, right = st.columns(2)

with left:
    overall_qual = st.slider("Overall Quality", 1, 10, 6)
    gr_liv_area = st.number_input("Living Area", 300, 6000, 1500)
    garage_cars = st.number_input("Garage Cars", 0, 5, 2)
    garage_area = st.number_input("Garage Area", 0, 1500, 400)
    total_bsmt = st.number_input("Basement Area", 0, 3000, 800)
    first_floor = st.number_input("1st Floor Area", 300, 4000, 1000)

with right:
    year_built = st.number_input("Year Built", 1870, 2026, 2000)
    full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
    year_remod = st.number_input("Year Remodeled", 1870, 2026, 2005)
    garage_year = st.number_input("Garage Year Built", 1870, 2026, 2000)
    mas_vnr_area = st.number_input("Masonry Veneer Area", 0, 2000, 100)
    chosen_model_name = st.selectbox("Model to use", results["Model"])

house = pd.DataFrame([{
    "Overall Qual": overall_qual,
    "Gr Liv Area": gr_liv_area,
    "Garage Cars": garage_cars,
    "Garage Area": garage_area,
    "Total Bsmt SF": total_bsmt,
    "1st Flr SF": first_floor,
    "Year Built": year_built,
    "Full Bath": full_bath,
    "Year Remod/Add": year_remod,
    "Garage Yr Blt": garage_year,
    "Mas Vnr Area": mas_vnr_area,
}])

if st.button("Predict Price", type="primary"):
    chosen_model = models[chosen_model_name]
    price = chosen_model.predict(house)[0]
    st.success(f"{chosen_model_name} prediction: ${price:,.0f}")

st.info(f"The best model is {best_model_name}, so this is the model we deploy.")
