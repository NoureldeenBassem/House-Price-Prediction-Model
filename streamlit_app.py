import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


st.set_page_config(page_title="House Price Predictor", layout="wide")


# ── Data loading & cleaning (mirrors the notebook pipeline) ──────────────────

@st.cache_data
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


@st.cache_resource
def train_models():
    df = load_data()
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale X
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Polynomial features (degree 2)
    poly2 = PolynomialFeatures(degree=2)
    X_train_poly2 = poly2.fit_transform(X_train_scaled)
    X_test_poly2  = poly2.transform(X_test_scaled)

    # Scale y for SVR
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    rows = []
    trained = {}   # stores everything needed to predict later

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
    trained["Linear Regression"] = ("scaled", lr, None)

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
    trained["Polynomial Degree 2"] = ("poly2", poly_lr, None)

    # 3. Ridge Regression
    ridge = Ridge(alpha=10)
    ridge.fit(X_train_poly2, y_train)
    rows.append({
        "Model": "Ridge Regression (α=10)",
        "Train R2": r2_score(y_train, ridge.predict(X_train_poly2)),
        "Test R2":  r2_score(y_test,  ridge.predict(X_test_poly2)),
        "MAE": mean_absolute_error(y_test, ridge.predict(X_test_poly2)),
        "MSE": mean_squared_error(y_test,  ridge.predict(X_test_poly2)),
    })
    trained["Ridge Regression (α=10)"] = ("poly2", ridge, None)

    # 4. Lasso Regression
    lasso = Lasso(alpha=10, max_iter=10000)
    lasso.fit(X_train_poly2, y_train)
    rows.append({
        "Model": "Lasso Regression (α=10)",
        "Train R2": r2_score(y_train, lasso.predict(X_train_poly2)),
        "Test R2":  r2_score(y_test,  lasso.predict(X_test_poly2)),
        "MAE": mean_absolute_error(y_test, lasso.predict(X_test_poly2)),
        "MSE": mean_squared_error(y_test,  lasso.predict(X_test_poly2)),
    })
    trained["Lasso Regression (α=10)"] = ("poly2", lasso, None)

    # 5. SVR — scale y, predict, inverse-transform
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
    trained["SVR (rbf)"] = ("svr", svr, y_scaler)

    # 6. Random Forest (no scaling needed)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rows.append({
        "Model": "Random Forest",
        "Train R2": r2_score(y_train, rf.predict(X_train)),
        "Test R2":  r2_score(y_test,  rf.predict(X_test)),
        "MAE": mean_absolute_error(y_test, rf.predict(X_test)),
        "MSE": mean_squared_error(y_test,  rf.predict(X_test)),
    })
    trained["Random Forest"] = ("raw", rf, None)

    results = pd.DataFrame(rows).sort_values("Test R2", ascending=False)

    # Return everything needed for predictions
    artifacts = {
        "scaler": scaler,
        "poly2": poly2,
        "feature_names": list(X.columns),
        "trained": trained,
    }
    return results, artifacts


# ── Load & train ─────────────────────────────────────────────────────────────

df = load_data()
results, artifacts = train_models()
best_model_name = results.iloc[0]["Model"]
FEATURES = artifacts["feature_names"]

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Ames House Price Predictor")
st.write("Compare models, choose the best one, then predict a house price.")

col1, col2, col3 = st.columns(3)
col1.metric("Best Model", best_model_name)
col2.metric("Best Test R²", f"{results.iloc[0]['Test R2']:.3f}")
col3.metric("Best MAE", f"${results.iloc[0]['MAE']:,.0f}")

st.subheader("Model Comparison")
show = results.copy()
show["Train R2"] = show["Train R2"].round(3)
show["Test R2"]  = show["Test R2"].round(3)
show["MAE"]      = show["MAE"].round(0).astype(int)
show["MSE"]      = show["MSE"].round(0).astype(int)
show["Overfit Gap"] = (show["Train R2"] - show["Test R2"]).round(3)
st.dataframe(show, use_container_width=True, hide_index=True)

st.subheader("Graphs")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Test R² Scores", "Train vs Test R²", "Quality vs Price", "Area vs Price"]
)

with tab1:
    st.bar_chart(results, x="Model", y="Test R2")

with tab2:
    chart_data = results.set_index("Model")[["Train R2", "Test R2"]]
    st.bar_chart(chart_data)

with tab3:
    raw_df = pd.read_csv("AmesHousing.csv")
    quality_prices = raw_df.groupby("Overall Qual")["SalePrice"].median()
    st.bar_chart(quality_prices)

with tab4:
    raw_df = pd.read_csv("AmesHousing.csv")
    st.scatter_chart(raw_df, x="Gr Liv Area", y="SalePrice")

# ── Predict ───────────────────────────────────────────────────────────────────

st.subheader("Predict Sale Price")
st.caption(
    f"The predictor uses the {len(FEATURES)} features selected by the notebook "
    f"(correlation > 0.5 with SalePrice). Numeric inputs use dataset medians as defaults; "
    f"boolean features (one-hot encoded) default to 0 (False)."
)

# Compute medians from the cleaned/encoded dataset for sensible defaults
feature_medians = df[FEATURES].median()

left, right = st.columns(2)

with left:
    overall_qual  = st.slider("Overall Quality (1–10)", 1, 10,
                              int(feature_medians.get("Overall Qual", 6)))
    gr_liv_area   = st.number_input("Above-Ground Living Area (sq ft)", 300, 6000,
                                    int(feature_medians.get("Gr Liv Area", 1500)))
    garage_cars   = st.number_input("Garage Cars", 0, 5,
                                    int(feature_medians.get("Garage Cars", 2)))
    garage_area   = st.number_input("Garage Area (sq ft)", 0, 1500,
                                    int(feature_medians.get("Garage Area", 400)))
    total_bsmt    = st.number_input("Total Basement Area (sq ft)", 0, 3000,
                                    int(feature_medians.get("Total Bsmt SF", 800)))
    first_floor   = st.number_input("1st Floor Area (sq ft)", 300, 4000,
                                    int(feature_medians.get("1st Flr SF", 1000)))
    year_built    = st.number_input("Year Built", 1870, 2026,
                                    int(feature_medians.get("Year Built", 1973)))

with right:
    full_bath     = st.number_input("Full Bathrooms", 0, 5,
                                    int(feature_medians.get("Full Bath", 2)))
    year_remod    = st.number_input("Year Remodeled", 1870, 2026,
                                    int(feature_medians.get("Year Remod/Add", 1994)))
    garage_yr     = st.number_input("Garage Year Built", 1870, 2026,
                                    int(feature_medians.get("Garage Yr Blt", 1980)))
    mas_vnr_area  = st.number_input("Masonry Veneer Area (sq ft)", 0, 2000,
                                    int(feature_medians.get("Mas Vnr Area", 0)))

    st.markdown("**One-hot encoded features** (check if applicable)")
    exter_qual_ta   = st.checkbox("Exterior Quality = TA (Typical/Average)",
                                  value=bool(feature_medians.get("Exter Qual_TA", 0)))
    kitchen_qual_ta = st.checkbox("Kitchen Quality = TA (Typical/Average)",
                                  value=bool(feature_medians.get("Kitchen Qual_TA", 0)))
    foundation_pconc = st.checkbox("Foundation = Poured Concrete",
                                   value=bool(feature_medians.get("Foundation_PConc", 0)))

    chosen_model_name = st.selectbox("Model to use", results["Model"])

# Build input row aligned to exactly the feature columns
input_values = {
    "Overall Qual":    overall_qual,
    "Gr Liv Area":     gr_liv_area,
    "Garage Cars":     garage_cars,
    "Garage Area":     garage_area,
    "Total Bsmt SF":   total_bsmt,
    "1st Flr SF":      first_floor,
    "Exter Qual_TA":   int(exter_qual_ta),
    "Year Built":      year_built,
    "Full Bath":       full_bath,
    "Year Remod/Add":  year_remod,
    "Kitchen Qual_TA": int(kitchen_qual_ta),
    "Foundation_PConc": int(foundation_pconc),
    "Garage Yr Blt":   garage_yr,
    "Mas Vnr Area":    mas_vnr_area,
}

house = pd.DataFrame([{f: input_values.get(f, 0) for f in FEATURES}])

if st.button("Predict Price", type="primary"):
    mode, model, y_sc = artifacts["trained"][chosen_model_name]
    scaler = artifacts["scaler"]
    poly2  = artifacts["poly2"]

    if mode == "raw":
        price = model.predict(house)[0]
    elif mode == "scaled":
        price = model.predict(scaler.transform(house))[0]
    elif mode == "poly2":
        price = model.predict(poly2.transform(scaler.transform(house)))[0]
    elif mode == "svr":
        scaled_pred = model.predict(scaler.transform(house)).reshape(-1, 1)
        price = y_sc.inverse_transform(scaled_pred).ravel()[0]

    st.success(f"**{chosen_model_name}** prediction: **${price:,.0f}**")

st.info(
    f"The best-performing model is **{best_model_name}** "
    f"(Test R² = {results.iloc[0]['Test R2']:.3f}, "
    f"MAE = ${results.iloc[0]['MAE']:,.0f})."
)
