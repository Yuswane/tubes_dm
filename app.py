import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")
st.title("Bank Marketing Classification (Random Forest)")
st.caption("Dataset: bank-full.csv | Prediksi y = yes/no")

# -----------------------------
# Robust CSV reader (handles ; delimiter)
# -----------------------------
def robust_read_csv(file) -> pd.DataFrame:
    df_try = pd.read_csv(file)
    if df_try.shape[1] == 1:
        file.seek(0)
        df_try = pd.read_csv(file, sep=";")
    return df_try

def build_preprocessor(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    if "y" in categorical_features:
        categorical_features.remove("y")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor, numeric_features, categorical_features

@st.cache_resource
def train_model(df: pd.DataFrame):
    df = df.copy()

    if "y" not in df.columns:
        raise ValueError("Kolom target 'y' tidak ditemukan. Pastikan dataset bank-full.csv versi Bank Marketing (UCI).")

    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    if y.isna().any():
        bad = df.loc[y.isna(), "y"].unique().tolist()
        raise ValueError(f"Nilai kolom y tidak sesuai mapping yes/no. Ditemukan: {bad}")

    preprocessor, numeric_features, categorical_features = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    params = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None]
    }

    grid = GridSearchCV(pipe, params, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    meta = {
        "df_head": df.head(10),
        "shape": df.shape,
        "X_cols": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_params": grid.best_params_,
        "cv_acc": float(grid.best_score_),
    }
    return grid.best_estimator_, meta

def build_input_ui(df_head: pd.DataFrame, X_cols, numeric_features, categorical_features, full_df: pd.DataFrame):
    st.subheader("Input Nasabah")

    defaults_num = full_df[numeric_features].median(numeric_only=True) if numeric_features else pd.Series(dtype=float)
    defaults_cat = full_df[categorical_features].mode().iloc[0] if categorical_features else pd.Series(dtype=object)

    user_data = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Numeric")
        for c in numeric_features:
            base = float(defaults_num[c]) if c in defaults_num else 0.0
            user_data[c] = st.number_input(c, value=base)

    with col2:
        st.markdown("### Categorical")
        for c in categorical_features:
            options = [str(x) for x in full_df[c].dropna().unique().tolist()]
            if not options:
                user_data[c] = st.text_input(c, value="")
                continue
            default_val = str(defaults_cat[c]) if c in defaults_cat else options[0]
            if default_val not in options:
                default_val = options[0]
            user_data[c] = st.selectbox(c, options=options, index=options.index(default_val))

    with col3:
        st.markdown("### Lainnya")
        other_cols = [c for c in X_cols if c not in numeric_features and c not in categorical_features]
        if not other_cols:
            st.write("Tidak ada.")
        for c in other_cols:
            example = ""
            if full_df[c].notna().any():
                example = str(full_df[c].dropna().iloc[0])
            user_data[c] = st.text_input(c, value=example)

    return pd.DataFrame([user_data], columns=X_cols)

# -----------------------------
# Data source: file upload OR local bank-full.csv
# -----------------------------
with st.sidebar:
    st.header("Sumber Data")
    uploaded = st.file_uploader("Upload bank-full.csv (opsional)", type=["csv"])
    st.caption("Kalau tidak upload, app akan cari file lokal: bank-full.csv")

try:
    if uploaded is not None:
        df = robust_read_csv(uploaded)
    else:
        # local file
        df = pd.read_csv("bank-full.csv", sep=";")  # bank-full.csv umumnya pakai ';'
except Exception as e:
    st.error(f"Gagal baca dataset: {e}")
    st.stop()

# Train model
try:
    model, meta = train_model(df)
except Exception as e:
    st.error(f"Gagal training model: {e}")
    st.stop()

with st.expander("Info Model", expanded=False):
    st.write("Dataset shape:", meta["shape"])
    st.write("Best Params:", meta["best_params"])
    st.write("CV Accuracy (mean):", round(meta["cv_acc"], 4))
    st.dataframe(meta["df_head"], use_container_width=True)

# Input UI
input_df = build_input_ui(meta["df_head"], meta["X_cols"], meta["numeric_features"], meta["categorical_features"], df)

st.divider()

# Predict
if st.button("Prediksi", type="primary"):
    try:
        proba_yes = float(model.predict_proba(input_df)[:, 1][0])
        pred = int(model.predict(input_df)[0])
        label = "yes" if pred == 1 else "no"

        a, b = st.columns(2)
        with a:
            st.metric("Prediksi (y)", label)
        with b:
            st.metric("Probabilitas (yes)", f"{proba_yes:.3f}")

        st.write("Input yang dipakai:")
        st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal prediksi: {e}")
