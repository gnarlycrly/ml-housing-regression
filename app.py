import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import boto3, os
from pathlib import Path

# ============================
# Minimal Light Soft Pink Theme
# ============================
st.markdown("""
    <style>
        /* Root theme colors */
        :root {
            --accent-pink: #ff7eb6;        /* soft bright pink */
            --accent-pink-light: #ffd9ec;  /* pale pastel pink */
            --text-dark: #2d2d2d;
            --text-light: #5f5f5f;
            --bg-light: #ffffff;
            --bg-soft: #fafafa;
        }

        /* Page background */
        .main {
            background-color: var(--bg-soft) !important;
        }

        /* Titles */
        h1, h2, h3, h4 {
            font-family: "Poppins", sans-serif !important;
            color: var(--text-dark) !important;
            letter-spacing: -0.5px;
        }

        h1 {
            font-size: 2.4rem !important;
            color: var(--accent-pink) !important;
        }

        h2, h3 {
            font-weight: 600 !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: var(--accent-pink) !important;
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 15px;
            font-weight: 600;
            transition: 0.2s;
        }

        .stButton>button:hover {
            background-color: #ff66a8 !important;
            transform: translateY(-1px);
        }

        /* Selectbox labels */
        .stSelectbox label {
            color: var(--text-light) !important;
            font-weight: 600 !important;
        }

        /* Dataframes */
        .dataframe {
            border-radius: 10px;
            border: 1px solid var(--accent-pink-light);
            background-color: var(--bg-light);
            overflow: hidden;
        }

        /* Metric cards */
        [data-testid="metric-container"] {
            background-color: white !important;
            border-radius: 12px !important;
            border: 1px solid var(--accent-pink-light) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            padding: 15px !important;
        }

        /* Chart wrapper */
        .js-plotly-plot, .plot-container {
            border-radius: 10px !important;
            border: 1px solid var(--accent-pink-light);
            background-color: white;
            padding: 5px;
        }

        /* Info + warning boxes */
        .st-info, .st-warning {
            border-left: 4px solid var(--accent-pink) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "ml-housing-regression-data")
REGION = os.getenv("AWS_REGION", "us-west-1")

s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        st.info(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# Paths (ensure available locally by fetching from S3 if missing)
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv"
)
HOLDOUT_META_PATH = load_from_s3(
    "processed/cleaning_holdout.csv",
    "data/processed/cleaning_holdout.csv"
)

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp

fe_df, disp_df = load_data()

# ============================
# UI
# ============================
st.title("✨ Housing Price Prediction — Holdout Explorer 🏠")

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

if st.button("Show Predictions 🔮"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}**")

        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            st.subheader("Predictions vs Actuals")
            st.dataframe(
                view[["date", "region", "actual_price", "prediction"]].reset_index(drop=True),
                use_container_width=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.0f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

            # ============================
            # Yearly Trend Chart
            # ============================
            if region == "All":
                yearly_data = disp_df[disp_df["year"] == year].copy()
                idx_all = yearly_data.index
                payload_all = fe_df.loc[idx_all].to_dict(orient="records")

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)

            else:
                yearly_data = disp_df[(disp_df["year"] == year) & (disp_df["region"] == region)].copy()
                idx_region = yearly_data.index
                payload_region = fe_df.loc[idx_region].to_dict(orient="records")

                resp_region = requests.post(API_URL, json=payload_region, timeout=60)
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_region, index=yearly_data.index).astype(float)

            # Aggregate by month
            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            # Highlight selected month
            monthly_avg["highlight"] = monthly_avg["month"].apply(lambda m: "Selected" if m == month else "Other")

            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=f"Yearly Trend — {year}{'' if region=='All' else f' — {region}'}"
            )

            # Add highlight with background shading
            highlight_month = month
            fig.add_vrect(
                x0=highlight_month - 0.5,
                x1=highlight_month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)

else:
    st.info("Choose filters and click **Show Predictions** to compute.")

