import os
import io
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Optional

# -----------------------------
# Page setup (wider layout)
# -----------------------------
st.set_page_config(page_title="🍽️ Food Recommendation System", layout="wide")
st.title("🍽️ Food Recommendation System by Nutritional Density")

# Backward-compatible cache (works on older Streamlit too)
try:
    cache_data = st.cache_data
except AttributeError:
    cache_data = st.cache

# -----------------------------
# Data loading (UPLOAD ONLY)
# -----------------------------
@cache_data(show_spinner=True)
def can_read_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False

PARQUET_OK = can_read_parquet()

@cache_data(show_spinner=True)
def load_data(uploaded_file) -> pd.DataFrame:
    """Load ONLY from the uploaded CSV/Parquet file."""
    if uploaded_file is None:
        raise ValueError("Please upload a CSV or Parquet file.")
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    bio = io.BytesIO(data)
    if name.endswith(".parquet"):
        if not PARQUET_OK:
            raise RuntimeError("Parquet file uploaded but no parquet engine (pyarrow/fastparquet) is available.")
        return pd.read_parquet(bio)
    if name.endswith(".csv"):
        return pd.read_csv(bio, low_memory=False)
    raise ValueError("Unsupported file type. Please upload a .csv or .parquet file.")

# Sidebar uploader
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])

# Stop the app until a file is uploaded
if uploaded is None:
    st.info("⬅️ Please upload a CSV or Parquet file to begin.")
    st.stop()

df = load_data(uploaded)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filter Options")

# Ensure expected dtypes are clean (keep text columns as strings)
for col in ("brand_name", "branded_food_category", "description"):
    if col in df.columns:
        # Use pandas StringDtype to avoid mixed types
        try:
            df[col] = df[col].astype("string")
        except Exception:
            df[col] = df[col].astype(str)

# Category list
if "branded_food_category" not in df.columns:
    st.error("Column 'branded_food_category' not found in data.")
    st.stop()

categories = df["branded_food_category"].dropna()
categories_sorted = sorted(categories.unique().tolist())

# Default = ONE category (most frequent), not "All"
default_category: Optional[str] = None
if len(categories_sorted) > 0:
    mode_series = df["branded_food_category"].mode(dropna=True)
    default_category = mode_series.iloc[0] if not mode_series.empty else categories_sorted[0]

selected_category = st.sidebar.selectbox(
    "Select food category",
    options=["All"] + categories_sorted,
    index=(["All"] + categories_sorted).index(default_category) if default_category in categories_sorted else 0
)

df_filtered = df if selected_category == "All" else df[df["branded_food_category"] == selected_category]

# -----------------------------
# Axis selectors
# -----------------------------
numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.warning("No numeric columns found in the filtered data.")
    st.stop()

x_axis = st.sidebar.selectbox(
    "Select X-axis", numeric_cols,
    index=numeric_cols.index("protein") if "protein" in numeric_cols else 0
)
y_axis = st.sidebar.selectbox(
    "Select Y-axis", numeric_cols,
    index=numeric_cols.index("saturated_fat") if "saturated_fat" in numeric_cols and len(numeric_cols) > 1 else min(1, len(numeric_cols)-1)
)

# -----------------------------
# Prep data for plotting
# -----------------------------
plot_df = df_filtered.copy()

# Avoid Plotly "NAType is not JSON serializable" issues:
# 1) Drop rows where x/y are NA
plot_df = plot_df.dropna(subset=[x_axis, y_axis])

# 2) Make sure hover fields are strings with no <NA>
hover_fields = [c for c in ["description", "branded_food_category"] if c in plot_df.columns]
for c in hover_fields:
    # Convert to pandas StringDtype, then fill NAs with "N/A"
    try:
        plot_df[c] = plot_df[c].astype("string").fillna("N/A")
    except Exception:
        plot_df[c] = plot_df[c].astype(str).fillna("N/A")

# Fallback for color if 'description' is missing
color_col = "description" if "description" in plot_df.columns else None

# -----------------------------
# Scatter Plot (wide)
# -----------------------------
fig = px.scatter(
    plot_df,
    x=x_axis,
    y=y_axis,
    color=color_col,  # keep your original choice if present; otherwise None (no legend explosion)
    hover_data=hover_fields,
    title=f"{y_axis} vs {x_axis} ({selected_category})"
)

# Make it feel wider/taller
fig.update_layout(
    height=600,              # taller for readability
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)
