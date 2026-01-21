import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.cluster import KMeans

# ======================================================
# 0. STREAMLIT CONFIG – MUST BE FIRST STREAMLIT CALL
# ======================================================

st.set_page_config(
    page_title="NEA Hawker Tender Analytics",
    layout="wide"
)

# ======================================================
# 0a. APPEARANCE / THEME TOGGLE
# ======================================================
st.sidebar.markdown("### Appearance")
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.sidebar.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

if dark_mode:
    # Global dark styling and plot defaults
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #e8e8e8; }
        html, body, [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #e8e8e8; }
        h1, h2, h3, h4, h5, h6, p, li, label, span, div { color: #e8e8e8; }
        [data-testid="stMarkdown"] { color: #e8e8e8; }
        [data-testid="stSidebar"] { background-color: #0b0d11; color: #e8e8e8; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    px.defaults.template = "plotly_dark"
    plt.style.use("dark_background")
else:
    px.defaults.template = "plotly"
    plt.style.use("default")

# ======================================================
# 1. DATA LOADING & CLEANING (ONE DATASET)
# ======================================================

DATA_PATH = "./NEA Stalls Data/Provisional_Results_Table_inuse.xlsx"

def get_file_mtime(path: str) -> float:
    """Get file modification time for cache invalidation"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

@st.cache_data
def load_and_clean(path: str = DATA_PATH, file_mtime: float = 0.0) -> pd.DataFrame:
    df = pd.read_excel(path)
    
    # Map column names (handle case-insensitive and common variations)
    def find_column(df, possible_names):
        """Find column by trying multiple possible names (case-insensitive)"""
        for name in possible_names:
            # Exact match
            if name in df.columns:
                return name
            # Case-insensitive match
            for col in df.columns:
                if col.upper() == name.upper():
                    return col
        return None
    
    # Find columns with flexible matching
    month_year_col = find_column(df, ["MONTH_YEAR", "Month_Year", "month_year", "MONTH", "Month"])
    hawker_col = find_column(df, ["HAWKER_CENTRE", "Hawker_Centre", "hawker_centre", "HAWKER_CENTER", "Hawker Center"])
    stall_col = find_column(df, ["STALL_NO", "Stall_No", "stall_no", "STALL", "Stall"])
    trade_col = find_column(df, ["TRADE", "Trade", "trade"])
    tenderer_col = find_column(df, ["TENDERER", "Tenderer", "tenderer", "BIDDER", "Bidder"])
    success_col = find_column(df, ["SUCCESS", "Success", "success"])
    category_col = find_column(df, ["CATEGORY", "Category", "category"])
    bid_col = find_column(df, ["BID", "Bid", "bid", "BID_VALUE", "Bid_Value"])
    rank_col = find_column(df, ["BID_RANK", "Bid_Rank", "bid_rank", "RANK", "Rank"])
    
    # Check required columns
    required = {
        "MONTH_YEAR": month_year_col,
        "HAWKER_CENTRE": hawker_col,
        "STALL_NO": stall_col,
        "TRADE": trade_col,
        "TENDERER": tenderer_col,
        "BID": bid_col,
        "BID_RANK": rank_col
    }
    
    missing = [k for k, v in required.items() if v is None]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.info(f"Available columns: {', '.join(df.columns.tolist())}")
        st.stop()
    
    # Standardize column names
    df = df.rename(columns={
        month_year_col: "MONTH_YEAR",
        hawker_col: "HAWKER_CENTRE",
        stall_col: "STALL_NO",
        trade_col: "TRADE",
        tenderer_col: "TENDERER",
        bid_col: "BID",
        rank_col: "BID_RANK"
    })
    
    if success_col:
        df = df.rename(columns={success_col: "SUCCESS"})
    
    if category_col:
        df = df.rename(columns={category_col: "CATEGORY"})

    # Ensure strings are stripped
    for col in ["MONTH_YEAR", "HAWKER_CENTRE", "STALL_NO", "TRADE", "TENDERER"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    if "SUCCESS" in df.columns:
        df["SUCCESS"] = df["SUCCESS"].astype(str).str.strip()
    
    if "CATEGORY" in df.columns:
        df["CATEGORY"] = df["CATEGORY"].astype(str).str.strip()

    # Handle MONTH_YEAR: e.g. "2025-02/03" -> "2025-02"
    def normalize_month(m):
        m = str(m).strip()
        if "/" in m:
            m = m.split("/")[0]
        return m

    df["MONTH_YEAR_NORM"] = df["MONTH_YEAR"].apply(normalize_month)

    # Parse to datetime (1st of month)
    df["MONTH_START"] = pd.to_datetime(df["MONTH_YEAR_NORM"], format="%Y-%m", errors="coerce")
    df["YEAR"] = df["MONTH_START"].dt.year
    df["MONTH"] = df["MONTH_START"].dt.month

    # Use BID as numeric bid value
    df["BID_VALUE"] = pd.to_numeric(df["BID"], errors="coerce")

    # Coerce BID_RANK to int
    df["BID_RANK"] = pd.to_numeric(df["BID_RANK"], errors="coerce").astype("Int64")

    # Success flag + winner flag
    if "SUCCESS" in df.columns:
        df["SUCCESS_FLAG"] = df["SUCCESS"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "YES": 1, "NO": 0}).fillna(0).astype(int)
    else:
        df["SUCCESS_FLAG"] = 0
    
    df["IS_WINNER"] = (df["BID_RANK"] == 1).astype(int)

    # Stall-level grouping
    group_cols = ["MONTH_YEAR_NORM", "HAWKER_CENTRE", "STALL_NO"]

    df["NUM_BIDDERS"] = df.groupby(group_cols)["TENDERER"].transform("nunique")

    agg = df.groupby(group_cols)["BID_VALUE"].agg(
        STALL_MAX_BID="max",
        STALL_MIN_BID="min",
        STALL_MEAN_BID="mean"
    ).reset_index()

    df = df.merge(agg, on=group_cols, how="left")

    df["BID_TO_MAX"] = df["BID_VALUE"] / df["STALL_MAX_BID"]
    df["BID_TO_MEAN"] = df["BID_VALUE"] / df["STALL_MEAN_BID"]

    # Ensure MONTH_START is Python datetime for sliders
    df["MONTH_START"] = df["MONTH_START"].dt.to_pydatetime()

    return df

# Load data with file modification time for automatic cache invalidation
file_mtime = get_file_mtime(DATA_PATH)
df = load_and_clean(DATA_PATH, file_mtime)

# ======================================================
# 2. GLOBAL FILTERS
# ======================================================

st.title("NEA Hawker Tender Analytics & Bid Strategy (Single Dataset)")

# Add cache clear button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reload Data (Clear Cache)", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.rerun()

st.markdown(
"""
This dashboard uses **one consolidated dataset** (`Provisional_Results_Table_WithSuccess.xlsx`)
to help potential bidders:

- Explore **historical bidding patterns** (winning & losing bids)
- **Predict expected winning price** for upcoming tenders
- **Estimate win probability** for a chosen bid
"""
)

st.sidebar.header("Global Filters")

# Initialize session state for clear filters
if 'clear_filters' not in st.session_state:
    st.session_state.clear_filters = False

# Clear Filters button
if st.sidebar.button("🗑️ Clear All Filters", use_container_width=True, type="secondary"):
    st.session_state.clear_filters = True
    st.rerun()

all_months = sorted(df["MONTH_START"].dropna().unique())
if len(all_months) > 0:
    min_month, max_month = min(all_months), max(all_months)
else:
    min_month = max_month = pd.to_datetime("2024-01-01")

# Convert pandas Timestamp to Python datetime for Streamlit compatibility
min_month_dt = min_month.to_pydatetime() if hasattr(min_month, 'to_pydatetime') else min_month
max_month_dt = max_month.to_pydatetime() if hasattr(max_month, 'to_pydatetime') else max_month

# Reset month range if clear was clicked
if st.session_state.clear_filters:
    month_range_value = (min_month_dt, max_month_dt)
else:
    # Try to get from session state, otherwise use default
    if 'month_range' not in st.session_state:
        st.session_state.month_range = (min_month_dt, max_month_dt)
    month_range_value = st.session_state.month_range

month_range = st.sidebar.slider(
    "Month range",
    min_value=min_month_dt,
    max_value=max_month_dt,
    value=month_range_value,
    format="YYYY-MM"
)

# Update session state with current month range
st.session_state.month_range = month_range

# Initialize session state for filter selections
if 'selected_centres' not in st.session_state:
    st.session_state.selected_centres = []
if 'selected_trades' not in st.session_state:
    st.session_state.selected_trades = []
if 'selected_stalls' not in st.session_state:
    st.session_state.selected_stalls = []
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = []
if 'selected_success' not in st.session_state:
    st.session_state.selected_success = []

# Reset filter selections if clear was clicked
if st.session_state.clear_filters:
    st.session_state.selected_centres = []
    st.session_state.selected_trades = []
    st.session_state.selected_stalls = []
    st.session_state.selected_categories = []
    st.session_state.selected_success = []

# Start with month range filter
mask = (df["MONTH_START"] >= month_range[0]) & (df["MONTH_START"] <= month_range[1])
df_temp = df[mask].copy()

# Dynamic filter: Hawker Centre (based on month range)
centre_options = sorted(df_temp["HAWKER_CENTRE"].unique())
# Filter out invalid selections (in case options changed)
st.session_state.selected_centres = [c for c in st.session_state.selected_centres if c in centre_options]
selected_centres = st.sidebar.multiselect("Hawker centre", centre_options, default=st.session_state.selected_centres)
st.session_state.selected_centres = selected_centres

# Update mask and temp dataframe after centre selection
if len(selected_centres) > 0:
    mask &= df["HAWKER_CENTRE"].isin(selected_centres)
    df_temp = df[mask].copy()

# Dynamic filter: Category (based on month range + centre)
selected_categories = []
if "CATEGORY" in df_temp.columns:
    category_options = sorted(df_temp["CATEGORY"].dropna().unique())
    if len(category_options) > 0:
        st.session_state.selected_categories = [c for c in st.session_state.selected_categories if c in category_options]
        selected_categories = st.sidebar.multiselect("Category", category_options, default=st.session_state.selected_categories)
        st.session_state.selected_categories = selected_categories
        if len(selected_categories) > 0:
            mask &= df["CATEGORY"].isin(selected_categories)
            df_temp = df[mask].copy()

# Dynamic filter: Trade (based on month range + centre + category)
trade_options = sorted(df_temp["TRADE"].unique())
st.session_state.selected_trades = [t for t in st.session_state.selected_trades if t in trade_options]
selected_trades = st.sidebar.multiselect("Trade", trade_options, default=st.session_state.selected_trades)
st.session_state.selected_trades = selected_trades

# Update mask and temp dataframe after trade selection
if len(selected_trades) > 0:
    mask &= df["TRADE"].isin(selected_trades)
    df_temp = df[mask].copy()

# Dynamic filter: Stall No. (based on previous filters)
stall_options = sorted(df_temp["STALL_NO"].unique())
st.session_state.selected_stalls = [s for s in st.session_state.selected_stalls if s in stall_options]
selected_stalls = st.sidebar.multiselect("Stall No.", stall_options, default=st.session_state.selected_stalls)
st.session_state.selected_stalls = selected_stalls

# Update mask and temp dataframe after stall selection
if len(selected_stalls) > 0:
    mask &= df["STALL_NO"].isin(selected_stalls)
    df_temp = df[mask].copy()

# Dynamic filter: Success (based on previous filters)
selected_success = []
if "SUCCESS" in df_temp.columns:
    success_options = sorted(df_temp["SUCCESS"].dropna().unique())
    if len(success_options) > 0:
        st.session_state.selected_success = [s for s in st.session_state.selected_success if s in success_options]
        selected_success = st.sidebar.multiselect("Success", success_options, default=st.session_state.selected_success)
        st.session_state.selected_success = selected_success
        if len(selected_success) > 0:
            mask &= df["SUCCESS"].isin(selected_success)
            df_temp = df[mask].copy()

df_filt = df[mask].copy()

# Reset clear_filters flag after all filters have been processed
if st.session_state.clear_filters:
    st.session_state.clear_filters = False

st.sidebar.write(f"Filtered rows: {len(df_filt):,}")

# ======================================================
# 3. OVERVIEW METRICS
# ======================================================

st.subheader("Overview for Selected Filters")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Bids", f"{len(df_filt):,}")
with col2:
    st.metric("Unique Hawker Centres", df_filt["HAWKER_CENTRE"].nunique())
with col3:
    st.metric("Unique Trades", df_filt["TRADE"].nunique())
with col4:
    avg_win = df_filt[df_filt["IS_WINNER"] == 1]["BID_VALUE"].mean()
    st.metric("Avg Winning Bid (filtered)", f"{avg_win:,.2f}" if not np.isnan(avg_win) else "-")

st.markdown("---")

# ======================================================
# 3B. FILTERED BID DATA TABLE
# ======================================================

st.subheader("Filtered Bid Data")

# Select relevant columns for display
display_cols_table = [
    "MONTH_YEAR", "HAWKER_CENTRE", "STALL_NO", "TRADE",
    "TENDERER", "BID_VALUE", "BID_RANK",
    "NUM_BIDDERS"
]

# Add optional columns if they exist
if "SUCCESS" in df_filt.columns:
    display_cols_table.append("SUCCESS")

# Filter to only show columns that exist
available_cols_table = [col for col in display_cols_table if col in df_filt.columns]

if len(df_filt) > 0:
    # Sort by MONTH_START, then by other columns
    sort_cols = ["MONTH_START"]
    if "HAWKER_CENTRE" in df_filt.columns:
        sort_cols.append("HAWKER_CENTRE")
    if "STALL_NO" in df_filt.columns:
        sort_cols.append("STALL_NO")
    if "BID_RANK" in df_filt.columns:
        sort_cols.append("BID_RANK")
    
    # Get sort columns that exist
    sort_cols = [col for col in sort_cols if col in df_filt.columns]
    
    bid_data_table = df_filt.sort_values(sort_cols)[available_cols_table]
    
    st.dataframe(
        bid_data_table,
        use_container_width=True,
        height=400
    )
    
    # Download button
    st.download_button(
        "Download Filtered Bid Data (CSV)",
        data=bid_data_table.to_csv(index=False).encode("utf-8"),
        file_name=f"filtered_bid_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.info("No data available for the selected filters.")

st.markdown("---")

# ======================================================
# 4. MARKET DYNAMICS
# ======================================================

st.subheader("Market Dynamics")

tab1, tab2, tab3 = st.tabs(["Bid Distribution", "Winning Bids Over Time", "Top Centres"])

with tab1:
    st.markdown("**Distribution of All Bids (Winners & Non-Winners)**")
    fig = px.histogram(df_filt, x="BID_VALUE", nbins=40)
    fig.update_layout(xaxis_title="Bid Value", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("**Average Winning Bid Over Time**")
    top_bids = (
        df_filt[df_filt["IS_WINNER"] == 1]
        .groupby("MONTH_START")["BID_VALUE"]
        .mean()
        .reset_index()
    )
    if len(top_bids) > 0:
        fig = px.line(top_bids, x="MONTH_START", y="BID_VALUE")
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Winning Bid"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No winning bids in the current filter selection.")

with tab3:
    st.markdown("**Top 10 Hawker Centres by Average Winning Bid**")
    centre_top = (
        df_filt[df_filt["IS_WINNER"] == 1]
        .groupby("HAWKER_CENTRE")["BID_VALUE"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    if len(centre_top) > 0:
        fig = px.bar(
            centre_top.sort_values("BID_VALUE"),
            x="BID_VALUE",
            y="HAWKER_CENTRE",
            orientation="h"
        )
        fig.update_layout(
            xaxis_title="Average Winning Bid",
            yaxis_title="Hawker Centre"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for top centres in current filter.")

st.markdown("---")

# ======================================================
# 5. TRADE INSIGHTS
# ======================================================

st.subheader("Trade Insights (What Type of Stall to Bid For?)")

# Create tabs - include Category tabs only if CATEGORY column exists
if "CATEGORY" in df_filt.columns:
    tab_t1, tab_t2, tab_t3, tab_t4 = st.tabs([
        "Bid Levels by Trade", 
        "Competition by Trade",
        "Bid Levels by Category",
        "Competition by Category"
    ])
else:
    tab_t1, tab_t2 = st.tabs(["Bid Levels by Trade", "Competition by Trade"])
    tab_t3 = None
    tab_t4 = None

with tab_t1:
    df_trade_win = df_filt[df_filt["IS_WINNER"] == 1]
    if len(df_trade_win) > 0:
        trade_bids = (
            df_trade_win
            .groupby("TRADE")["BID_VALUE"]
            .median()
            .reset_index()
            .sort_values("BID_VALUE", ascending=False)
        )
        st.markdown("**Median Winning Bid by Trade**")
        fig = px.bar(
            trade_bids,
            x="BID_VALUE",
            y="TRADE",
            orientation="h",
        )
        fig.update_layout(
            xaxis_title="Median Winning Bid",
            yaxis_title="Trade"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No winning bids for current filters to compute trade statistics.")

with tab_t2:
    trade_comp = (
        df_filt
        .groupby("TRADE")["NUM_BIDDERS"]
        .mean()
        .reset_index()
        .sort_values("NUM_BIDDERS", ascending=False)
    )
    if len(trade_comp) > 0:
        st.markdown("**Average Number of Bidders per Trade**")
        fig = px.bar(
            trade_comp,
            x="NUM_BIDDERS",
            y="TRADE",
            orientation="h"
        )
        fig.update_layout(
            xaxis_title="Average Number of Bidders",
            yaxis_title="Trade"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to compute competition by trade.")

# Category tabs (only if CATEGORY column exists)
if tab_t3 is not None:
    with tab_t3:
        df_category_win = df_filt[df_filt["IS_WINNER"] == 1]
        if len(df_category_win) > 0 and "CATEGORY" in df_category_win.columns:
            category_bids = (
                df_category_win
                .groupby("CATEGORY")["BID_VALUE"]
                .median()
                .reset_index()
                .sort_values("BID_VALUE", ascending=False)
            )
            st.markdown("**Median Winning Bid by Category**")
            fig = px.bar(
                category_bids,
                x="BID_VALUE",
                y="CATEGORY",
                orientation="h",
            )
            fig.update_layout(
                xaxis_title="Median Winning Bid",
                yaxis_title="Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No winning bids for current filters to compute category statistics.")

if tab_t4 is not None:
    with tab_t4:
        if "CATEGORY" in df_filt.columns:
            category_comp = (
                df_filt
                .groupby("CATEGORY")["NUM_BIDDERS"]
                .mean()
                .reset_index()
                .sort_values("NUM_BIDDERS", ascending=False)
            )
            if len(category_comp) > 0:
                st.markdown("**Average Number of Bidders per Category**")
                fig = px.bar(
                    category_comp,
                    x="NUM_BIDDERS",
                    y="CATEGORY",
                    orientation="h"
                )
                fig.update_layout(
                    xaxis_title="Average Number of Bidders",
                    yaxis_title="Category"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to compute competition by category.")
        else:
            st.info("Category column not available in the dataset.")

# Statistics Table
st.markdown("### Statistics Table")

# Create tabs for Category and Trade
if "CATEGORY" in df_filt.columns:
    stat_tab1, stat_tab2 = st.tabs(["By Trade", "By Category"])
else:
    stat_tab1 = st.container()
    stat_tab2 = None

# Tab 1: Statistics by Trade
with stat_tab1:
    trade_stats_table = (
        df_filt.groupby("TRADE")
        .agg(
            Total_Bids=("BID_VALUE", "count"),
            Winning_Bids=("IS_WINNER", "sum"),
            Avg_Bid=("BID_VALUE", "mean"),
            Median_Bid=("BID_VALUE", "median"),
            Min_Bid=("BID_VALUE", "min"),
            Max_Bid=("BID_VALUE", "max"),
            Avg_Num_Bidders=("NUM_BIDDERS", "mean"),
            Unique_Centres=("HAWKER_CENTRE", "nunique"),
            Unique_Stalls=("STALL_NO", "nunique")
        )
        .reset_index()
        .round(2)
        .sort_values("Avg_Bid", ascending=False)
    )

    if len(trade_stats_table) > 0:
        st.dataframe(
            trade_stats_table,
            use_container_width=True,
            height=400
        )
    else:
        st.info("No data available for trade statistics table.")

# Tab 2: Statistics by Category (only if CATEGORY column exists)
if stat_tab2 is not None:
    with stat_tab2:
        if "CATEGORY" in df_filt.columns:
            category_stats_table = (
                df_filt.groupby("CATEGORY")
                .agg(
                    Total_Bids=("BID_VALUE", "count"),
                    Winning_Bids=("IS_WINNER", "sum"),
                    Avg_Bid=("BID_VALUE", "mean"),
                    Median_Bid=("BID_VALUE", "median"),
                    Min_Bid=("BID_VALUE", "min"),
                    Max_Bid=("BID_VALUE", "max"),
                    Avg_Num_Bidders=("NUM_BIDDERS", "mean"),
                    Unique_Centres=("HAWKER_CENTRE", "nunique"),
                    Unique_Stalls=("STALL_NO", "nunique")
                )
                .reset_index()
                .round(2)
                .sort_values("Avg_Bid", ascending=False)
            )

            if len(category_stats_table) > 0:
                st.dataframe(
                    category_stats_table,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No data available for category statistics table.")
        else:
            st.info("Category column not available in the dataset.")

st.markdown("---")

# ======================================================
# 6. TRAIN ML MODELS (REG + CLF)
# ======================================================

@st.cache_resource
def train_models(df_all: pd.DataFrame):
    # Regression: predict bid value
    features_reg = ["HAWKER_CENTRE", "TRADE", "YEAR", "MONTH",
                    "NUM_BIDDERS", "STALL_MEAN_BID"]
    df_reg = df_all.dropna(subset=["BID_VALUE"] + features_reg)
    X_reg = df_reg[features_reg]
    y_reg = df_reg["BID_VALUE"]

    cat_features = ["HAWKER_CENTRE", "TRADE"]
    num_features = ["YEAR", "MONTH", "NUM_BIDDERS", "STALL_MEAN_BID"]

    preprocess_reg = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    reg_model = Pipeline(steps=[
        ("preprocess", preprocess_reg),
        ("model", RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    reg_mae = mean_absolute_error(y_test, y_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    reg_r2 = r2_score(y_test, y_pred)

    reg_metrics = {"MAE": reg_mae, "RMSE": reg_rmse, "R2": reg_r2}

    # Classification: winner vs non-winner
    features_clf = ["HAWKER_CENTRE", "TRADE", "YEAR", "MONTH",
                    "NUM_BIDDERS", "BID_VALUE", "STALL_MEAN_BID"]
    df_clf = df_all.dropna(subset=features_clf + ["IS_WINNER"])
    X_clf = df_clf[features_clf]
    y_clf = df_clf["IS_WINNER"]

    cat_features_clf = ["HAWKER_CENTRE", "TRADE"]
    num_features_clf = ["YEAR", "MONTH", "NUM_BIDDERS",
                        "BID_VALUE", "STALL_MEAN_BID"]

    preprocess_clf = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features_clf),
            ("num", "passthrough", num_features_clf)
        ]
    )

    clf_model = Pipeline(steps=[
        ("preprocess", preprocess_clf),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    clf_model.fit(Xc_train, yc_train)
    yc_pred = clf_model.predict(Xc_test)

    clf_rep = classification_report(yc_test, yc_pred, output_dict=True)
    winner_metrics = clf_rep.get("1", {})

    return reg_model, reg_metrics, clf_model, winner_metrics

reg_model, reg_metrics, clf_model, winner_metrics = train_models(df)

# ======================================================
# 7. ML INSIGHTS & BID STRATEGY
# ======================================================

st.subheader("ML: Price Prediction & Win Probability (Upcoming Bids)")

tab_reg, tab_clf, tab_sim = st.tabs(
    ["Expected Winning Price", "Single Bid Win Probability", "Bid Strategy Simulator"]
)

# --- 7A. Regression ---
with tab_reg:
    st.markdown("### Regression Model Performance")
    st.write(f"**MAE:** {reg_metrics['MAE']:.2f}")
    st.write(f"**RMSE:** {reg_metrics['RMSE']:.2f}")
    st.write(f"**R²:** {reg_metrics['R2']:.3f}")

    st.markdown("### Estimate Expected Winning Price")

    c1, c2 = st.columns(2)
    with c1:
        centre_in = st.selectbox(
            "Hawker Centre",
            sorted(df["HAWKER_CENTRE"].unique()),
            key="reg_centre"
        )
        trade_in = st.selectbox(
            "Trade",
            sorted(df["TRADE"].unique()),
            key="reg_trade"
        )
        year_in = st.number_input(
            "Year of tender",
            min_value=2024,
            max_value=2035,
            value=int(df["YEAR"].max()),
            key="reg_year"
        )
        month_in = st.number_input(
            "Month of tender",
            min_value=1,
            max_value=12,
            value=int(df["MONTH"].mode()[0]),
            key="reg_month"
        )

    with c2:
        num_bidders_in = st.number_input(
            "Expected number of bidders",
            min_value=1,
            max_value=20,
            value=3,
            key="reg_bidders"
        )
        default_mean = float(
            df[df["HAWKER_CENTRE"] == centre_in]["STALL_MEAN_BID"].median()
            if (df["HAWKER_CENTRE"] == centre_in).any()
            else df["STALL_MEAN_BID"].median()
        )
        stall_mean_in = st.number_input(
            "Estimated mean bid (based on history)",
            min_value=0.0,
            value=default_mean,
            key="reg_mean"
        )

    input_df = pd.DataFrame([{
        "HAWKER_CENTRE": centre_in,
        "TRADE": trade_in,
        "YEAR": year_in,
        "MONTH": month_in,
        "NUM_BIDDERS": num_bidders_in,
        "STALL_MEAN_BID": stall_mean_in
    }])

    if st.button("Predict Expected Winning Price"):
        pred_bid = reg_model.predict(input_df)[0]
        st.success(f"Estimated Expected Winning Bid: **${pred_bid:,.2f}**")

# --- 7B. Classification ---
with tab_clf:
    st.markdown("### Classification Model Performance")
    st.write(f"**Winner Precision:** {winner_metrics.get('precision', float('nan')):.3f}")
    st.write(f"**Winner Recall:** {winner_metrics.get('recall', float('nan')):.3f}")
    st.write(f"**Winner F1-Score:** {winner_metrics.get('f1-score', float('nan')):.3f}")

    st.markdown("### Check Win Probability for a Single Bid")

    c1, c2 = st.columns(2)
    with c1:
        centre_c = st.selectbox(
            "Hawker Centre",
            sorted(df["HAWKER_CENTRE"].unique()),
            key="clf_centre"
        )
        trade_c = st.selectbox(
            "Trade",
            sorted(df["TRADE"].unique()),
            key="clf_trade"
        )
        year_c = st.number_input(
            "Year of tender",
            min_value=2024,
            max_value=2035,
            value=int(df["YEAR"].max()),
            key="clf_year"
        )
        month_c = st.number_input(
            "Month of tender",
            min_value=1,
            max_value=12,
            value=int(df["MONTH"].mode()[0]),
            key="clf_month"
        )

    with c2:
        num_bidders_c = st.number_input(
            "Expected number of bidders",
            min_value=1,
            max_value=20,
            value=3,
            key="clf_bidders"
        )
        bid_val_c = st.number_input(
            "Your proposed bid value",
            min_value=0.0,
            value=float(df["BID_VALUE"].median()),
            key="clf_bid"
        )
        default_mean_c = float(
            df[df["HAWKER_CENTRE"] == centre_c]["STALL_MEAN_BID"].median()
            if (df["HAWKER_CENTRE"] == centre_c).any()
            else df["STALL_MEAN_BID"].median()
        )
        stall_mean_c = st.number_input(
            "Estimated mean bid (history)",
            min_value=0.0,
            value=default_mean_c,
            key="clf_mean"
        )

    input_clf_df = pd.DataFrame([{
        "HAWKER_CENTRE": centre_c,
        "TRADE": trade_c,
        "YEAR": year_c,
        "MONTH": month_c,
        "NUM_BIDDERS": num_bidders_c,
        "BID_VALUE": bid_val_c,
        "STALL_MEAN_BID": stall_mean_c
    }])

    if st.button("Predict Win Probability"):
        prob = clf_model.predict_proba(input_clf_df)[0][1]
        st.success(f"Estimated probability of winning: **{prob*100:.1f}%**")

# --- 7C. Bid Strategy Simulator ---
with tab_sim:
    st.markdown("### Bid Strategy Simulator – Bid vs Win Probability")

    c1, c2 = st.columns(2)
    with c1:
        centre_s = st.selectbox(
            "Hawker Centre",
            sorted(df["HAWKER_CENTRE"].unique()),
            key="sim_centre"
        )
        trade_s = st.selectbox(
            "Trade",
            sorted(df["TRADE"].unique()),
            key="sim_trade"
        )
        year_s = st.number_input(
            "Year of tender",
            min_value=2024,
            max_value=2035,
            value=int(df["YEAR"].max()),
            key="sim_year"
        )
        month_s = st.number_input(
            "Month of tender",
            min_value=1,
            max_value=12,
            value=int(df["MONTH"].mode()[0]),
            key="sim_month"
        )

    with c2:
        num_bidders_s = st.number_input(
            "Expected number of bidders",
            min_value=1,
            max_value=20,
            value=3,
            key="sim_bidders"
        )
        base_mean = float(
            df[df["HAWKER_CENTRE"] == centre_s]["STALL_MEAN_BID"].median()
            if (df["HAWKER_CENTRE"] == centre_s).any()
            else df["STALL_MEAN_BID"].median()
        )
        bid_min = st.number_input(
            "Min bid to simulate",
            min_value=0.0,
            value=max(0.0, base_mean * 0.5),
            key="sim_min"
        )
        bid_max = st.number_input(
            "Max bid to simulate",
            min_value=0.0,
            value=base_mean * 1.8,
            key="sim_max"
        )
        stall_mean_s = st.number_input(
            "Estimated mean bid (history)",
            min_value=0.0,
            value=base_mean,
            key="sim_mean"
        )

    if bid_max > bid_min:
        bids = np.linspace(bid_min, bid_max, 40)
        sim_df = pd.DataFrame({
            "HAWKER_CENTRE": [centre_s] * len(bids),
            "TRADE": [trade_s] * len(bids),
            "YEAR": [year_s] * len(bids),
            "MONTH": [month_s] * len(bids),
            "NUM_BIDDERS": [num_bidders_s] * len(bids),
            "BID_VALUE": bids,
            "STALL_MEAN_BID": [stall_mean_s] * len(bids)
        })
        probs = clf_model.predict_proba(sim_df)[:, 1]
        sim_result = pd.DataFrame({"Bid": bids, "Win_Prob": probs})

        fig = px.line(sim_result, x="Bid", y="Win_Prob")
        fig.update_layout(
            xaxis_title="Bid Value",
            yaxis_title="Probability of Winning",
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)

        targets = [0.5, 0.7, 0.9]
        st.markdown("#### Suggested Bid Levels for Target Win Probabilities")
        rows = []
        for t in targets:
            possible = sim_result[sim_result["Win_Prob"] >= t]
            if len(possible) > 0:
                rec = possible.iloc[0]
                rows.append({
                    "Target Probability": f"{int(t*100)}%",
                    "Suggested Bid": f"${rec['Bid']:,.0f}",
                    "Modelled Probability": f"{rec['Win_Prob']*100:,.1f}%"
                })
        if rows:
            st.table(pd.DataFrame(rows))
        else:
            st.info("Even at the max simulated bid, target probabilities were not reached.")
    else:
        st.warning("Max bid must be greater than min bid to run simulation.")

st.markdown("---")

# ======================================================
# 8. CENTRE SEGMENTATION & SCORECARD
# ======================================================

st.subheader("Hawker Centre Segmentation & Scorecard")

centre_profile = (
    df[df["IS_WINNER"] == 1]
    .groupby("HAWKER_CENTRE")
    .agg(
        AVG_WIN_BID=("BID_VALUE", "mean"),
        MEDIAN_WIN_BID=("BID_VALUE", "median"),
        NUM_STALLS=("STALL_NO", "nunique"),
        NUM_TRADES=("TRADE", "nunique"),
        AVG_NUM_BIDDERS=("NUM_BIDDERS", "mean")
    )
    .reset_index()
)

if len(centre_profile) > 0:
    kmeans_features = ["AVG_WIN_BID", "MEDIAN_WIN_BID", "NUM_STALLS", "NUM_TRADES"]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    centre_profile["CLUSTER"] = kmeans.fit_predict(centre_profile[kmeans_features])
else:
    centre_profile["CLUSTER"] = np.nan

cluster_labels = {
    0: "Cluster 0 – Lower to Mid Range",
    1: "Cluster 1 – Mid Market",
    2: "Cluster 2 – Premium / High Bids"
}

tab_seg, tab_score = st.tabs(["Segmentation View", "Centre Scorecard"])

with tab_seg:
    if len(centre_profile) > 0:
        fig = px.scatter(
            centre_profile,
            x="AVG_WIN_BID",
            y="MEDIAN_WIN_BID",
            size="NUM_STALLS",
            color="CLUSTER",
            hover_data=["HAWKER_CENTRE", "NUM_TRADES", "AVG_NUM_BIDDERS"],
            labels={
                "AVG_WIN_BID": "Average Winning Bid",
                "MEDIAN_WIN_BID": "Median Winning Bid",
                "NUM_STALLS": "Number of Stalls"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download centre profile with clusters (CSV)",
            data=centre_profile.to_csv(index=False).encode("utf-8"),
            file_name="centre_profile_clusters.csv",
            mime="text/csv"
        )
    else:
        st.info("Not enough winning-bid data to perform clustering.")

with tab_score:
    centre_choice = st.selectbox(
        "Select a centre for detailed scorecard",
        sorted(df["HAWKER_CENTRE"].unique()),
        key="score_centre"
    )

    df_c = df[df["HAWKER_CENTRE"] == centre_choice]
    win_c = df_c[df_c["IS_WINNER"] == 1]
    c_prof = centre_profile[centre_profile["HAWKER_CENTRE"] == centre_choice]

    if len(c_prof) > 0:
        cp = c_prof.iloc[0]
        clus = int(cp["CLUSTER"])
        clus_desc = cluster_labels.get(clus, f"Cluster {clus}")
    else:
        cp = None
        clus_desc = "Not enough data to cluster this centre."

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        med_win = win_c["BID_VALUE"].median()
        st.metric("Median Winning Bid", f"${med_win:,.0f}" if not np.isnan(med_win) else "-")
    with col_b:
        avg_comp = df_c["NUM_BIDDERS"].mean()
        st.metric("Average # Bidders", f"{avg_comp:,.1f}" if not np.isnan(avg_comp) else "-")
    with col_c:
        st.metric("Cluster", clus_desc)

    st.markdown("#### Historical Winning Bids at This Centre")
    if len(win_c) > 0:
        wb = (
            win_c.groupby("MONTH_START")["BID_VALUE"]
            .mean()
            .reset_index()
            .sort_values("MONTH_START")
        )
        fig = px.line(wb, x="MONTH_START", y="BID_VALUE")
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Winning Bid"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No winning bids recorded for this centre yet.")

    st.markdown("#### Raw Bid History (Filtered by Centre)")
    # Sort by MONTH_START first, then select columns for display
    display_cols = ["MONTH_YEAR", "STALL_NO", "TRADE", "TENDERER", "BID_VALUE", "BID_RANK", "NUM_BIDDERS"]
    if "SUCCESS" in df_c.columns:
        display_cols.append("SUCCESS")
    display_df = df_c.sort_values(["MONTH_START", "STALL_NO", "BID_RANK"])[display_cols]
    st.dataframe(display_df, use_container_width=True)
