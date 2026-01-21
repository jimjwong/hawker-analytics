import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.cluster import KMeans

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="NEA Hawker Tender Analytics",
    layout="wide"
)

# --------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data(path: str = "nea_bids_cleaned.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["MONTH_START"] = pd.to_datetime(df["MONTH_START"])
    return df

df = load_data()


st.title("NEA Hawker Tender Analytics & Bid Strategy Dashboard")

st.markdown(
"""
This dashboard is designed for **bidders who are considering hawker stall tenders in Singapore**.

**Objectives:**
1. Let you explore *historical bidding data* (winning & losing bids).
2. Help you *estimate a fair bid price* using ML models based on similar past tenders.
"""
)

# --------------------------------------------------
# 2. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Global Filters")

all_months = sorted(df["MONTH_START"].unique())
min_month, max_month = min(all_months), max(all_months)

# Convert pandas Timestamp to Python datetime for Streamlit compatibility
min_month_dt = min_month.to_pydatetime() if hasattr(min_month, 'to_pydatetime') else min_month
max_month_dt = max_month.to_pydatetime() if hasattr(max_month, 'to_pydatetime') else max_month

month_range = st.sidebar.slider(
    "Month range",
    min_value=min_month_dt,
    max_value=max_month_dt,
    value=(min_month_dt, max_month_dt),
    format="YYYY-MM"
)

centre_options = sorted(df["HAWKER_CENTRE"].unique())
trade_options = sorted(df["TRADE"].unique())

selected_centres = st.sidebar.multiselect("Hawker centre", centre_options, default=[])
selected_trades = st.sidebar.multiselect("Trade", trade_options, default=[])

# Apply filters
mask = (df["MONTH_START"] >= month_range[0]) & (df["MONTH_START"] <= month_range[1])

if len(selected_centres) > 0:
    mask &= df["HAWKER_CENTRE"].isin(selected_centres)

if len(selected_trades) > 0:
    mask &= df["TRADE"].isin(selected_trades)

df_filt = df[mask].copy()

st.sidebar.write(f"Filtered rows: {len(df_filt):,}")

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

main_tab1, main_tab2 = st.tabs(["Overview", "Table"])

with main_tab1:
    # --------------------------------------------------
    # 3. OVERVIEW METRICS
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 4. VISUALISATIONS – MARKET DYNAMICS
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 5. TRADE INSIGHTS (Competitiveness & Prices)
    # --------------------------------------------------

    st.subheader("Trade Insights (What Type of Stall to Bid For?)")

    tab_t1, tab_t2 = st.tabs(["Bid Levels by Trade", "Competition by Trade"])

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

    st.markdown("---")

# --------------------------------------------------
# 6. TRAIN ML MODELS (Reg + Class) – REUSED LATER
# --------------------------------------------------

@st.cache_resource
def train_models(df_all: pd.DataFrame):
    # REGRESSION: predict bid value
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

    # CLASSIFICATION: winner vs non-winner
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

# --------------------------------------------------
# 7. ML INSIGHTS + BID STRATEGY SIMULATOR
# --------------------------------------------------

with main_tab1:
    st.subheader("ML: Price Prediction & Win Probability (For Upcoming Bids)")

    tab_reg, tab_clf, tab_sim = st.tabs(
        ["Expected Winning Price", "Single Bid Win Probability", "Bid Strategy Simulator"]
    )

    # --- 7A. REGRESSION: Expected Winning Price ---
    with tab_reg:
        st.markdown("### Model Performance (Regression)")
        st.write(f"**MAE:** {reg_metrics['MAE']:.2f}")
        st.write(f"**RMSE:** {reg_metrics['RMSE']:.2f}")
        st.write(f"**R²:** {reg_metrics['R2']:.3f}")

        st.markdown("### Estimate Expected Winning Price for an Upcoming Tender")

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
            default_mean = float(df[df["HAWKER_CENTRE"] == centre_in]["STALL_MEAN_BID"].median()
                                 if (df["HAWKER_CENTRE"] == centre_in).any()
                                 else df["STALL_MEAN_BID"].median())
            stall_mean_in = st.number_input(
                "Estimated mean bid (based on past similar stalls)",
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
            st.caption("Use this as a benchmark. You may want to bid above this if the stall is very desirable.")

    # --- 7B. CLASSIFICATION: Single Bid Probability ---
    with tab_clf:
        st.markdown("### Model Performance (Classification)")
        st.write(f"**Winner Precision:** {winner_metrics.get('precision', float('nan')):.3f}")
        st.write(f"**Winner Recall:** {winner_metrics.get('recall', float('nan')):.3f}")
        st.write(f"**Winner F1-Score:** {winner_metrics.get('f1-score', float('nan')):.3f}")

        st.markdown("### Check Your Win Probability for a Single Bid")

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
            default_mean_c = float(df[df["HAWKER_CENTRE"] == centre_c]["STALL_MEAN_BID"].median()
                                   if (df["HAWKER_CENTRE"] == centre_c).any()
                                   else df["STALL_MEAN_BID"].median())
            stall_mean_c = st.number_input(
                "Estimated mean bid (from history)",
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

    # --- 7C. STRATEGY SIMULATOR: Bid vs Probability Curve ---
    with tab_sim:
        st.markdown(
            """
            ### Bid Strategy Simulator  
            See how your **win probability** changes as you increase or decrease your bid.
            """
        )

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

            base_mean = float(df[df["HAWKER_CENTRE"] == centre_s]["STALL_MEAN_BID"].median()
                              if (df["HAWKER_CENTRE"] == centre_s).any()
                              else df["STALL_MEAN_BID"].median())

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
                "Estimated mean bid (history-based)",
                min_value=0.0,
                value=base_mean,
                key="sim_mean"
            )

            if bid_max <= bid_min:
                st.warning("Max bid must be greater than min bid to run simulation.")
            else:
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

                # Recommended bids for target probabilities
                targets = [0.5, 0.7, 0.9]
                st.markdown("#### Suggested Bid Levels for Target Win Probabilities")
                rec_rows = []
                for t in targets:
                    possible = sim_result[sim_result["Win_Prob"] >= t]
                    if len(possible) > 0:
                        rec = possible.iloc[0]
                        rec_rows.append(
                            {"Target Probability": f"{int(t*100)}%",
                             "Suggested Bid": f"${rec['Bid']:,.0f}",
                             "Modelled Probability": f"{rec['Win_Prob']*100:,.1f}%"}
                        )
                if rec_rows:
                    st.table(pd.DataFrame(rec_rows))
                else:
                    st.info("Even at the max simulated bid, win probability never reached the targets.")

    st.markdown("---")

    # --------------------------------------------------
    # 8. CENTRE SEGMENTATION & SCORECARD
    # --------------------------------------------------

    st.subheader("Hawker Centre Segmentation & Scorecard")

    # Centre clustering on full dataset
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

    kmeans_features = ["AVG_WIN_BID", "MEDIAN_WIN_BID", "NUM_STALLS", "NUM_TRADES"]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    centre_profile["CLUSTER"] = kmeans.fit_predict(centre_profile[kmeans_features])

    cluster_labels = {
        0: "Cluster 0 – Lower to Mid Range",
        1: "Cluster 1 – Mid Market",
        2: "Cluster 2 – Premium / High Bids"
    }

    tab_seg, tab_score = st.tabs(["Segmentation View", "Centre Scorecard"])

    with tab_seg:
        st.write(
            "Each point represents a hawker centre. Clusters group centres with similar winning bid levels and complexity."
        )

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
            if cp is not None:
                st.metric("Cluster", clus_desc)
            else:
                st.metric("Cluster", "N/A")

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
        display_df = df_c.sort_values(["MONTH_START", "STALL_NO", "BID_RANK"])[[
            "MONTH_YEAR", "STALL_NO", "TRADE",
            "TENDERER", "BID_VALUE", "BID_RANK",
            "NUM_BIDDERS", "SUCCESS"
        ]]
        st.dataframe(display_df, use_container_width=True)

with main_tab2:
    # --------------------------------------------------
    # TABLE VIEWS
    # --------------------------------------------------
    
    st.subheader("Bidding Information Tables")
    
    table_tab1, table_tab2, table_tab3 = st.tabs(["All Bids", "Winning Bids", "Summary Statistics"])
    
    with table_tab1:
        st.markdown("### All Bids (Filtered)")
        display_cols = [
            "MONTH_YEAR", "HAWKER_CENTRE", "STALL_NO", "TRADE",
            "TENDERER", "BID_VALUE", "BID_RANK",
            "NUM_BIDDERS", "SUCCESS", "IS_WINNER"
        ]
        # Filter to only show columns that exist
        available_cols = [col for col in display_cols if col in df_filt.columns]
        all_bids_df = df_filt[available_cols].sort_values(
            ["MONTH_START", "HAWKER_CENTRE", "STALL_NO", "BID_RANK"]
        )
        st.dataframe(all_bids_df, use_container_width=True, height=600)
        
        st.download_button(
            "Download All Bids (CSV)",
            data=all_bids_df.to_csv(index=False).encode("utf-8"),
            file_name="all_bids_filtered.csv",
            mime="text/csv"
        )
    
    with table_tab2:
        st.markdown("### Winning Bids Only (Filtered)")
        winning_bids = df_filt[df_filt["IS_WINNER"] == 1].copy()
        if len(winning_bids) > 0:
            display_cols_win = [
                "MONTH_YEAR", "HAWKER_CENTRE", "STALL_NO", "TRADE",
                "TENDERER", "BID_VALUE", "NUM_BIDDERS", "SUCCESS"
            ]
            available_cols_win = [col for col in display_cols_win if col in winning_bids.columns]
            winning_df = winning_bids[available_cols_win].sort_values(
                ["MONTH_START", "HAWKER_CENTRE", "STALL_NO"]
            )
            st.dataframe(winning_df, use_container_width=True, height=600)
            
            st.download_button(
                "Download Winning Bids (CSV)",
                data=winning_df.to_csv(index=False).encode("utf-8"),
                file_name="winning_bids_filtered.csv",
                mime="text/csv"
            )
        else:
            st.info("No winning bids found for the current filter selection.")
    
    with table_tab3:
        st.markdown("### Summary Statistics by Hawker Centre")
        summary_stats = (
            df_filt.groupby("HAWKER_CENTRE")
            .agg(
                Total_Bids=("BID_VALUE", "count"),
                Winning_Bids=("IS_WINNER", "sum"),
                Avg_Bid=("BID_VALUE", "mean"),
                Median_Bid=("BID_VALUE", "median"),
                Min_Bid=("BID_VALUE", "min"),
                Max_Bid=("BID_VALUE", "max"),
                Avg_Num_Bidders=("NUM_BIDDERS", "mean"),
                Unique_Trades=("TRADE", "nunique"),
                Unique_Stalls=("STALL_NO", "nunique")
            )
            .reset_index()
            .round(2)
            .sort_values("Avg_Bid", ascending=False)
        )
        st.dataframe(summary_stats, use_container_width=True, height=600)
        
        st.markdown("### Summary Statistics by Trade")
        trade_stats = (
            df_filt.groupby("TRADE")
            .agg(
                Total_Bids=("BID_VALUE", "count"),
                Winning_Bids=("IS_WINNER", "sum"),
                Avg_Bid=("BID_VALUE", "mean"),
                Median_Bid=("BID_VALUE", "median"),
                Min_Bid=("BID_VALUE", "min"),
                Max_Bid=("BID_VALUE", "max"),
                Avg_Num_Bidders=("NUM_BIDDERS", "mean"),
                Unique_Centres=("HAWKER_CENTRE", "nunique")
            )
            .reset_index()
            .round(2)
            .sort_values("Avg_Bid", ascending=False)
        )
        st.dataframe(trade_stats, use_container_width=True, height=600)
        
        st.download_button(
            "Download Summary Statistics (CSV)",
            data=pd.concat([summary_stats, trade_stats], keys=["By_Centre", "By_Trade"]).to_csv(index=True).encode("utf-8"),
            file_name="summary_statistics.csv",
            mime="text/csv"
        )
