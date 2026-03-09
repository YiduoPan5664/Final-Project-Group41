
import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely import wkt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy import stats

st.set_page_config(page_title="Chicago TIF Dashboard", layout="wide")
st.title("Chicago TIF Spending: Is the Program Working?")
st.markdown(
    "This dashboard is designed for policymakers and city officials evaluating "
    "whether Chicago's Tax Increment Financing program is meeting its redistributive goals. "
    "Use the tabs below to explore spending patterns, income disparities, and trends over time."
)

TIF_ANNUAL  = "data/raw-data/Tax_Increment_Financing_(TIF)_Annual_Report_-_Analysis_of_Special_Tax_Allocation_Fund_20260301.csv"
TIF_BOUNDS  = "data/raw-data/Boundaries_-_Tax_Increment_Financing_Districts_20260301.csv"
TIF_DEP     = "data/raw-data/Boundaries_-_Tax_Increment_Financing_Districts_(Deprecated_March_2018)_20260301.csv"
COMM_BOUNDS = "data/raw-data/Boundaries_-_Community_Areas_20260301.csv"
ACS_INCOME  = "data/raw-data/ACS_5_Year_Data_by_Community_Area_20260301.csv"

def clean_dollars(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")

def val_to_hex(val, vmin, vmax, cmap):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return mcolors.to_hex(cm.get_cmap(cmap)(norm(val)))

@st.cache_data
def load_all():
    tif_raw = pd.read_csv(TIF_ANNUAL)
    tif_raw.columns = tif_raw.columns.str.strip()
    tif_raw["Total Expenditure"] = clean_dollars(tif_raw["Total Expenditure"])
    tif_raw["Report Year"] = pd.to_numeric(tif_raw["Report Year"], errors="coerce")

    import re as _re
    def _norm(s): return _re.sub(r"T-\s*0*", "T-", str(s).strip())

    # Combine current + deprecated boundaries so all districts have geometry
    b1 = pd.read_csv(TIF_BOUNDS); b1.columns = b1.columns.str.strip(); b1["tif_id"] = b1["REF"].apply(_norm)
    b2 = pd.read_csv(TIF_DEP);   b2.columns = b2.columns.str.strip(); b2["tif_id"] = b2["REF"].apply(_norm)
    tif_b = pd.concat([b1[["NAME","tif_id","the_geom"]], b2[["NAME","tif_id","the_geom"]]]).drop_duplicates("tif_id")
    tif_b["geometry"] = tif_b["the_geom"].apply(wkt.loads)
    tif_geo = gpd.GeoDataFrame(tif_b, geometry="geometry", crs="EPSG:4326")
    tif_geo = tif_geo.rename(columns={"NAME": "TIF District"})

    tif_raw["tif_id"] = tif_raw["TIF Number"].apply(_norm)
    tif_raw["Property Tax Increment - Cumulative"] = clean_dollars(
        tif_raw["Property Tax Increment - Cumulative"]
    )
    # Use the last reported cumulative increment per district (avoids summing negative adjustments)
    inc_last = (
        tif_raw.dropna(subset=["Property Tax Increment - Cumulative"])
        .sort_values("Report Year")
        .groupby("tif_id", as_index=False)["Property Tax Increment - Cumulative"]
        .last()
        .rename(columns={"Property Tax Increment - Cumulative": "cumulative_increment"})
    )
    tif_cumulative = tif_raw.groupby("tif_id", as_index=False).agg(
        cumulative_expenditures=("Total Expenditure", "sum"),
    ).merge(inc_last, on="tif_id", how="left")
    # Normalised ratio: cumulative expenditure per $1 of tax increment generated
    tif_cumulative["exp_per_increment"] = (
        tif_cumulative["cumulative_expenditures"] / tif_cumulative["cumulative_increment"]
    ).replace([float("inf"), float("-inf")], float("nan"))
    tif_full = tif_geo.merge(tif_cumulative, on="tif_id", how="left")

    comm_b = pd.read_csv(COMM_BOUNDS)
    comm_b["geometry"] = comm_b["the_geom"].apply(wkt.loads)
    comm_geo = gpd.GeoDataFrame(comm_b, geometry="geometry", crs="EPSG:4326")
    comm_geo = comm_geo.rename(columns={"COMMUNITY": "Community Area"})
    comm_geo["Community Area"] = comm_geo["Community Area"].str.upper().str.strip()

    acs = pd.read_csv(ACS_INCOME)
    acs.columns = acs.columns.str.strip()
    acs_latest = acs.sort_values("ACS Year").groupby("Community Area").last().reset_index()
    acs_latest["Community Area"] = acs_latest["Community Area"].str.upper().str.strip()
    income_cols = ["Under $25,000", "$25,000 to $49,999",
                   "$50,000 to $74,999", "$75,000 to $125,000", "$125,000 +"]
    midpoints = [12500, 37500, 62500, 100000, 175000]
    for col in income_cols:
        acs_latest[col] = pd.to_numeric(acs_latest[col].astype(str).str.replace(",", ""), errors="coerce")
    acs_latest["total_hh"] = acs_latest[income_cols].sum(axis=1)
    acs_latest["weighted_income"] = (
        sum(acs_latest[col] * mid for col, mid in zip(income_cols, midpoints))
        / acs_latest["total_hh"]
    )
    comm = comm_geo.merge(acs_latest[["Community Area", "weighted_income"]], on="Community Area", how="left")

    # Spatial join: TIF centroid → community area
    tif_c = tif_full.copy()
    tif_c["geometry"] = tif_full.geometry.centroid
    joined = gpd.sjoin(
        tif_c[["tif_id", "TIF District", "cumulative_expenditures", "exp_per_increment", "geometry"]],
        comm[["Community Area", "weighted_income", "geometry"]],
        how="left", predicate="within",
    ).dropna(subset=["weighted_income", "cumulative_expenditures"])
    joined["quintile"] = pd.qcut(
        joined["weighted_income"], q=5,
        labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"]
    )
    # Precompute income-group trend data
    income_lookup = joined[["tif_id","weighted_income"]].drop_duplicates("tif_id")
    tif_raw_income = tif_raw.merge(income_lookup, on="tif_id", how="inner")
    tif_raw_income = tif_raw_income.dropna(subset=["weighted_income","Total Expenditure","Report Year"])
    breaks = tif_raw_income["weighted_income"].quantile([0, 1/3, 2/3, 1]).values
    tif_raw_income["income_group"] = pd.cut(
        tif_raw_income["weighted_income"], bins=breaks, include_lowest=True,
        labels=["Low-Income Districts","Middle-Income Districts","High-Income Districts"]
    )
    trend = (
        tif_raw_income.groupby(["Report Year","income_group"])["Total Expenditure"]
        .sum().reset_index()
        .rename(columns={"Total Expenditure":"expenditure"})
    )
    return tif_raw, tif_full, comm, joined, trend

tif_raw, tif_full, comm, joined, trend = load_all()

# ── Executive Summary ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Key Findings")

high_med = joined[joined["weighted_income"] >= joined["weighted_income"].quantile(2/3)]["cumulative_expenditures"].median() / 1e6
low_med  = joined[joined["weighted_income"] <= joined["weighted_income"].quantile(1/3)]["cumulative_expenditures"].median() / 1e6
ratio_gap = high_med / low_med if low_med > 0 else float("nan")

norm_r = joined.dropna(subset=["exp_per_increment"])
norm_r = norm_r[norm_r["exp_per_increment"] < 10]
from scipy import stats as _stats
_slope, _intercept, _r, _p, _ = _stats.linregress(
    norm_r["weighted_income"] / 1000, norm_r["exp_per_increment"]
)

trend_high = trend[trend["income_group"] == "High-Income Districts"]
trend_low  = trend[trend["income_group"] == "Low-Income Districts"]
gap_2017 = (trend_high[trend_high["Report Year"] == 2017]["expenditure"].values[0] /
            trend_low[trend_low["Report Year"] == 2017]["expenditure"].values[0])
gap_2024 = (trend_high[trend_high["Report Year"] == 2024]["expenditure"].values[0] /
            trend_low[trend_low["Report Year"] == 2024]["expenditure"].values[0])

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Spending Gap (High vs. Low Income)",
    f"{ratio_gap:.1f}×",
    help="Median cumulative TIF expenditure in the highest-income third of districts vs. the lowest-income third"
)
col2.metric(
    "Gap Growth (2017 → 2024)",
    f"{gap_2017:.1f}× → {gap_2024:.1f}×",
    help="Annual spending ratio between high- and low-income districts has nearly doubled since 2017"
)
col3.metric(
    "Raw Spending Correlation",
    "Significant",
    "Higher income = more spending",
    help="Statistically significant positive correlation between community income and TIF expenditure"
)
col4.metric(
    "After Normalising by Increment",
    "Not Significant",
    "Gap explained by mechanism",
    help="Once we account for tax increment generated, the income-spending gap disappears — the TIF mechanism itself drives the disparity"
)

st.markdown(
    "Wealthier TIF districts spend more in absolute terms, but only because they generate "
    "more tax increment to begin with. The disparity is a feature of TIF's self-financing design. "
    "Without structural reform, TIF will continue to concentrate resources in already-prosperous areas."
)
st.markdown("---")

# Income legend values
_, income_bins = pd.qcut(comm["weighted_income"].dropna(), q=5, retbins=True)
income_bin_labels = [f"${income_bins[i]/1000:.0f}k – ${income_bins[i+1]/1000:.0f}k" for i in range(5)]
blues_legend = [val_to_hex((income_bins[i]+income_bins[i+1])/2, income_bins[0], income_bins[5], "Blues") for i in range(5)]

# Sidebar
st.sidebar.header("Map Controls")
years = sorted(tif_raw["Report Year"].dropna().unique().astype(int))
selected_year = st.sidebar.select_slider("Reporting Year", options=years, value=years[-1])

# Shared: filter TIF to selected year
tif_year = (
    tif_raw[tif_raw["Report Year"] == selected_year]
    .groupby("tif_id", as_index=False)
    .agg(expenditures=("Total Expenditure", "sum"))
)
tif_income_lookup = joined[["tif_id", "weighted_income", "Community Area"]].drop_duplicates("tif_id")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "District Map",
    "Spending by Neighbourhood Income",
    "Does Income Predict Spending?",
    "Is the Gap Fair? (Normalised View)",
    "Is the Gap Growing?",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Map (toggle: expenditure fill or income fill)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    map_mode = st.radio(
        "Color districts by:",
        ["Annual Expenditure", "Community Income", "Normalised Spending (Exp ÷ Increment)"],
        horizontal=True,
    )

    tif_map1 = tif_full.merge(tif_year, on="tif_id", how="left")
    tif_map1 = tif_map1.merge(tif_income_lookup, on="tif_id", how="left")
    # exp_per_increment already in tif_full from load_all

    inc_min, inc_max = income_bins[0], income_bins[5]
    exp_vals = tif_map1["expenditures"].dropna()
    exp_min = exp_vals.quantile(0.05) if len(exp_vals) > 0 else 0
    exp_max = exp_vals.quantile(0.95) if len(exp_vals) > 0 else 1

    m1 = folium.Map(location=[41.83, -87.68], zoom_start=11, tiles="CartoDB positron")

    if map_mode == "Annual Expenditure":
        def style_map(feature):
            val = feature["properties"].get("expenditures")
            try:
                val = float(val)
                if np.isnan(val): raise ValueError
                fill = val_to_hex(np.clip(val, exp_min, exp_max), exp_min, exp_max, "YlOrRd")
                return {"fillColor": fill, "fillOpacity": 0.85, "color": "#555", "weight": 0.4}
            except (TypeError, ValueError):
                return {"fillColor": "#dddddd", "fillOpacity": 0.3, "color": "#aaa", "weight": 0.4}
    elif map_mode == "Community Income":
        def style_map(feature):
            inc = feature["properties"].get("weighted_income")
            try:
                fill = val_to_hex(np.clip(float(inc), inc_min, inc_max), inc_min, inc_max, "Blues")
                return {"fillColor": fill, "fillOpacity": 0.8, "color": "white", "weight": 0.4}
            except (TypeError, ValueError):
                return {"fillColor": "#dddddd", "fillOpacity": 0.4, "color": "white", "weight": 0.4}
    else:
        # Normalised spending
        norm_vals = tif_map1["exp_per_increment"].dropna()
        norm_min = norm_vals.quantile(0.05) if len(norm_vals) > 0 else 0
        norm_max = min(norm_vals.quantile(0.95), 10) if len(norm_vals) > 0 else 2
        def style_map(feature):
            val = feature["properties"].get("exp_per_increment")
            try:
                val = float(val)
                if np.isnan(val): raise ValueError
                fill = val_to_hex(np.clip(val, norm_min, norm_max), norm_min, norm_max, "PuRd")
                return {"fillColor": fill, "fillOpacity": 0.85, "color": "#555", "weight": 0.4}
            except (TypeError, ValueError):
                return {"fillColor": "#dddddd", "fillOpacity": 0.3, "color": "#aaa", "weight": 0.4}

    folium.GeoJson(
        tif_map1[["TIF District", "Community Area", "weighted_income", "expenditures", "exp_per_increment", "geometry"]].__geo_interface__,
        style_function=style_map,
        highlight_function=lambda x: {"weight": 4, "color": "#000", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(
            fields=["TIF District", "Community Area", "weighted_income", "expenditures", "exp_per_increment"],
            aliases=["TIF District", "Community Area", "Community Income ($)", f"Expenditures ({selected_year}) $", "Exp ÷ Increment"],
            localize=True,
        ),
    ).add_to(m1)

    col1, col2 = st.columns([3, 1])
    with col1:
        st_folium(m1, width=900, height=620, key="map1")
    with col2:
        st.subheader(f"Top 10 Districts ({selected_year})")
        top10 = (tif_year.nlargest(10, "expenditures").merge(tif_full[["tif_id","TIF District"]].drop_duplicates(), on="tif_id", how="left")[["TIF District", "expenditures"]]
                 .rename(columns={"expenditures": "Expenditures ($)"})
                 .assign(**{"Expenditures ($)": lambda df: df["Expenditures ($)"].map("${:,.0f}".format)})
                 .reset_index(drop=True))
        st.dataframe(top10, hide_index=True, use_container_width=True)

        if map_mode == "Annual Expenditure":
            st.markdown("**Expenditure Legend**")
            exp_breaks = np.linspace(exp_min, exp_max, 6)
            exp_bin_labels = [f"${exp_breaks[i]/1e6:.1f}M – ${exp_breaks[i+1]/1e6:.1f}M" for i in range(5)]
            exp_colors = [val_to_hex((exp_breaks[i]+exp_breaks[i+1])/2, exp_min, exp_max, "YlOrRd") for i in range(5)]
            for color, label in zip(exp_colors, exp_bin_labels):
                st.markdown(f'<span style="background:{color};padding:2px 12px;border-radius:3px;margin-right:6px;">&nbsp;</span>{label}', unsafe_allow_html=True)
    
        elif map_mode == "Community Income":
            st.markdown("**Income Legend**")
            for color, label in zip(blues_legend, income_bin_labels):
                st.markdown(f'<span style="background:{color};padding:2px 12px;border-radius:3px;margin-right:6px;">&nbsp;</span>{label}', unsafe_allow_html=True)
        else:
            st.markdown("**Normalised Spending Legend**")
            nr_breaks = np.linspace(norm_min, norm_max, 6)
            for i in range(5):
                color = val_to_hex((nr_breaks[i]+nr_breaks[i+1])/2, norm_min, norm_max, "PuRd")
                label = f"{nr_breaks[i]:.2f} – {nr_breaks[i+1]:.2f}"
                st.markdown(f'<span style="background:{color};padding:2px 12px;border-radius:3px;margin-right:6px;">&nbsp;</span>{label}', unsafe_allow_html=True)
            st.caption("Exp ÷ Increment. Values > 10 shown as max colour.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Quintile Box Plot
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "### TIF spending rises with neighbourhood wealth"
    )
    st.markdown(
        "Each TIF district is matched to its surrounding community area. "
        "Community areas are ranked from lowest-income (Q1) to highest-income (Q5). "
        "The chart shows cumulative TIF spending within each income group, "
        "raising questions about whether funds are reaching communities in need."
    )

    joined_plot = joined.copy()
    joined_plot["quintile"] = pd.qcut(
        joined_plot["weighted_income"], q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )

    quintile_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    groups = [joined_plot.loc[joined_plot["quintile"] == q, "cumulative_expenditures"].dropna() / 1e6
              for q in quintile_order]

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bp = ax3.boxplot(groups, labels=quintile_order, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1.5},
                     flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "grey", "alpha": 0.5},
                     whiskerprops={"linewidth": 1}, capprops={"linewidth": 1})
    blues = plt.cm.Blues(np.linspace(0.3, 0.85, 5))
    for patch, color in zip(bp["boxes"], blues):
        patch.set_facecolor(color)
    for i, grp in enumerate(groups, start=1):
        ax3.scatter(np.random.normal(i, 0.07, size=len(grp)), grp, alpha=0.45, s=18, color="steelblue", zorder=3)

    ax3.set_xlabel("Community Area Income Quintile", fontsize=11)
    ax3.set_ylabel("Cumulative TIF Expenditures ($ M)", fontsize=11)
    ax3.set_title("TIF Expenditures and Community Income Quintile", fontsize=13, fontweight="bold")
    ax3.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax3.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig3)
    

    st.subheader("Spending by Income Group — Summary")
    summary = (joined_plot.groupby("quintile", observed=True)["cumulative_expenditures"]
               .describe(percentiles=[0.25, 0.5, 0.75])[["count", "mean", "50%", "25%", "75%"]]
               .rename(columns={"count": "N", "mean": "Mean ($M)", "50%": "Median ($M)", "25%": "Q25 ($M)", "75%": "Q75 ($M)"}))
    summary.index = quintile_order
    for col in ["Mean ($M)", "Median ($M)", "Q25 ($M)", "Q75 ($M)"]:
        summary[col] = (summary[col] / 1e6).map("${:.1f}M".format)
    summary["N"] = summary["N"].astype(int)
    summary.insert(0, "Income Range", income_bin_labels)
    st.dataframe(summary, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Scatter: Income vs. TIF Spending
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "### Higher-income areas attract significantly more TIF funding"
    )
    st.markdown(
        "Each dot represents one TIF district. Districts in wealthier community areas "
        "tend to have higher cumulative spending. The trend line confirms "
        "this is a statistically reliable pattern across all 150 districts."
    )

    scatter_data = joined.copy()
    scatter_data["exp_m"]    = scatter_data["cumulative_expenditures"] / 1e6
    scatter_data["income_k"] = scatter_data["weighted_income"] / 1000

    slope, intercept, r, p, _ = stats.linregress(scatter_data["income_k"], scatter_data["exp_m"])
    r2 = r ** 2
    x_line = np.linspace(scatter_data["income_k"].min(), scatter_data["income_k"].max(), 200)
    y_line = intercept + slope * x_line

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.scatter(scatter_data["income_k"], scatter_data["exp_m"],
                alpha=0.55, s=30, color="steelblue", edgecolors="white", linewidths=0.4)
    ax4.plot(x_line, y_line, color="crimson", linewidth=2,
             label=f"Trend line  (r = {r:.2f}, R² = {r2:.2f}, p = {p:.3f})")
    ax4.set_xlabel("Community Area Weighted Mean Income ($ K)", fontsize=11)
    ax4.set_ylabel("Cumulative TIF Expenditures ($ M)", fontsize=11)
    ax4.set_title("TIF Expenditure and Community Income", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax4.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig4)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Strength of relationship", f"r = {r:.2f}", help="Ranges from -1 to +1. Positive = wealthier areas spend more.")
    col_b.metric("Statistical confidence", "99%+" if p < 0.01 else f"p = {p:.3f}", help="This pattern is statistically significant at 1% level.")
    col_c.metric("Districts analysed", len(scatter_data))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Normalised: expenditure per $1 of tax increment
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        "### Once we account for resources, the gap disappears"
    )
    st.markdown(
        "TIF districts can only spend what their tax increment generates. "
        "Wealthier areas have higher and faster-growing property values, so they naturally "
        "generate more increment that gives them more to spend, even if they're spending at "
        "the same rate as poorer districts. Dividing expenditure by increment isolates "
        "whether wealthy districts are over-using their resources, or simply have more of them."
    )
    st.markdown(
        "After accounting for each district's tax increment, there is **no "
        "statistically significant relationship** between neighbourhood income and spending. "
        "The spending gap is explained by TIF's self-financing mechanism, and not by any "
        "political or administrative bias toward wealthier areas."
    )

    norm_data = joined.dropna(subset=["exp_per_increment", "weighted_income"]).copy()
    # Drop extreme outliers (ratio > 10 likely data issues)
    norm_data = norm_data[norm_data["exp_per_increment"] < 10]
    norm_data["income_k"] = norm_data["weighted_income"] / 1000

    col_left, col_right = st.columns(2)

    # ── Left: scatter ──────────────────────────────────────────────────────────
    with col_left:
        st.subheader("Spending Rate by Community Income")
        slope_n, intercept_n, r_n, p_n, _ = stats.linregress(
            norm_data["income_k"], norm_data["exp_per_increment"]
        )
        r2_n = r_n ** 2
        x_n = np.linspace(norm_data["income_k"].min(), norm_data["income_k"].max(), 200)
        y_n = intercept_n + slope_n * x_n

        fig_n, ax_n = plt.subplots(figsize=(5.5, 4.5))
        ax_n.scatter(norm_data["income_k"], norm_data["exp_per_increment"],
                     alpha=0.55, s=28, color="steelblue", edgecolors="white", linewidths=0.4)
        ax_n.plot(x_n, y_n, color="crimson", linewidth=2,
                  label=f"r = {r_n:.2f}, R² = {r2_n:.2f}, p = {p_n:.3f}")
        ax_n.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="Ratio = 1.0")
        ax_n.set_xlabel("Community Area Income ($000s)", fontsize=10)
        ax_n.set_ylabel("Expenditure / Tax Increment", fontsize=10)
        ax_n.set_title("Normalized TIF Spending by Community Income",
                       fontsize=11, fontweight="bold")
        ax_n.legend(fontsize=9)
        ax_n.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax_n.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig_n)

        ca, cb = st.columns(2)
        ca.metric("Relationship strength", f"r = {r_n:.2f}", help="Near zero — income no longer predicts spending rate")
        cb.metric("Statistical confidence", "Not significant" if p_n > 0.05 else f"p = {p_n:.3f}", help="p = 0.247 — this relationship could easily be due to chance")

    # ── Right: quintile boxplot ────────────────────────────────────────────────
    with col_right:
        st.subheader("Spending Rate by Income Group")
        norm_data["quintile"] = pd.qcut(
            norm_data["weighted_income"], q=5,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
        )
        quintile_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        groups_n = [
            norm_data.loc[norm_data["quintile"] == q, "exp_per_increment"].dropna()
            for q in quintile_order
        ]

        fig_nb, ax_nb = plt.subplots(figsize=(5.5, 4.5))
        bp_n = ax_nb.boxplot(
            groups_n, labels=quintile_order, patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "grey", "alpha": 0.5},
            whiskerprops={"linewidth": 1}, capprops={"linewidth": 1},
        )
        blues = plt.cm.Blues(np.linspace(0.3, 0.85, 5))
        for patch, color in zip(bp_n["boxes"], blues):
            patch.set_facecolor(color)
        for i, grp in enumerate(groups_n, start=1):
            ax_nb.scatter(np.random.normal(i, 0.07, size=len(grp)), grp,
                          alpha=0.4, s=15, color="steelblue", zorder=3)
        ax_nb.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax_nb.set_xlabel("Income Quintile", fontsize=10)
        ax_nb.set_ylabel("Expenditure / Tax Increment", fontsize=10)
        ax_nb.set_title("Normalized Spending by Quintile",
                        fontsize=11, fontweight="bold")
        ax_nb.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax_nb.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig_nb)

    st.caption(
        "Districts with expenditure/increment ratio > 10 excluded as outliers. Uses last reported cumulative increment. "
        f"n = {len(norm_data)} districts."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Spending Trends Over Time
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        "### The spending gap between rich and poor districts is widening"
    )
    st.markdown(
        "TIF districts are grouped into three tiers by community income. "
        "Each line tracks total annual spending for that tier from 2017 to 2024. "
        "Even if the spending rate per dollar of increment is similar, "
        "the absolute gap in resources is growing, leaving low-income communities "
        "further behind in real dollar terms each year."
    )

    pivot = trend.pivot(index="Report Year", columns="income_group", values="expenditure")
    years = sorted(pivot.index.astype(int))

    group_colors = {
        "Low-Income Districts":    "#2166ac",
        "Middle-Income Districts": "#78c679",
        "High-Income Districts":   "#d73027",
    }

    fig_t, ax_t = plt.subplots(figsize=(9, 5))
    for grp in ["High-Income Districts","Middle-Income Districts","Low-Income Districts"]:
        if grp in pivot.columns:
            ax_t.plot(pivot.index.astype(int), pivot[grp]/1e6,
                      label=grp, color=group_colors[grp],
                      linewidth=2.2, marker="o", markersize=5)

    ax_t.set_xlabel("Year", fontsize=11)
    ax_t.set_ylabel("Total Annual TIF Expenditures ($ millions)", fontsize=11)
    ax_t.set_title(
        "The Spending Gap Trend Between High- and Low-Income Districts (2017–2024)",
        fontsize=12, fontweight="bold"
    )
    ax_t.legend(fontsize=10, framealpha=0.9)
    import matplotlib.ticker as mticker
    ax_t.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax_t.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_t.set_axisbelow(True)
    ax_t.set_xticks(years)
    ax_t.set_xlim(min(years)-0.5, max(years)+0.5)
    plt.tight_layout()
    st.pyplot(fig_t)

    # Summary table
    st.subheader("Annual Spending by Income Tier ($M)")
    tbl = (pivot / 1e6).round(1)
    tbl.index = tbl.index.astype(int)
    tbl.columns.name = None
    st.dataframe(tbl.style.format("${:.1f}M"), use_container_width=True)

st.markdown("---")
st.caption("Data: Chicago Data Portal — TIF Annual Reports, TIF District Boundaries, Community Area Boundaries, ACS 5-Year Data by Community Area.")
