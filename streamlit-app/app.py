"""
app.py

Interactive dashboard: Chicago TIF expenditures vs. community area income.

Run locally (from repo root):
    streamlit run streamlit-app/app.py

Deploy: Streamlit Community Cloud — point to streamlit-app/app.py.
Note: Streamlit apps need to be "woken up" if they have not been run in the
last 24 hours. This is normal behaviour, not a bug.
"""

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

st.set_page_config(page_title="Chicago TIF vs. Income", page_icon="🏙️", layout="wide")
st.title("🏙️ Chicago TIF Spending vs. Community Income")

TIF_ANNUAL  = "data/raw-data/Tax_Increment_Financing_(TIF)_Annual_Report_-_Analysis_of_Special_Tax_Allocation_Fund_20260301.csv"
TIF_BOUNDS  = "data/raw-data/Boundaries_-_Tax_Increment_Financing_Districts_20260301.csv"
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

    tif_b = pd.read_csv(TIF_BOUNDS)
    tif_b["geometry"] = tif_b["the_geom"].apply(wkt.loads)
    tif_geo = gpd.GeoDataFrame(tif_b, geometry="geometry", crs="EPSG:4326")
    tif_geo = tif_geo.rename(columns={"NAME": "TIF District"})
    tif_cumulative = tif_raw.groupby("TIF District", as_index=False).agg(
        cumulative_expenditures=("Total Expenditure", "sum")
    )
    tif_full = tif_geo.merge(tif_cumulative, on="TIF District", how="left")

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
        tif_c[["TIF District", "cumulative_expenditures", "geometry"]],
        comm[["Community Area", "weighted_income", "geometry"]],
        how="left", predicate="within",
    ).dropna(subset=["weighted_income", "cumulative_expenditures"])
    joined["quintile"] = pd.qcut(
        joined["weighted_income"], q=5,
        labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"]
    )
    return tif_raw, tif_full, comm, joined

tif_raw, tif_full, comm, joined = load_all()

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
    .groupby("TIF District", as_index=False)
    .agg(expenditures=("Total Expenditure", "sum"))
)
tif_income_lookup = joined[["TIF District", "weighted_income", "Community Area"]].drop_duplicates("TIF District")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Income Map",
    "🗺️ Expenditure Map",
    "📊 Expenditure by Quintile",
    "📈 Income vs. TIF Spending",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Income Map: TIF districts filled by community income
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "Each polygon is one **TIF district**, filled by the **household income** "
        "of its surrounding community area. Darker blue = wealthier neighbourhood. "
        "Hover for details."
    )

    tif_map1 = tif_full.merge(tif_year, on="TIF District", how="left")
    tif_map1 = tif_map1.merge(tif_income_lookup, on="TIF District", how="left")
    inc_min, inc_max = income_bins[0], income_bins[5]

    m1 = folium.Map(location=[41.83, -87.68], zoom_start=11, tiles="CartoDB positron")

    def style_income(feature):
        inc = feature["properties"].get("weighted_income")
        try:
            fill = val_to_hex(np.clip(float(inc), inc_min, inc_max), inc_min, inc_max, "Blues")
            return {"fillColor": fill, "fillOpacity": 0.8, "color": "white", "weight": 0.5}
        except (TypeError, ValueError):
            return {"fillColor": "#dddddd", "fillOpacity": 0.4, "color": "white", "weight": 0.5}

    folium.GeoJson(
        tif_map1[["TIF District", "Community Area", "weighted_income", "expenditures", "geometry"]].__geo_interface__,
        style_function=style_income,
        highlight_function=lambda x: {"weight": 2, "color": "#000", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(
            fields=["TIF District", "Community Area", "weighted_income", "expenditures"],
            aliases=["TIF District", "Community Area", f"Community Income ($)", f"Expenditures ({selected_year}) $"],
            localize=True,
        ),
    ).add_to(m1)

    col1, col2 = st.columns([3, 1])
    with col1:
        st_folium(m1, width=900, height=620, key="map1")
    with col2:
        st.subheader(f"Top 10 Districts ({selected_year})")
        top10 = (tif_year.nlargest(10, "expenditures")[["TIF District", "expenditures"]]
                 .rename(columns={"expenditures": "Expenditures ($)"})
                 .assign(**{"Expenditures ($)": lambda df: df["Expenditures ($)"].map("${:,.0f}".format)})
                 .reset_index(drop=True))
        st.dataframe(top10, hide_index=True, use_container_width=True)
        st.markdown("**Income Legend**")
        for color, label in zip(blues_legend, income_bin_labels):
            st.markdown(f'<span style="background:{color};padding:2px 12px;border-radius:3px;margin-right:6px;">&nbsp;</span>{label}', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Expenditure Map: TIF districts filled by annual spending
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "Each polygon is one **TIF district**, filled by its **annual expenditure** "
        "for the selected year. Darker orange/red = higher spending. "
        "Hover for details."
    )

    tif_map2 = tif_full.merge(tif_year, on="TIF District", how="left")
    exp_vals = tif_map2["expenditures"].dropna()
    exp_min = exp_vals.quantile(0.05) if len(exp_vals) > 0 else 0
    exp_max = exp_vals.quantile(0.95) if len(exp_vals) > 0 else 1

    m2 = folium.Map(location=[41.83, -87.68], zoom_start=11, tiles="CartoDB positron")

    def style_expenditure(feature):
        val = feature["properties"].get("expenditures")
        try:
            val = float(val)
            if np.isnan(val): raise ValueError
            fill = val_to_hex(np.clip(val, exp_min, exp_max), exp_min, exp_max, "YlOrRd")
            return {"fillColor": fill, "fillOpacity": 0.85, "color": "#555", "weight": 0.4}
        except (TypeError, ValueError):
            return {"fillColor": "#dddddd", "fillOpacity": 0.3, "color": "#aaa", "weight": 0.4}

    folium.GeoJson(
        tif_map2[["TIF District", "expenditures", "geometry"]].__geo_interface__,
        style_function=style_expenditure,
        highlight_function=lambda x: {"weight": 2, "color": "#000", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(
            fields=["TIF District", "expenditures"],
            aliases=["TIF District", f"Expenditures ({selected_year}) $"],
            localize=True,
        ),
    ).add_to(m2)

    col1b, col2b = st.columns([3, 1])
    with col1b:
        st_folium(m2, width=900, height=620, key="map2")
    with col2b:
        st.subheader(f"Top 10 Districts ({selected_year})")
        top10b = (tif_year.nlargest(10, "expenditures")[["TIF District", "expenditures"]]
                  .rename(columns={"expenditures": "Expenditures ($)"})
                  .assign(**{"Expenditures ($)": lambda df: df["Expenditures ($)"].map("${:,.0f}".format)})
                  .reset_index(drop=True))
        st.dataframe(top10b, hide_index=True, use_container_width=True)
        st.markdown("**Expenditure Legend**")
        for frac, label in [(0.1, "Low"), (0.35, "Low-Mid"), (0.6, "Mid-High"), (0.85, "Highest")]:
            color = val_to_hex(exp_min + (exp_max - exp_min) * frac, exp_min, exp_max, "YlOrRd")
            st.markdown(f'<span style="background:{color};padding:2px 12px;border-radius:3px;margin-right:6px;">&nbsp;</span>{label}', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Quintile Box Plot
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "Each TIF district is assigned to the community area its centroid falls in. "
        "Community areas are ranked into **five income quintiles** (Q1 = lowest, Q5 = highest). "
        "Values below 1st and above 99th percentile are trimmed."
    )

    p1  = joined["cumulative_expenditures"].quantile(0.01)
    p99 = joined["cumulative_expenditures"].quantile(0.99)
    joined_trim = joined[joined["cumulative_expenditures"].between(p1, p99)].copy()
    joined_trim["quintile"] = pd.qcut(
        joined_trim["weighted_income"], q=5,
        labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"]
    )

    quintile_order = ["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"]
    groups = [joined_trim.loc[joined_trim["quintile"] == q, "cumulative_expenditures"].dropna() / 1e6
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
    ax3.set_ylabel("Cumulative TIF Expenditures ($ millions)", fontsize=11)
    ax3.set_title("TIF Expenditures Are Highest in the Wealthiest Neighbourhoods", fontsize=13, fontweight="bold")
    ax3.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax3.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig3)
    st.caption("Values below 1st and above 99th percentile trimmed.")

    st.subheader("Summary Statistics by Quintile")
    summary = (joined_trim.groupby("quintile", observed=True)["cumulative_expenditures"]
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
with tab4:
    st.markdown(
        "Each point is one **TIF district**, plotted by its community area income (x) "
        "against cumulative TIF expenditure (y). Top 1% trimmed."
    )

    p99s = joined["cumulative_expenditures"].quantile(0.99)
    scatter_data = joined[joined["cumulative_expenditures"] <= p99s].copy()
    scatter_data["exp_m"]    = scatter_data["cumulative_expenditures"] / 1e6
    scatter_data["income_k"] = scatter_data["weighted_income"] / 1000

    slope, intercept, r, p, _ = stats.linregress(scatter_data["income_k"], scatter_data["exp_m"])
    x_line = np.linspace(scatter_data["income_k"].min(), scatter_data["income_k"].max(), 200)
    y_line = intercept + slope * x_line

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.scatter(scatter_data["income_k"], scatter_data["exp_m"],
                alpha=0.55, s=30, color="steelblue", edgecolors="white", linewidths=0.4)
    ax4.plot(x_line, y_line, color="crimson", linewidth=2, label=f"Trend line  (r = {r:.2f}, p = {p:.3f})")
    ax4.set_xlabel("Community Area Weighted Mean Income ($000s)", fontsize=11)
    ax4.set_ylabel("Cumulative TIF Expenditures ($ millions)", fontsize=11)
    ax4.set_title("Wealthier Neighbourhoods Attract More TIF Spending", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax4.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig4)
    st.caption(f"Top 1% of expenditure values excluded (>{p99s/1e6:.0f}M). n = {len(scatter_data)} TIF districts.")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Correlation (r)", f"{r:.2f}")
    col_b.metric("p-value", f"{p:.3f}")
    col_c.metric("TIF Districts (trimmed)", len(scatter_data))
    col_d.metric("R²", f"{r**2:.2f}")

st.markdown("---")
st.caption("Data: Chicago Data Portal — TIF Annual Reports, TIF District Boundaries, Community Area Boundaries, ACS 5-Year Data by Community Area.")
