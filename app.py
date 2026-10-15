"""
NovaRetail Customer Intelligence Dashboard
==========================================
Interactive Streamlit dashboard for Sophia Martinez, Director of Customer
Intelligence, built on NR_dataset.xlsx. It supports three decisions:
  a) find growth opportunities in the existing customer base,
  b) detect early warning signs of customer decline,
  c) prioritize commercial and marketing actions.

Deployed from a public GitHub repository on Streamlit Community Cloud.
Run locally with:  streamlit run app.py
"""

from datetime import date
from pathlib import Path

import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI


# =============================================================================
# 1. CONSTANTS
# =============================================================================

# Path relative to this file, so it works locally and on Streamlit Cloud.
DATA_FILE = Path(__file__).parent / "NR_dataset.xlsx"
SHEET_NAME = "data"

REQUIRED_COLUMNS = [
    "idx", "label", "CustomerID", "TransactionID", "TransactionDate",
    "ProductCategory", "PurchaseAmount", "CustomerAgeGroup", "CustomerGender",
    "CustomerRegion", "CustomerSatisfaction", "RetailChannel",
]

UNLABELED = "Unlabeled"
SEGMENT_ORDER = ["Promising", "Growth", "Stable", "Decline", UNLABELED]

# Colorblind-safe categorical palette, validated for adjacent and all-pairs
# separation. Each segment keeps its color in every chart.
SEGMENT_COLORS = {
    "Promising": "#1baf7a",   # aqua
    "Growth": "#2a78d6",      # blue
    "Stable": "#eda100",      # yellow
    "Decline": "#e34948",     # red
    UNLABELED: "#8c8b86",     # neutral gray
}
OTHERS_COLOR = "#8c8b86"
SEQUENTIAL_SCALE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
GRID_COLOR = "rgba(128, 128, 128, 0.25)"   # recessive in both light and dark themes

AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_FIXES = {"55-64": "55+"}

# The raw file has 35 inconsistent category names; they are grouped here.
CATEGORY_GROUPS = {
    "Electronics": ["Electronics", "Gaming"],
    "Clothing & Fashion": [
        "Clothing", "Fashion", "Fashion & Apparel", "Fashion Accessories",
        "Sportswear", "Children's Clothing",
    ],
    "Groceries & Food": ["Groceries", "Grocery", "Grocery Items", "Food & Beverages"],
    "Books": ["Books", "Books & Magazines"],
    "Home & Furniture": [
        "Home Appliances", "Home Decor", "Home & Garden", "Home Improvement",
        "Furniture", "Furniture & Decor", "Gardening Tools",
    ],
    "Health & Beauty": [
        "Health & Wellness", "Health Supplements", "Health & Beauty",
        "Beauty Products", "Beauty & Personal Care", "Cosmetics",
    ],
    "Sports & Outdoors": [
        "Sporting Goods", "Sports & Outdoors", "Sports Equipment", "Outdoor Equipment",
    ],
    "Toys & Games": ["Toys", "Toys & Games"],
    "Other": ["Office Supplies", "Automotive"],
}
CATEGORY_LOOKUP = {raw: group for group, raws in CATEGORY_GROUPS.items() for raw in raws}
CATEGORY_ORDER = list(CATEGORY_GROUPS.keys())

LOW_SAT_THRESHOLD = 2     # satisfaction <= 2 counts as "low"
MIN_RELIABLE_N = 10       # warn when a figure rests on fewer purchase lines

TAB_NAMES = [
    "Overview",
    "Segments & Growth Opportunities",
    "Early Warning Signs",
    "Region, Category & Channel",
    "Ask the Data (AI)",
    "Data Table & Download",
    "About the Data",
]

# --- AI assistant (Groq, OpenAI-compatible API) ---
PROVIDER_NAME = "Groq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = ["openai/gpt-oss-20b"]   # tried in order if a model is retired
HISTORY_LENGTH = 6                          # past messages sent with each question
SUGGESTED_QUESTIONS = [
    "Which segment generates the most revenue?",
    "What early warning signs do you see in the Decline segment?",
    "Where should we focus investment to grow revenue?",
]
MODEL_UNAVAILABLE_PHRASES = [
    "not found", "decommissioned", "deprecated", "does not exist",
    "not available", "model_not_found", "model_decommissioned",
]
SYSTEM_PROMPT = """You are a concise data assistant for NovaRetail's Customer Intelligence team, \
working for Sophia Martinez, Director of Customer Intelligence.

Rules:
- Answer ONLY from the DATA SUMMARY below. It reflects the user's current sidebar filters. \
When the filters are not "all", start your answer by naming the filters it is based on.
- Amounts are revenue in U.S. dollars. Rows are purchase lines, not customers; say "customers" \
only for unique customer counts.
- The data is a small sample (about 34 customers over about five weeks in early 2023). When a \
figure rests on fewer than 10 purchase lines, say so.
- When asked for recommendations, base each one on specific numbers from the summary and label \
it as a suggestion for Sophia's team to evaluate.
- If a question cannot be answered from the data (profit, costs, margins, customer lifetime, \
other years, reasons behind a customer's behavior), say so and name the data that would be needed.
- Use $ and thousands separators, double-check arithmetic, and keep answers short.

DATA SUMMARY
{context}"""

# Sidebar multiselect filters: (label, dataframe column, session-state key).
MULTISELECT_FILTERS = [
    ("Segment", "Segment", "f_segment"),
    ("Region", "CustomerRegion", "f_region"),
    ("Product category group", "CategoryGroup", "f_category"),
    ("Retail channel", "RetailChannel", "f_channel"),
    ("Age group", "AgeGroup", "f_age"),
    ("Gender", "CustomerGender", "f_gender"),
]


# =============================================================================
# 2. DATA LOADING
# =============================================================================

@st.cache_data(show_spinner="Loading NovaRetail data…")
def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the workbook exactly as delivered."""
    return pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")


# =============================================================================
# 3. CLEANING AND VALIDATION
# =============================================================================

@st.cache_data
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready copy of the raw data. All cleaning lives here."""
    df = raw.copy()
    df["Segment"] = df["label"].fillna(UNLABELED).astype(str)
    df["AgeGroup"] = df["CustomerAgeGroup"].replace(AGE_FIXES)
    df["CategoryGroup"] = df["ProductCategory"].map(CATEGORY_LOOKUP).fillna("Other")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    # Week starting Monday, used for the trend chart.
    df["WeekStart"] = df["TransactionDate"].dt.to_period("W-SUN").dt.start_time
    df["LowSatisfaction"] = df["CustomerSatisfaction"] <= LOW_SAT_THRESHOLD
    return df


def run_validation_checks(raw: pd.DataFrame) -> pd.DataFrame:
    """Data-quality checks shown on the About tab."""
    checks = []

    def add(check, status, detail):
        checks.append({"Check": check, "Status": status, "Detail": detail})

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    add("Required columns present",
        "✅ Pass" if not missing_cols else "⚠️ Fail",
        "All 12 columns found." if not missing_cols else f"Missing: {missing_cols}")
    if missing_cols:
        return pd.DataFrame(checks)

    n_missing = int(raw["label"].isna().sum())
    add("Missing segment labels", "ℹ️ Handled" if n_missing else "✅ Pass",
        f"{n_missing} row(s) without a label, shown as '{UNLABELED}'.")

    n_age = int(raw["CustomerAgeGroup"].isin(AGE_FIXES.keys()).sum())
    add("Age-group consistency", "ℹ️ Handled" if n_age else "✅ Pass",
        f"{n_age} row(s) with '55-64' merged into '55+'.")

    unmapped = sorted(set(raw["ProductCategory"]) - set(CATEGORY_LOOKUP))
    add("Category mapping coverage", "✅ Pass" if not unmapped else "⚠️ Check",
        f"{raw['ProductCategory'].nunique()} raw categories mapped into "
        f"{len(CATEGORY_GROUPS)} groups." + (f" Unmapped (sent to Other): {unmapped}" if unmapped else ""))

    dup_ids = raw["TransactionID"].duplicated(keep=False)
    add("TransactionID uniqueness", "ℹ️ Info",
        f"{raw['TransactionID'].nunique()} distinct IDs across {len(raw)} rows; "
        f"{int(dup_ids.sum())} rows share an ID. idx is used as the row key instead.")

    multi = raw.dropna(subset=["label"]).groupby("CustomerID")["label"].nunique()
    add("Customers with more than one label", "ℹ️ Info",
        f"{int((multi > 1).sum())} customers appear under several segments, so "
        "segment analysis is at the purchase-line level.")

    sat_ok = raw["CustomerSatisfaction"].between(1, 5).all()
    add("Satisfaction within 1-5", "✅ Pass" if sat_ok else "⚠️ Fail",
        f"Observed range {raw['CustomerSatisfaction'].min()}–{raw['CustomerSatisfaction'].max()}.")

    amt_ok = (raw["PurchaseAmount"] > 0).all()
    add("Purchase amounts above $0", "✅ Pass" if amt_ok else "⚠️ Fail",
        f"Observed range ${raw['PurchaseAmount'].min():,.2f}–${raw['PurchaseAmount'].max():,.2f}.")

    dates = pd.to_datetime(raw["TransactionDate"])
    add("Date range", "ℹ️ Info", f"{dates.min():%b %d, %Y} to {dates.max():%b %d, %Y}.")

    add("Row key (idx) unique", "✅ Pass" if raw["idx"].is_unique else "⚠️ Fail",
        f"{len(raw)} rows, {raw['CustomerID'].nunique()} unique customers.")
    return pd.DataFrame(checks)


# =============================================================================
# 4. FILTERS
# =============================================================================

def ordered_options(df: pd.DataFrame, column: str) -> list:
    """Filter options in a meaningful order (segments, ages, categories)."""
    present = set(df[column].dropna().unique())
    preferred = {"Segment": SEGMENT_ORDER, "AgeGroup": AGE_ORDER,
                 "CategoryGroup": CATEGORY_ORDER}.get(column)
    if preferred:
        return [v for v in preferred if v in present]
    return sorted(present)


def filter_defaults(df: pd.DataFrame) -> dict:
    """Default value for every filter widget, keyed by session-state key."""
    defaults = {}
    for _, column, key in MULTISELECT_FILTERS:
        defaults[key] = ordered_options(df, column)
        defaults[f"{key}_all"] = True
    defaults["f_dates"] = (df["TransactionDate"].min().date(), df["TransactionDate"].max().date())
    defaults["f_sat"] = (1, 5)
    return defaults


def reset_filters(defaults: dict) -> None:
    """Callback for the Reset Filters button."""
    for key, value in defaults.items():
        st.session_state[key] = value


def render_sidebar(df: pd.DataFrame) -> tuple[dict, str]:
    """Draw the sidebar filters; return the selections and a readable summary."""
    defaults = filter_defaults(df)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    sb = st.sidebar
    sb.header("Filters")
    sb.button("Reset filters", on_click=reset_filters, args=(defaults,), width="stretch")

    selections = {}
    for label, column, key in MULTISELECT_FILTERS:
        options = defaults[key]
        select_all = sb.checkbox(f"All {label.lower()}s", key=f"{key}_all")
        chosen = sb.multiselect(label, options, key=key, disabled=select_all)
        selections[column] = options if select_all else chosen

    min_d, max_d = defaults["f_dates"]
    picked = sb.date_input("Date range", key="f_dates", min_value=min_d, max_value=max_d)
    # While the user is mid-selection, date_input can return a single date.
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
    else:
        start = picked[0] if isinstance(picked, (tuple, list)) and picked else min_d
        end = max_d
    selections["dates"] = (start, end)
    selections["satisfaction"] = sb.slider("Satisfaction score", 1, 5, key="f_sat")

    summary = active_filter_summary(selections, defaults)
    sb.divider()
    sb.markdown("**Active filters**")
    sb.caption(summary)
    return selections, summary


def active_filter_summary(sel: dict, defaults: dict) -> str:
    """One readable line per filter that is narrower than 'all'."""
    lines = []
    for label, column, key in MULTISELECT_FILTERS:
        if len(sel[column]) < len(defaults[key]):
            shown = ", ".join(map(str, sel[column])) or "none selected"
            lines.append(f"{label}: {shown}")
    if tuple(sel["dates"]) != tuple(defaults["f_dates"]):
        lines.append(f"Dates: {sel['dates'][0]:%b %d} – {sel['dates'][1]:%b %d, %Y}")
    if tuple(sel["satisfaction"]) != (1, 5):
        lines.append(f"Satisfaction: {sel['satisfaction'][0]}–{sel['satisfaction'][1]}")
    return "  \n".join(lines) if lines else "None. Showing all data."


def apply_filters(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    """Return the rows that match every active filter."""
    mask = pd.Series(True, index=df.index)
    for _, column, _ in MULTISELECT_FILTERS:
        mask &= df[column].isin(sel[column])
    start, end = sel["dates"]
    mask &= df["TransactionDate"].dt.date.between(start, end)
    lo, hi = sel["satisfaction"]
    mask &= df["CustomerSatisfaction"].between(lo, hi)
    return df[mask]


# =============================================================================
# 5. KPIs AND SUMMARY TABLES
# =============================================================================

def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def compute_kpis(df: pd.DataFrame) -> dict:
    """KPI values for the cards (and, in step 2, for the chatbot context)."""
    revenue = df["PurchaseAmount"].sum()
    decline_rev = df.loc[df["Segment"] == "Decline", "PurchaseAmount"].sum()
    return {
        "revenue": revenue,
        "lines": len(df),
        "customers": df["CustomerID"].nunique(),
        "avg_purchase": df["PurchaseAmount"].mean() if len(df) else 0.0,
        "avg_satisfaction": df["CustomerSatisfaction"].mean() if len(df) else 0.0,
        "decline_rev_share": decline_rev / revenue if revenue else 0.0,
        "low_sat_share": df["LowSatisfaction"].mean() if len(df) else 0.0,
    }


def summarize_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Revenue, volume and satisfaction for each value of one column."""
    out = (
        df.groupby(column, observed=True)
        .agg(Revenue=("PurchaseAmount", "sum"),
             Lines=("idx", "count"),
             Customers=("CustomerID", "nunique"),
             AvgPurchase=("PurchaseAmount", "mean"),
             AvgSatisfaction=("CustomerSatisfaction", "mean"),
             LowSatShare=("LowSatisfaction", "mean"))
        .reset_index()
    )
    total = out["Revenue"].sum()
    out["RevenueShare"] = out["Revenue"] / total if total else 0.0
    if column == "Segment":
        out["Segment"] = pd.Categorical(out["Segment"], SEGMENT_ORDER, ordered=True)
        out = out.sort_values("Segment")
        out["Segment"] = out["Segment"].astype(str)
    return out


def small_n_note(n: int) -> str:
    return f" (small sample: {n} lines)" if n < MIN_RELIABLE_N else ""


def build_insights(df: pd.DataFrame) -> list:
    """Insight cards computed from the filtered data. Never hard-coded."""
    insights = []
    seg = summarize_by(df, "Segment")
    labeled = seg[seg["Segment"] != UNLABELED]

    if not labeled.empty:
        top = labeled.loc[labeled["Revenue"].idxmax()]
        insights.append((
            "Top revenue segment",
            f"**{top['Segment']}** brings **{top['RevenueShare']:.0%}** of revenue "
            f"({fmt_money(top['Revenue'])}) from {int(top['Lines'])} lines"
            f"{small_n_note(int(top['Lines']))}.",
        ))
        low = labeled.loc[labeled["AvgSatisfaction"].idxmin()]
        insights.append((
            "Lowest satisfaction",
            f"**{low['Segment']}** averages **{low['AvgSatisfaction']:.2f}/5**; "
            f"{low['LowSatShare']:.0%} of its lines rate {LOW_SAT_THRESHOLD} or lower"
            f"{small_n_note(int(low['Lines']))}.",
        ))

    cat = summarize_by(df, "CategoryGroup")
    cat = cat[cat["Lines"] >= 3]
    if not cat.empty:
        best = cat.loc[cat["AvgPurchase"].idxmax()]
        insights.append((
            "Highest average ticket",
            f"**{best['CategoryGroup']}** averages **{fmt_money(best['AvgPurchase'])}** "
            f"per purchase across {int(best['Lines'])} lines{small_n_note(int(best['Lines']))}.",
        ))

    decline = df[df["Segment"] == "Decline"]
    total = df["PurchaseAmount"].sum()
    if not decline.empty and total:
        insights.append((
            "Revenue at risk",
            f"Decline-labeled lines carry **{fmt_money(decline['PurchaseAmount'].sum())}** "
            f"(**{decline['PurchaseAmount'].sum() / total:.0%}** of revenue) with average "
            f"satisfaction {decline['CustomerSatisfaction'].mean():.2f}/5.",
        ))
    else:
        insights.append(("Revenue at risk", "No Decline-labeled lines in the current selection."))
    return insights


# =============================================================================
# 6. CHARTS
# =============================================================================

def style_fig(fig: go.Figure, height: int = 380, value_axis: str = "y") -> go.Figure:
    """Shared, low-decoration styling. Value axes always start at zero."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=96, b=10),
        title=dict(font_size=16, y=0.98, yanchor="top"),
        legend_title_text="",
        # Legend sits in its own row between the title and the plot.
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=13),
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    if value_axis == "y":
        fig.update_yaxes(rangemode="tozero")
    elif value_axis == "x":
        fig.update_xaxes(rangemode="tozero", showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(showgrid=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


SEGMENT_HOVER = (
    "<b>%{x}</b><br>Revenue: $%{customdata[0]:,.2f}<br>Share: %{customdata[1]:.0%}"
    "<br>Purchase lines: %{customdata[2]}<br>Customers: %{customdata[3]}"
    "<br>Avg purchase: $%{customdata[4]:,.2f}<br>Avg satisfaction: %{customdata[5]:.2f}"
    "<extra></extra>"
)
SEGMENT_CUSTOM = ["Revenue", "RevenueShare", "Lines", "Customers", "AvgPurchase", "AvgSatisfaction"]


def chart_revenue_by_segment(df: pd.DataFrame) -> go.Figure:
    seg = summarize_by(df, "Segment")
    fig = px.bar(seg, x="Segment", y="Revenue", color="Segment",
                 color_discrete_map=SEGMENT_COLORS, custom_data=SEGMENT_CUSTOM,
                 text=seg["Revenue"].map(lambda v: f"${v:,.0f}"),
                 title="Revenue by segment")
    fig.update_traces(hovertemplate=SEGMENT_HOVER, textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="Revenue (USD)", xaxis_title="")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return style_fig(fig)


def chart_satisfaction_by_segment(df: pd.DataFrame) -> go.Figure:
    seg = summarize_by(df, "Segment")
    fig = px.bar(seg, x="Segment", y="AvgSatisfaction", color="Segment",
                 color_discrete_map=SEGMENT_COLORS, custom_data=SEGMENT_CUSTOM,
                 text=seg["AvgSatisfaction"].map(lambda v: f"{v:.2f}"),
                 title="Average satisfaction by segment (1–5)")
    fig.update_traces(hovertemplate=SEGMENT_HOVER, textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="Average satisfaction", xaxis_title="")
    fig.update_yaxes(range=[0, 5.4], dtick=1)
    return style_fig(fig, value_axis=None)


def chart_weekly_trend(df: pd.DataFrame) -> go.Figure:
    weekly = (df.groupby(["WeekStart", "Segment"], observed=True)
                .agg(Revenue=("PurchaseAmount", "sum"), Lines=("idx", "count")))
    # A week with no purchases is $0, not a gap, so lines never skip weeks.
    full = pd.MultiIndex.from_product([sorted(df["WeekStart"].unique()), df["Segment"].unique()],
                                      names=["WeekStart", "Segment"])
    weekly = weekly.reindex(full, fill_value=0).reset_index()
    fig = px.line(weekly, x="WeekStart", y="Revenue", color="Segment", markers=True,
                  color_discrete_map=SEGMENT_COLORS,
                  category_orders={"Segment": SEGMENT_ORDER}, custom_data=["Segment", "Lines"],
                  title="Weekly revenue by segment")
    fig.update_traces(line_width=2, marker_size=8,
                      hovertemplate="<b>%{customdata[0]}</b><br>Week of %{x|%b %d}"
                                    "<br>Revenue: $%{y:,.2f}<br>Lines: %{customdata[1]}<extra></extra>")
    fig.update_layout(xaxis_title="Week starting", yaxis_title="Revenue (USD)")
    fig.update_xaxes(tickformat="%b %d", tickvals=sorted(weekly["WeekStart"].unique()))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return style_fig(fig)


def chart_heatmap(df: pd.DataFrame, rows: str, cols: str, title: str,
                  row_order=None, col_order=None) -> go.Figure:
    """Revenue heatmap with the dollar value written in each cell."""
    pivot = df.pivot_table(index=rows, columns=cols, values="PurchaseAmount",
                           aggfunc="sum", fill_value=0, observed=True)
    if row_order:
        pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    if col_order:
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    text = pivot.map(lambda v: f"${v:,.0f}" if v else "")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale=SEQUENTIAL_SCALE, text=text.values, texttemplate="%{text}",
        hovertemplate="%{y} · %{x}<br>Revenue: $%{z:,.2f}<extra></extra>",
        colorbar=dict(title="Revenue", tickprefix="$", tickformat=",.0f"),
        xgap=2, ygap=2,
    ))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=max(320, 60 * len(pivot.index) + 120), value_axis=None)


def chart_avg_purchase_by_category(df: pd.DataFrame) -> go.Figure:
    cat = summarize_by(df, "CategoryGroup").sort_values("AvgPurchase")
    fig = px.bar(cat, x="AvgPurchase", y="CategoryGroup", orientation="h",
                 text=cat["AvgPurchase"].map(lambda v: f"${v:,.0f}"),
                 custom_data=["Lines", "Revenue"], title="Average purchase by category group")
    fig.update_traces(marker_color=SEGMENT_COLORS["Growth"], textposition="outside", cliponaxis=False,
                      hovertemplate="<b>%{y}</b><br>Avg purchase: $%{x:,.2f}"
                                    "<br>Lines: %{customdata[0]}<br>Revenue: $%{customdata[1]:,.2f}<extra></extra>")
    fig.update_layout(xaxis_title="Average purchase (USD)", yaxis_title="")
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return style_fig(fig, height=max(320, 38 * len(cat) + 100), value_axis="x")


def chart_revenue_by(df: pd.DataFrame, column: str, title: str, split: bool,
                     horizontal: bool = False) -> go.Figure:
    """Revenue by one dimension, optionally stacked by segment."""
    if split:
        data = (df.groupby([column, "Segment"], observed=True)
                  .agg(Revenue=("PurchaseAmount", "sum"), Lines=("idx", "count")).reset_index())
        color_args = dict(color="Segment", color_discrete_map=SEGMENT_COLORS,
                          category_orders={"Segment": SEGMENT_ORDER})
    else:
        data = (df.groupby(column, observed=True)
                  .agg(Revenue=("PurchaseAmount", "sum"), Lines=("idx", "count")).reset_index())
        color_args = {}
    totals = data.groupby(column)["Revenue"].sum().sort_values(ascending=horizontal)
    order = {column: list(totals.index)}
    color_args.setdefault("category_orders", {}).update(order)
    x, y = ("Revenue", column) if horizontal else (column, "Revenue")
    fig = px.bar(data, x=x, y=y, orientation="h" if horizontal else "v",
                 custom_data=["Lines"], title=title, **color_args)
    value = "%{x:,.2f}" if horizontal else "%{y:,.2f}"
    label = "%{y}" if horizontal else "%{x}"
    fig.update_traces(marker_line_color="white", marker_line_width=1,
                      hovertemplate=f"<b>{label}</b><br>Revenue: ${value}"
                                    "<br>Lines: %{customdata[0]}<extra></extra>")
    if not split:
        fig.update_traces(marker_color=SEGMENT_COLORS["Growth"])
    fig.update_layout(barmode="stack", xaxis_title="Revenue (USD)" if horizontal else "",
                      yaxis_title="" if horizontal else "Revenue (USD)")
    if horizontal:
        fig.update_xaxes(tickprefix="$", tickformat=",.0f")
        return style_fig(fig, height=max(320, 38 * len(totals) + 120), value_axis="x")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return style_fig(fig)


def chart_low_sat_share(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    data = summarize_by(df, column).sort_values("LowSatShare", ascending=False)
    fig = px.bar(data, x=column, y="LowSatShare", custom_data=["Lines"], title=title,
                 text=data["LowSatShare"].map(lambda v: f"{v:.0%}"))
    fig.update_traces(marker_color=SEGMENT_COLORS["Decline"], textposition="outside", cliponaxis=False,
                      hovertemplate="<b>%{x}</b><br>Low-satisfaction share: %{y:.0%}"
                                    "<br>Lines: %{customdata[0]}<extra></extra>")
    fig.update_layout(xaxis_title="", yaxis_title=f"Share of lines rated ≤ {LOW_SAT_THRESHOLD}")
    fig.update_yaxes(tickformat=".0%", range=[0, max(0.1, data["LowSatShare"].max() * 1.2)])
    return style_fig(fig, height=340, value_axis=None)


def chart_decline_comparison(df: pd.DataFrame, metric: str, title: str, fmt: str) -> go.Figure:
    """One measure, Decline vs. all other lines (one axis per chart)."""
    data = df.assign(Group=df["Segment"].eq("Decline").map({True: "Decline", False: "All other"}))
    agg = {"AvgSatisfaction": ("CustomerSatisfaction", "mean"),
           "AvgPurchase": ("PurchaseAmount", "mean"),
           "LowSatShare": ("LowSatisfaction", "mean")}[metric]
    g = data.groupby("Group").agg(Value=agg, Lines=("idx", "count")).reindex(["Decline", "All other"]).dropna().reset_index()
    text = g["Value"].map(lambda v: format(v, fmt) if "%" in fmt else (f"${v:,.0f}" if fmt == "$" else f"{v:.2f}"))
    fig = px.bar(g, x="Group", y="Value", color="Group", text=text, custom_data=["Lines"],
                 color_discrete_map={"Decline": SEGMENT_COLORS["Decline"], "All other": OTHERS_COLOR},
                 title=title)
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="<b>%{x}</b><br>Value: %{text}<br>Lines: %{customdata[0]}<extra></extra>")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    if metric == "AvgSatisfaction":
        fig.update_yaxes(range=[0, 5.4], dtick=1)
    elif metric == "LowSatShare":
        fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
    else:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f", rangemode="tozero")
    return style_fig(fig, height=320, value_axis=None)


def chart_satisfaction_heatmap(df: pd.DataFrame) -> go.Figure:
    """Share of each segment's lines at each satisfaction score."""
    counts = pd.crosstab(df["Segment"], df["CustomerSatisfaction"])
    counts = counts.reindex(index=[s for s in SEGMENT_ORDER if s in counts.index],
                            columns=[1, 2, 3, 4, 5], fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    text = shares.map(lambda v: f"{v:.0%}" if v else "")
    fig = go.Figure(go.Heatmap(
        z=shares.values, x=[str(c) for c in shares.columns], y=list(shares.index),
        colorscale=SEQUENTIAL_SCALE, zmin=0, zmax=1, text=text.values, texttemplate="%{text}",
        customdata=counts.values, xgap=2, ygap=2,
        hovertemplate="%{y} · score %{x}<br>Share of lines: %{z:.0%}"
                      "<br>Lines: %{customdata}<extra></extra>",
        colorbar=dict(title="Share", tickformat=".0%"),
    ))
    fig.update_layout(title="Satisfaction distribution by segment (share of lines)",
                      xaxis_title="Satisfaction score")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=max(300, 55 * len(shares) + 120), value_axis=None)


def chart_purchase_vs_satisfaction(df: pd.DataFrame) -> go.Figure:
    """Scatter with a small horizontal jitter so overlapping scores stay visible."""
    plot = df.copy()
    offsets = {s: (i - 2) * 0.08 for i, s in enumerate(SEGMENT_ORDER)}
    plot["SatJitter"] = plot["CustomerSatisfaction"] + plot["Segment"].map(offsets)
    fig = px.scatter(plot, x="SatJitter", y="PurchaseAmount", color="Segment", symbol="Segment",
                     color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENT_ORDER},
                     custom_data=["CustomerID", "CategoryGroup", "CustomerSatisfaction", "RetailChannel"],
                     title="Purchase amount vs. satisfaction")
    fig.update_traces(marker=dict(size=10, line=dict(color="white", width=1.5)),
                      hovertemplate="Customer %{customdata[0]}<br>%{customdata[1]} · %{customdata[3]}"
                                    "<br>Amount: $%{y:,.2f}<br>Satisfaction: %{customdata[2]}<extra></extra>")
    fig.update_layout(xaxis_title="Satisfaction score", yaxis_title="Purchase amount (USD)")
    fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], range=[0.5, 5.5])
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return style_fig(fig, height=420)


# =============================================================================
# 7. PAGE SECTIONS AND TABS
# =============================================================================

def render_header(df: pd.DataFrame) -> None:
    st.title("NovaRetail Customer Intelligence Dashboard")
    st.markdown(
        "Explore which customers drive revenue, which segments show signs of decline, "
        "and where to focus investment, by segment, region, product category and channel."
    )
    st.caption("Source: NovaRetail sample dataset, NR_dataset.xlsx (course-provided).")
    st.info(
        f"**Small sample:** {len(df):,} purchase lines from {df['CustomerID'].nunique():,} customers, "
        f"{df['TransactionDate'].min():%b %d, %Y} to {df['TransactionDate'].max():%b %d, %Y}. "
        "Amounts are revenue in USD. Each row is a purchase line, not a customer.",
        icon="ℹ️",
    )


def render_kpis(df: pd.DataFrame) -> None:
    k = compute_kpis(df)
    row1 = st.columns(4)
    row1[0].metric("Total revenue", fmt_money(k["revenue"]))
    row1[1].metric("Purchase lines", f"{k['lines']:,}")
    row1[2].metric("Unique customers", f"{k['customers']:,}")
    row1[3].metric("Average purchase", fmt_money(k["avg_purchase"]))
    row2 = st.columns(4)
    row2[0].metric("Average satisfaction", f"{k['avg_satisfaction']:.2f} / 5")
    row2[1].metric("Revenue from Decline lines", f"{k['decline_rev_share']:.1%}",
                   help="Share of revenue from lines labeled Decline: revenue at risk.")
    row2[2].metric(f"Lines rated ≤ {LOW_SAT_THRESHOLD}", f"{k['low_sat_share']:.1%}",
                   help="Share of purchase lines with satisfaction of 1 or 2.")
    if k["lines"] < MIN_RELIABLE_N:
        row2[3].warning(f"Only {k['lines']} lines match. Treat figures with caution.", icon="⚠️")


def render_overview(df: pd.DataFrame) -> None:
    insights = build_insights(df)
    cols = st.columns(len(insights))
    for col, (title, text) in zip(cols, insights):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(text)
    left, right = st.columns(2)
    with left:
        show(chart_revenue_by_segment(df))
    with right:
        show(chart_satisfaction_by_segment(df))
    show(chart_weekly_trend(df))
    st.caption("About five weeks of data, so treat trends as directional.")


def render_segments(df: pd.DataFrame) -> None:
    seg = summarize_by(df, "Segment")
    seg["Note"] = seg["Lines"].map(lambda n: f"⚠️ n < {MIN_RELIABLE_N}" if n < MIN_RELIABLE_N else "")
    st.subheader("Segment scorecard")
    st.dataframe(
        seg[["Segment", "Revenue", "RevenueShare", "Lines", "Customers", "AvgPurchase",
             "AvgSatisfaction", "LowSatShare", "Note"]],
        hide_index=True, width="stretch",
        column_config={
            "Revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
            "RevenueShare": st.column_config.NumberColumn("Share of revenue", format="percent"),
            "Lines": st.column_config.NumberColumn("Purchase lines", format="%d"),
            "Customers": st.column_config.NumberColumn("Customers", format="%d"),
            "AvgPurchase": st.column_config.NumberColumn("Avg purchase", format="dollar"),
            "AvgSatisfaction": st.column_config.NumberColumn("Avg satisfaction", format="%.2f"),
            "LowSatShare": st.column_config.NumberColumn(f"Lines rated ≤ {LOW_SAT_THRESHOLD}", format="percent"),
        },
    )
    st.caption("Some customers appear under more than one segment, so segment customer counts "
               "do not add up to the total.")
    show(chart_heatmap(df, "Segment", "CategoryGroup", "Revenue by segment and category group",
                       SEGMENT_ORDER, CATEGORY_ORDER))
    show(chart_avg_purchase_by_category(df))


def render_early_warning(df: pd.DataFrame) -> None:
    if not df["Segment"].eq("Decline").any():
        st.info("No Decline-labeled lines in the current selection. Widen the Segment filter to compare.")
    st.subheader("Decline vs. all other lines")
    c1, c2, c3 = st.columns(3)
    with c1:
        show(chart_decline_comparison(df, "AvgSatisfaction", "Average satisfaction", ".2f"))
    with c2:
        show(chart_decline_comparison(df, "AvgPurchase", "Average purchase", "$"))
    with c3:
        show(chart_decline_comparison(df, "LowSatShare", f"Lines rated ≤ {LOW_SAT_THRESHOLD}", ".0%"))
    left, right = st.columns(2)
    with left:
        show(chart_satisfaction_heatmap(df))
    with right:
        show(chart_purchase_vs_satisfaction(df))

    st.subheader(f"Low-satisfaction purchases (rated ≤ {LOW_SAT_THRESHOLD}), highest value first")
    low = df[df["LowSatisfaction"]].sort_values("PurchaseAmount", ascending=False)
    if low.empty:
        st.info("No low-satisfaction lines in the current selection.")
    else:
        st.dataframe(
            low[["TransactionDate", "CustomerID", "Segment", "CategoryGroup", "ProductCategory",
                 "PurchaseAmount", "CustomerSatisfaction", "CustomerRegion", "RetailChannel"]],
            hide_index=True, width="stretch",
            column_config={
                "TransactionDate": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
                "CustomerID": st.column_config.NumberColumn("Customer", format="%d"),
                "CategoryGroup": "Category group",
                "ProductCategory": "Original category",
                "PurchaseAmount": st.column_config.NumberColumn("Amount", format="dollar"),
                "CustomerSatisfaction": "Satisfaction",
                "CustomerRegion": "Region",
                "RetailChannel": "Channel",
            },
        )


def render_region_category_channel(df: pd.DataFrame) -> None:
    split = st.toggle("Split bars by segment", value=True)
    left, right = st.columns(2)
    with left:
        show(chart_revenue_by(df, "CustomerRegion", "Revenue by region", split))
    with right:
        show(chart_revenue_by(df, "RetailChannel", "Revenue by channel", split))
    show(chart_revenue_by(df, "CategoryGroup", "Revenue by category group", split, horizontal=True))
    show(chart_heatmap(df, "CustomerRegion", "RetailChannel", "Revenue by region and channel"))
    left, right = st.columns(2)
    with left:
        show(chart_low_sat_share(df, "CustomerRegion", "Low-satisfaction share by region"))
    with right:
        show(chart_low_sat_share(df, "RetailChannel", "Low-satisfaction share by channel"))


def render_data_table(df: pd.DataFrame) -> None:
    query = st.text_input("Search the table", placeholder="e.g. Decline, West, Electronics, 12350")
    table = df[["idx", "TransactionDate", "CustomerID", "TransactionID", "Segment", "CategoryGroup",
                "ProductCategory", "PurchaseAmount", "CustomerSatisfaction", "AgeGroup",
                "CustomerGender", "CustomerRegion", "RetailChannel"]]
    if query:
        text = table.astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False))
        table = table[text.any(axis=1)]
    st.caption(f"{len(table):,} rows shown.")
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "idx": "Row",
            "TransactionDate": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
            "CustomerID": st.column_config.NumberColumn("Customer", format="%d"),
            "TransactionID": "Transaction ID (not unique)",
            "CategoryGroup": "Category group",
            "ProductCategory": "Original category",
            "PurchaseAmount": st.column_config.NumberColumn("Amount", format="dollar"),
            "CustomerSatisfaction": "Satisfaction",
            "AgeGroup": "Age group",
            "CustomerGender": "Gender",
            "CustomerRegion": "Region",
            "RetailChannel": "Channel",
        },
    )
    st.download_button(
        "Download filtered data (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"novaretail_filtered_{date.today():%Y%m%d}.csv",
        mime="text/csv",
    )


def render_about(raw: pd.DataFrame) -> None:
    st.subheader("Variables used")
    st.dataframe(pd.DataFrame([
        ("label", "Used", "Segment (Promising, Growth, Stable, Decline); core of every view."),
        ("CustomerID", "Used", "Counts unique customers."),
        ("TransactionDate", "Used", "Date filter and weekly trend."),
        ("ProductCategory", "Used (grouped)", "35 raw names grouped into 9 categories."),
        ("PurchaseAmount", "Used", "Revenue in USD."),
        ("CustomerSatisfaction", "Used", "1–5 rating; main early-warning signal."),
        ("CustomerRegion", "Used", "Regional performance."),
        ("RetailChannel", "Used", "Online vs. Physical Store."),
        ("CustomerAgeGroup", "Used", "Demographic filter ('55-64' merged into '55+')."),
        ("CustomerGender", "Used", "Demographic filter."),
        ("idx", "Row key only", "Unique row number, used for counting lines."),
        ("TransactionID", "Reference only", "Repeats across rows, dates and customers, so it cannot count transactions."),
    ], columns=["Variable", "Role", "Reason"]), hide_index=True, width="stretch", height=35 * 13 + 3)

    st.subheader("How to read the numbers")
    st.markdown(
        "- Each row is a **purchase line**. Customers are counted as unique CustomerIDs.\n"
        "- Segment labels are attached to purchase lines; some customers appear under several "
        "segments, so segment figures describe lines, not people.\n"
        f"- Figures resting on fewer than {MIN_RELIABLE_N} lines are flagged as small samples.\n"
        "- The data covers about five weeks of early 2023, so trends are directional.\n"
        "- The AI assistant never sees the raw rows. It receives a summary of the filtered data "
        "(the KPI values and the segment, region, channel, category and weekly tables) computed "
        "with the same functions as this dashboard."
    )
    with st.expander("Data-validation checks", expanded=False):
        st.dataframe(run_validation_checks(raw), hide_index=True, width="stretch")
        st.caption("Category mapping used:")
        st.dataframe(pd.DataFrame(
            [(g, ", ".join(r)) for g, r in CATEGORY_GROUPS.items()],
            columns=["Category group", "Original categories"]), hide_index=True, width="stretch")


# =============================================================================
# 8. AI DATA ASSISTANT (CHATBOT)
# =============================================================================

class AllModelsUnavailable(Exception):
    """Raised when every configured model is retired or unknown."""


def get_secret(name: str):
    """Read one secret without crashing when no secrets file exists."""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def candidate_models() -> list:
    """Configured model first (LLM_MODEL secret overrides), then fallbacks, no repeats."""
    first = get_secret("LLM_MODEL") or DEFAULT_MODEL
    return list(dict.fromkeys([first] + FALLBACK_MODELS))


def is_model_unavailable_error(err: Exception) -> bool:
    """True only for 'this model is retired / unknown' errors."""
    message = str(err).lower()
    return any(phrase in message for phrase in MODEL_UNAVAILABLE_PHRASES)


def table_text(df: pd.DataFrame, column: str, label: str, sort_by_revenue: bool = True) -> str:
    """Compact pipe-separated table for one dimension."""
    data = summarize_by(df, column)
    if sort_by_revenue:
        data = data.sort_values("Revenue", ascending=False)
    lines = [f"{label} | Revenue | Share | Lines | Customers | Avg purchase | Avg satisfaction | Lines rated <=2"]
    for _, r in data.iterrows():
        lines.append(
            f"{r[column]} | {fmt_money(r['Revenue'])} | {r['RevenueShare']:.1%} | {int(r['Lines']):,} | "
            f"{int(r['Customers']):,} | {fmt_money(r['AvgPurchase'])} | {r['AvgSatisfaction']:.2f} | "
            f"{r['LowSatShare']:.1%}"
        )
    return "\n".join(lines)


def build_data_context(df: pd.DataFrame, active_filters: str) -> str:
    """Text summary of the filtered data sent to the model (no raw rows).
    Uses the same functions as the KPI cards and charts, so numbers match the screen."""
    k = compute_kpis(df)
    parts = [
        "ACTIVE FILTERS: " + active_filters.replace("  \n", "; "),
        "",
        "KPIs (same as the dashboard cards):",
        f"Total revenue {fmt_money(k['revenue'])} | Purchase lines {k['lines']:,} | "
        f"Unique customers {k['customers']:,} | Average purchase {fmt_money(k['avg_purchase'])} | "
        f"Average satisfaction {k['avg_satisfaction']:.2f}/5 | "
        f"Revenue from Decline lines {k['decline_rev_share']:.1%} | "
        f"Lines rated <=2 {k['low_sat_share']:.1%}",
        "",
        "BY SEGMENT:",
        table_text(df, "Segment", "Segment", sort_by_revenue=False),
        "",
        "BY REGION:",
        table_text(df, "CustomerRegion", "Region"),
        "",
        "BY CHANNEL:",
        table_text(df, "RetailChannel", "Channel"),
        "",
        "BY CATEGORY GROUP:",
        table_text(df, "CategoryGroup", "Category group"),
        "",
        "REVENUE BY SEGMENT AND CHANNEL:",
    ]
    seg_ch = df.pivot_table(index="Segment", columns="RetailChannel", values="PurchaseAmount",
                            aggfunc="sum", fill_value=0, observed=True)
    seg_ch = seg_ch.reindex([s for s in SEGMENT_ORDER if s in seg_ch.index])
    for seg, row in seg_ch.iterrows():
        parts.append(f"{seg}: " + ", ".join(f"{ch} {fmt_money(v)}" for ch, v in row.items()))

    parts += ["", "WEEKLY REVENUE (week starting Monday):"]
    weekly = df.groupby("WeekStart").agg(Revenue=("PurchaseAmount", "sum"), Lines=("idx", "count"))
    for week, row in weekly.iterrows():
        parts.append(f"{week:%b %d, %Y}: {fmt_money(row['Revenue'])} from {int(row['Lines'])} lines")

    parts += ["", f"FIVE HIGHEST-VALUE LINES RATED <= {LOW_SAT_THRESHOLD}:"]
    low = df[df["LowSatisfaction"]].nlargest(5, "PurchaseAmount")
    if low.empty:
        parts.append("None in the current selection.")
    for _, r in low.iterrows():
        parts.append(
            f"Customer {r['CustomerID']} | {r['TransactionDate']:%b %d, %Y} | {r['Segment']} | "
            f"{r['CategoryGroup']} | {fmt_money(r['PurchaseAmount'])} | satisfaction {r['CustomerSatisfaction']} | "
            f"{r['CustomerRegion']} | {r['RetailChannel']}"
        )

    multi = int((df.groupby("CustomerID")["Segment"].nunique() > 1).sum())
    parts += [
        "",
        "DATA NOTES:",
        f"- Sample: {len(df):,} purchase lines from {df['CustomerID'].nunique():,} customers, "
        f"{df['TransactionDate'].min():%b %d, %Y} to {df['TransactionDate'].max():%b %d, %Y}.",
        "- Each row is a purchase line, not a customer.",
        f"- {multi} customers in this selection appear under more than one segment label.",
        f"- Unlabeled lines in this selection: {int((df['Segment'] == UNLABELED).sum())}.",
    ]
    return "\n".join(parts)


def stream_text(response):
    """Yield only the answer text from a streamed response."""
    for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


def ask_model(client: OpenAI, messages: list) -> str:
    """Stream an answer into the page, falling back to other models if one is retired."""
    for model in candidate_models():
        kwargs = dict(model=model, messages=messages, temperature=0.2, max_tokens=2000, stream=True)
        if "gpt-oss" in model:
            kwargs["reasoning_effort"] = "low"
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as err:
            if is_model_unavailable_error(err):
                continue
            raise
        st.session_state["active_model"] = model
        answer = st.write_stream(stream_text(response))
        return answer if isinstance(answer, str) else "".join(map(str, answer or []))
    raise AllModelsUnavailable()


def friendly_error(err: Exception) -> str:
    if isinstance(err, openai.AuthenticationError):
        return "The API key was rejected. Please check the key saved in your Streamlit Secrets."
    if isinstance(err, openai.RateLimitError):
        return "The free-tier rate limit was reached. Please wait a minute and try again."
    if isinstance(err, AllModelsUnavailable):
        return ("None of the configured AI models are available. Add an LLM_MODEL secret with a "
                "model currently listed in the Groq console.")
    return f"Sorry, something went wrong contacting the AI service: {err}"


def render_chatbot(filtered_df: pd.DataFrame, active_filters: str) -> None:
    st.subheader("Ask the Data Assistant")
    model_shown = st.session_state.get("active_model", candidate_models()[0])
    st.caption(
        f"Powered by {PROVIDER_NAME} · model {model_shown}. Answers are based on the data matching "
        "your current sidebar filters. AI can make mistakes; verify key numbers with the KPI cards and charts."
    )
    st.session_state.setdefault("messages", [])
    context = build_data_context(filtered_df, active_filters)

    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        st.warning(
            "The AI assistant is not set up yet. Add your Groq key in Streamlit Cloud: "
            "**Manage app → ⋮ → Settings → Secrets**, then paste `GROQ_API_KEY = \"gsk_...\"` and save. "
            "The rest of the dashboard works without it.",
            icon="🔑",
        )

    pending = None
    cols = st.columns(len(SUGGESTED_QUESTIONS) + 1)
    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        if col.button(question, width="stretch", disabled=not api_key, help=question):
            pending = question
    if cols[-1].button("🗑️ Clear chat", width="stretch"):
        st.session_state.messages = []

    with st.expander("Data the assistant can see"):
        st.caption("This exact summary is sent with every question. Compare it with the KPI cards and charts.")
        st.code(context, language=None, wrap_lines=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask a question about NovaRetail customers…", disabled=not api_key)
    question = typed or pending
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            history = st.session_state.messages[-HISTORY_LENGTH:]
            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}] + history
            answer = ask_model(get_client(api_key), messages)
            if not answer or not answer.strip():
                answer = "The model returned an empty answer. Please try rephrasing your question."
                st.markdown(answer)
        except Exception as err:
            answer = friendly_error(err)
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


# =============================================================================
# 9. MAIN APPLICATION
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="NovaRetail Customer Intelligence", page_icon="📊", layout="wide")

    if not DATA_FILE.exists():
        st.error(f"Dataset not found: {DATA_FILE.name}. Upload it to the repository root next to app.py.")
        st.stop()

    raw = load_raw_data(DATA_FILE)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        st.error(f"The dataset is missing required columns: {missing}")
        st.stop()
    df = clean_data(raw)

    render_header(df)
    selections, filter_summary = render_sidebar(df)
    filtered = apply_filters(df, selections)

    if filtered.empty:
        st.warning("No purchase lines match the current filters. Widen a filter or press **Reset filters**.",
                   icon="⚠️")
        st.stop()

    render_kpis(filtered)

    tabs = st.tabs(TAB_NAMES)
    with tabs[0]:
        render_overview(filtered)
    with tabs[1]:
        render_segments(filtered)
    with tabs[2]:
        render_early_warning(filtered)
    with tabs[3]:
        render_region_category_channel(filtered)
    with tabs[4]:
        render_chatbot(filtered, filter_summary)
    with tabs[5]:
        render_data_table(filtered)
    with tabs[6]:
        render_about(raw)


if __name__ == "__main__":
    main()
