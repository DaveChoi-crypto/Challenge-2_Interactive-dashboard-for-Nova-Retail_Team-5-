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

import html
import re
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
MORE_FILTER_COLUMNS = {"AgeGroup", "CustomerGender"}   # shown inside a "More filters" expander

# Satisfaction scores 1-5: low = red/orange, neutral = gray, high = green (status meaning).
SCORE_COLORS = {1: "#d03b3b", 2: "#ec835a", 3: "#8c8b86", 4: "#5cb85c", 5: "#0ca30c"}
RISK_COLOR = SEGMENT_COLORS["Decline"]
NEUTRAL_FILL = "rgba(128, 128, 128, 0.28)"
ACCENT_COLOR = "#5c6b8a"     # UI accent; also set in .streamlit/config.toml
GOAL_COLORS = {"growth": SEGMENT_COLORS["Growth"], "warning": SEGMENT_COLORS["Decline"], "action": ACCENT_COLOR}


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

    def multiselect_filter(container, label, column, key):
        options = defaults[key]
        select_all = container.checkbox(f"All {label.lower()}s", key=f"{key}_all")
        chosen = container.multiselect(label, options, key=key, disabled=select_all)
        selections[column] = options if select_all else chosen

    # Main business dimensions first; demographic and score filters are folded away.
    for label, column, key in MULTISELECT_FILTERS:
        if column not in MORE_FILTER_COLUMNS:
            multiselect_filter(sb, label, column, key)

    min_d, max_d = defaults["f_dates"]
    picked = sb.date_input("Date range", key="f_dates", min_value=min_d, max_value=max_d)
    # While the user is mid-selection, date_input can return a single date.
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
    else:
        start = picked[0] if isinstance(picked, (tuple, list)) and picked else min_d
        end = max_d
    selections["dates"] = (start, end)

    with sb.expander("More filters (age, gender, satisfaction)"):
        for label, column, key in MULTISELECT_FILTERS:
            if column in MORE_FILTER_COLUMNS:
                multiselect_filter(st, label, column, key)
        selections["satisfaction"] = st.slider("Satisfaction score", 1, 5, key="f_sat")

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
    lines_per_customer = df.groupby("CustomerID").size()
    repeat = lines_per_customer[lines_per_customer > 1]
    return {
        "revenue": revenue,
        "lines": len(df),
        "customers": df["CustomerID"].nunique(),
        "repeat_customers": len(repeat),
        "repeat_customer_share": len(repeat) / len(lines_per_customer) if len(lines_per_customer) else 0.0,
        "repeat_line_share": repeat.sum() / len(df) if len(df) else 0.0,
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


def build_goal_insights(df: pd.DataFrame) -> list:
    """Three insight cards, one per business goal, computed from the filtered data.
    Returns (goal key, title, text) with **bold** markers. Never hard-coded."""
    total = df["PurchaseAmount"].sum()
    k = compute_kpis(df)
    cards = []

    # a) Growth opportunity
    core = df[df["Segment"].isin(["Growth", "Promising"])]
    if not core.empty and total:
        text = (f"**Growth + Promising** bring **{core['PurchaseAmount'].sum() / total:.1%}** of revenue "
                f"at **{core['CustomerSatisfaction'].mean():.2f}/5** satisfaction. "
                f"**{k['repeat_customers']} repeat buyers** ({k['repeat_customer_share']:.0%} of customers) "
                f"make **{k['repeat_line_share']:.0%}** of purchases{small_n_note(len(core))}.")
    else:
        seg = summarize_by(df, "Segment")
        top = seg.loc[seg["Revenue"].idxmax()]
        text = (f"No Growth or Promising lines in this selection. **{top['Segment']}** leads with "
                f"**{fmt_money(top['Revenue'])}**{small_n_note(int(top['Lines']))}.")
    cards.append(("growth", "Growth opportunity", text))

    # b) Early warning
    decline = df[df["Segment"] == "Decline"]
    if not decline.empty and total:
        text = (f"**Decline** lines hold **{decline['PurchaseAmount'].sum() / total:.1%}** of revenue "
                f"({fmt_money(decline['PurchaseAmount'].sum())}) but average "
                f"**{decline['CustomerSatisfaction'].mean():.2f}/5**; "
                f"{decline['LowSatisfaction'].mean():.0%} rate {LOW_SAT_THRESHOLD} or lower. "
                f"Last Decline purchase: **{decline['TransactionDate'].max():%b %d}**"
                f"{small_n_note(len(decline))}.")
    else:
        seg = summarize_by(df, "Segment")
        low = seg.loc[seg["AvgSatisfaction"].idxmin()]
        text = (f"No Decline lines in this selection. Lowest satisfaction: **{low['Segment']}** "
                f"at **{low['AvgSatisfaction']:.2f}/5**{small_n_note(int(low['Lines']))}.")
    cards.append(("warning", "Early warning", text))

    # c) Recommended action
    if not decline.empty:
        spots = (decline.groupby(["CustomerRegion", "RetailChannel"])
                 .agg(Revenue=("PurchaseAmount", "sum"), Customers=("CustomerID", "nunique"))
                 .sort_values("Revenue", ascending=False))
        (region, channel), top = next(spots.iterrows())
        text = (f"Re-engage Decline customers in **{region} · {channel}** first: "
                f"**{fmt_money(top['Revenue'])}** at risk from {int(top['Customers'])} customers. "
                "They spend well but are unhappy, so service recovery should come before promotions.")
    else:
        cat = summarize_by(df, "CategoryGroup").sort_values("Revenue", ascending=False).iloc[0]
        text = (f"Grow **{cat['CategoryGroup']}**, the top category in this selection "
                f"(**{fmt_money(cat['Revenue'])}**), with targeted offers to satisfied repeat buyers.")
    cards.append(("action", "Recommended action", text))
    return cards


# =============================================================================
# 6. CHARTS
# =============================================================================

def cap_bar_width(fig: go.Figure) -> None:
    """Keep bars slim when a filter leaves only one to three categories,
    so a single bar never fills the whole chart."""
    bars = [t for t in fig.data if t.type == "bar"]
    if not bars:
        return
    horizontal = bars[0].orientation == "h"
    values = pd.concat([pd.Series(t.y if horizontal else t.x) for t in bars], ignore_index=True)
    if values.empty or pd.api.types.is_datetime64_any_dtype(values):
        return   # date axes (weekly bars) set their own width
    widths = {1: 0.25, 2: 0.4, 3: 0.5}
    n = values.nunique()
    if n in widths:
        fig.update_traces(width=widths[n], selector=dict(type="bar"))


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
        hoverlabel=dict(font_size=13),
        bargap=0.35,
        barcornerradius=4,          # rounded data-ends on every bar
    )
    cap_bar_width(fig)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    if value_axis == "y":
        fig.update_yaxes(rangemode="tozero")
    elif value_axis == "x":
        fig.update_xaxes(rangemode="tozero", showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(showgrid=False)
    return fig


def show(fig: go.Figure) -> None:
    # Hover tooltips stay; the toolbar (zoom, camera...) is hidden to reduce noise.
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# --- KPI mini charts: one form per KPI, chosen for the question each KPI answers ---

def mini_fig(fig: go.Figure, height: int = 64) -> go.Figure:
    """Bare styling for charts inside KPI cards: no axes, no legend, no margins."""
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      barmode="stack", bargap=0.15, barcornerradius=3,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font_size=12), uniformtext=dict(minsize=9, mode="hide"))
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    return fig


def share_bar(parts: list, fmt: str = ".1%") -> go.Figure:
    """Single 100% horizontal bar. parts = [(label, value, color, hover_text)].
    Labels use the same rounding as the headline numbers (one decimal for revenue shares)."""
    total = sum(v for _, v, _, _ in parts) or 1
    fig = go.Figure()
    for label, value, color, hover in parts:
        share = value / total
        fig.add_trace(go.Bar(
            x=[share], y=[""], orientation="h", marker_color=color, name=label,
            text=f"{label} {share:{fmt}}" if share >= 0.14 else "", textposition="inside",
            insidetextanchor="middle",
            # Labels on the neutral part use a mid gray that reads on light and dark themes.
            textfont=dict(size=11, color="rgb(128,128,128)" if color == NEUTRAL_FILL else None),
            hovertemplate=f"{hover}<extra></extra>",
        ))
    fig.update_xaxes(range=[0, 1])
    fig = mini_fig(fig, height=56)
    fig.update_layout(bargap=0.42)   # slim bar, same card height as the score chart
    return fig


def mini_revenue_mix(df: pd.DataFrame) -> go.Figure:
    seg = summarize_by(df, "Segment")
    return share_bar([
        (r["Segment"], r["Revenue"], SEGMENT_COLORS[r["Segment"]],
         f"{r['Segment']}: {fmt_money(r['Revenue'])} ({r['RevenueShare']:.1%})")
        for _, r in seg.iterrows()
    ])


def mini_repeat_buyers(k: dict) -> go.Figure:
    one_time = k["customers"] - k["repeat_customers"]
    return share_bar([
        ("Repeat", k["repeat_customers"], ACCENT_COLOR,
         f"Repeat buyers: {k['repeat_customers']} customers, {k['repeat_line_share']:.0%} of purchases"),
        ("One-time", one_time, NEUTRAL_FILL, f"One-time buyers: {one_time} customers"),
    ], fmt=".0%")   # whole percent, matching the insight card


def mini_score_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["CustomerSatisfaction"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    shares = counts / counts.sum() if counts.sum() else counts
    fig = go.Figure(go.Bar(
        x=[str(s) for s in counts.index], y=shares.values,
        marker_color=[SCORE_COLORS[s] for s in counts.index],
        text=[f"{s}★ {v:.0%}" for s, v in shares.items()], textposition="outside",
        textfont=dict(size=10), cliponaxis=False, customdata=counts.values,
        hovertemplate="Score %{x}: %{y:.0%} of lines (%{customdata})<extra></extra>",
    ))
    fig.update_yaxes(range=[0, max(0.01, shares.max()) * 1.45])
    return mini_fig(fig, height=56)


def mini_risk_bar(k: dict) -> go.Figure:
    at_risk = k["revenue"] * k["decline_rev_share"]
    return share_bar([
        ("At risk", at_risk, RISK_COLOR, f"Decline revenue: {fmt_money(at_risk)}"),
        ("Other", k["revenue"] - at_risk, NEUTRAL_FILL, f"Other revenue: {fmt_money(k['revenue'] - at_risk)}"),
    ])


def show_mini(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "staticPlot": False})


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


def chart_segment_map(df: pd.DataFrame) -> go.Figure:
    """Quadrant bubble chart: average purchase vs. satisfaction, bubble size = revenue.
    Dashed lines mark the averages of the current selection."""
    seg = summarize_by(df, "Segment")
    seg = seg[seg["Segment"] != UNLABELED]   # one unlabeled line would only add noise here
    if seg.empty:
        return None
    avg_sat, avg_buy = df["CustomerSatisfaction"].mean(), df["PurchaseAmount"].mean()
    y_max = max(seg["AvgPurchase"].max(), avg_buy) * 1.5
    fig = px.scatter(seg, x="AvgSatisfaction", y="AvgPurchase", size="Revenue", color="Segment",
                     color_discrete_map=SEGMENT_COLORS, text="Segment", custom_data=SEGMENT_CUSTOM,
                     size_max=42, title="Segment map: average purchase vs. satisfaction (bubble = revenue)")
    fig.update_traces(
        textposition="top center", cliponaxis=False,
        marker=dict(line=dict(color="rgba(128,128,128,0.6)", width=1), opacity=0.85),
        hovertemplate=SEGMENT_HOVER.replace("%{x}", "%{text}"),
    )
    # Growth and Promising sit close together; label one above and one below.
    fig.update_traces(textposition="bottom center", selector=dict(name="Promising"))
    guide = dict(line_dash="dash", line_color="rgba(128,128,128,0.6)", line_width=1,
                 annotation_font=dict(size=10, color="rgba(128,128,128,0.95)"))
    fig.add_vline(x=avg_sat, annotation_text=f"avg {avg_sat:.2f}", annotation_position="top left", **guide)
    fig.add_hline(y=avg_buy, annotation_text=f"avg ${avg_buy:,.0f}", annotation_position="top left", **guide)
    corners = [
        (0.85, y_max * 0.97, "left", "Protect: high value, unhappy"),
        (5.15, y_max * 0.97, "right", "Grow: high value, satisfied"),
        (0.85, y_max * 0.03, "left", "Watch: low value, unhappy"),
        (5.15, y_max * 0.03, "right", "Develop: low value, satisfied"),
    ]
    for x, y, align, label in corners:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, xanchor=align,
                           font=dict(size=11, color="rgba(128,128,128,0.95)"))
    fig.update_layout(showlegend=False, xaxis_title="Average satisfaction (1–5)",
                      yaxis_title="Average purchase (USD)")
    fig.update_xaxes(range=[0.8, 5.2], dtick=1, showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(range=[0, y_max], tickprefix="$", tickformat=",.0f")
    return style_fig(fig, value_axis=None)


def partial_weeks(start: date, end: date) -> set:
    """Week-start dates whose week is only partly inside the selected date range."""
    days = pd.date_range(start, end)
    counts = pd.Series(1, index=days).groupby(days.to_period("W-SUN").start_time).sum()
    return set(counts[counts < 7].index)


def chart_weekly_trend(df: pd.DataFrame, start: date, end: date) -> go.Figure:
    """Weekly revenue as stacked bars by segment: total and mix in one view.
    Weeks only partly inside the date range are drawn lighter and marked with *."""
    weekly = (df.groupby(["WeekStart", "Segment"], observed=True)
                .agg(Revenue=("PurchaseAmount", "sum"), Lines=("idx", "count")).reset_index())
    partial = partial_weeks(start, end)
    weekly["Partial"] = weekly["WeekStart"].isin(partial)
    fig = px.bar(weekly, x="WeekStart", y="Revenue", color="Segment",
                 color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENT_ORDER},
                 custom_data=["Segment", "Lines", "Partial"], title="Weekly revenue by segment")
    week_ms = 7 * 24 * 3600 * 1000
    fig.update_traces(width=week_ms * 0.6, marker_line_color="rgba(128,128,128,0.35)", marker_line_width=1,
                      hovertemplate="<b>%{customdata[0]}</b><br>Week of %{x|%b %d}"
                                    "<br>Revenue: $%{y:,.2f}<br>Lines: %{customdata[1]}<extra></extra>")
    for trace in fig.data:   # lighter bars for partial weeks
        trace.marker.opacity = [0.4 if pd.Timestamp(x) in partial else 1.0 for x in trace.x]
    weeks = sorted(weekly["WeekStart"].unique())
    labels = [f"{pd.Timestamp(w):%b %d}" + ("*" if pd.Timestamp(w) in partial else "") for w in weeks]
    fig.update_layout(barmode="stack", xaxis_title="Week starting (* partial week: fewer than 7 days of data)",
                      yaxis_title="Revenue (USD)")
    fig.update_xaxes(tickvals=weeks, ticktext=labels)
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
        showscale=False,   # every cell is labeled, so the color bar is redundant
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


def decline_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Decline vs. all other lines on three measures, as a compact table
    (three different scales, so a table reads better than three charts)."""
    is_decline = df["Segment"].eq("Decline")
    rows = []
    for name, part in [("Decline", df[is_decline]), ("All other", df[~is_decline])]:
        if part.empty:
            rows.append((name, "–", "–", "–", "0"))
            continue
        rows.append((name, f"{part['CustomerSatisfaction'].mean():.2f} / 5",
                     fmt_money(part["PurchaseAmount"].mean()),
                     f"{part['LowSatisfaction'].mean():.0%}", f"{len(part):,}"))
    return pd.DataFrame(rows, columns=["Group", "Avg satisfaction", "Avg purchase",
                                       f"Lines rated ≤ {LOW_SAT_THRESHOLD}", "Purchase lines"])


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
        showscale=False,
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
        "Which customers drive revenue, which segments show early signs of decline, and where "
        "to invest: explore by segment, region, product category and channel."
    )
    st.caption(
        f"ℹ️ Small sample: {len(df):,} purchase lines from {df['CustomerID'].nunique():,} customers, "
        f"{df['TransactionDate'].min():%b %d} – {df['TransactionDate'].max():%b %d, %Y}. "
        "Amounts are revenue in USD; each row is a purchase line, not a customer. "
        "Source: NR_dataset.xlsx (course-provided)."
    )


def render_kpis(df: pd.DataFrame, full: pd.DataFrame) -> None:
    """Executive row: four KPI cards. Each card pairs the number with a small chart chosen
    for the question that KPI answers, plus a comparison with all data when filters are active."""
    k, base = compute_kpis(df), compute_kpis(full)
    filtered = len(df) < len(full)
    cols = st.columns(4)

    with cols[0].container(border=True):
        st.metric("Total revenue", fmt_money(k["revenue"]),
                  delta=f"{k['revenue'] / base['revenue']:.0%} of all revenue" if filtered else None,
                  delta_color="off", delta_arrow="off",
                  help="Sum of PurchaseAmount (USD). Bar: revenue mix by segment.")
        show_mini(mini_revenue_mix(df))
        st.caption("Revenue mix by segment")

    with cols[1].container(border=True):
        st.metric("Unique customers", f"{k['customers']:,}",
                  delta=f"of {base['customers']:,} customers" if filtered else None,
                  delta_color="off", delta_arrow="off",
                  help="Distinct CustomerIDs. Repeat buyers have more than one purchase line.")
        show_mini(mini_repeat_buyers(k))
        st.caption(f"{k['repeat_customers']} repeat · {k['repeat_line_share']:.0%} of purchases")

    with cols[2].container(border=True):
        st.metric("Average satisfaction", f"{k['avg_satisfaction']:.2f} / 5",
                  delta=f"{k['avg_satisfaction'] - base['avg_satisfaction']:+.2f}" if filtered else None,
                  delta_description="vs all data" if filtered else None,
                  help="Mean rating, 1 (lowest) to 5 (highest). Bars: share of lines at each score.")
        show_mini(mini_score_distribution(df))
        st.caption("Share of lines by score (1–5)")

    with cols[3].container(border=True):
        st.metric("Revenue at risk (Decline)", f"{k['decline_rev_share']:.1%}",
                  delta=f"{(k['decline_rev_share'] - base['decline_rev_share']) * 100:+.1f} pts" if filtered else None,
                  delta_description="vs all data" if filtered else None, delta_color="inverse",
                  help="Share of revenue from Decline-labeled lines.")
        show_mini(mini_risk_bar(k))
        st.caption(f"{fmt_money(k['revenue'] * k['decline_rev_share'])} from Decline lines")

    if k["lines"] < MIN_RELIABLE_N:
        noun = "purchase line matches" if k["lines"] == 1 else "purchase lines match"
        st.warning(f"Only {k['lines']} {noun}. Treat figures with caution.", icon="⚠️")


def insight_card_html(goal: str, title: str, text: str) -> str:
    """Callout card with a colored left edge. The title names the goal, so color is never the only cue."""
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))
    color = GOAL_COLORS[goal]
    return (
        f'<div style="border:1px solid rgba(128,128,128,0.3); border-left:5px solid {color}; '
        f'border-radius:8px; padding:12px 16px; min-height:168px; line-height:1.5;">'
        f'<div style="font-size:0.78rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; '
        f'color:{color}; margin-bottom:6px;">{title}</div>'
        f'<div style="font-size:0.95rem;">{body}</div></div>'
    )


def render_overview(df: pd.DataFrame, start: date, end: date) -> None:
    cards = build_goal_insights(df)
    cols = st.columns(len(cards))
    for col, (goal, title, text) in zip(cols, cards):
        col.markdown(insight_card_html(goal, title, text), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left:
        show(chart_revenue_by_segment(df))
    with right:
        fig = chart_segment_map(df)
        if fig is None:
            st.info("The segment map needs at least one labeled segment in the selection.")
        else:
            show(fig)
    show(chart_weekly_trend(df, start, end))
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
    st.dataframe(decline_comparison_table(df), hide_index=True, width="stretch")
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
    show(chart_low_sat_share(df, "CustomerRegion", "Low-satisfaction share by region"))
    by_channel = summarize_by(df, "RetailChannel")
    st.caption("Low-satisfaction share by channel: " + ", ".join(
        f"{r['RetailChannel']} {r['LowSatShare']:.0%} ({int(r['Lines'])} lines)" for _, r in by_channel.iterrows()))


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
        "KPI CARDS (same as the dashboard):",
        f"Total revenue {fmt_money(k['revenue'])} | Unique customers {k['customers']:,} | "
        f"Average satisfaction {k['avg_satisfaction']:.2f}/5 | "
        f"Revenue at risk (Decline share of revenue) {k['decline_rev_share']:.1%}",
        "OTHER TOTALS:",
        f"Purchase lines {k['lines']:,} | Average purchase {fmt_money(k['avg_purchase'])} | "
        f"Lines rated <=2 {k['low_sat_share']:.1%}",
        f"Repeat buyers (more than one purchase line): {k['repeat_customers']} of {k['customers']} customers "
        f"({k['repeat_customer_share']:.0%}), making {k['repeat_line_share']:.0%} of purchase lines",
        "Satisfaction score distribution (share of lines): " + ", ".join(
            f"{s}: {v:.0%}" for s, v in df["CustomerSatisfaction"].value_counts(normalize=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0).items()),
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

    render_kpis(filtered, df)

    tabs = st.tabs(TAB_NAMES)
    with tabs[0]:
        render_overview(filtered, *selections["dates"])
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
