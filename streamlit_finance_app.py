"""
╔══════════════════════════════════════════════════════════════════╗
║  Equity Research Dashboard — Senior Financial Data Engineer     ║
║  Streamlit + yfinance + SEC EDGAR + Plotly                      ║
║  Dark Mode | Graduate Finance Curriculum                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import re
import html as html_mod
from datetime import datetime, timedelta
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & DARK THEME
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Equity Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG = "#0E1117"
CARD_BG = "#161B22"
ACCENT  = "#4F98A3"
ACCENT_HOVER = "#227F8B"
TEXT    = "#CDCCCA"
TEXT_MUTED = "#797876"
BORDER  = "#30363D"
RED     = "#DD6974"
GREEN   = "#6DAA45"
GOLD    = "#FFC553"
BLUE    = "#5591C7"
PURPLE  = "#A86FDF"
TERRA   = "#A84B2F"

# Chart color sequence
CHART_COLORS = ["#4F98A3", "#A84B2F", "#5591C7", "#FFC553", "#DD6974", "#A86FDF", "#6DAA45", "#848456"]

st.markdown(f"""
<style>
    /* ── Global ── */
    .stApp {{
        background-color: {DARK_BG};
        color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {TEXT};
    }}

    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 16px 20px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        font-variant-numeric: tabular-nums;
    }}
    [data-testid="stMetricDelta"] > div {{
        font-variant-numeric: tabular-nums;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 6px 6px 0 0;
        color: {TEXT_MUTED};
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT} !important;
        color: #fff !important;
        border-color: {ACCENT} !important;
    }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        color: {TEXT};
    }}
    .streamlit-expanderContent {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-top: none;
        border-radius: 0 0 6px 6px;
    }}

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 6px;
    }}

    /* ── Section dividers ── */
    hr {{
        border-color: {BORDER};
    }}

    /* ── Custom card class ── */
    .dashboard-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 12px;
    }}
    .card-title {{
        color: {ACCENT};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
        font-weight: 600;
    }}
    .hero-value {{
        color: {TEXT};
        font-size: 2.2rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }}
    .section-header {{
        color: {ACCENT};
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding-bottom: 6px;
        border-bottom: 2px solid {ACCENT};
        margin-bottom: 16px;
    }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# 2. PLOTLY DARK TEMPLATE
# ──────────────────────────────────────────────────────────────────

# Base layout — does NOT include xaxis/yaxis/legend to avoid conflicts
# with subplots and charts that override those keys.
PLOTLY_BASE = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, -apple-system, sans-serif", size=12),
    title_font=dict(size=14, color=TEXT),
    margin=dict(l=20, r=20, t=50, b=20),
    hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT, bordercolor=BORDER),
)

# Full layout for simple single-axis charts
PLOTLY_LAYOUT = dict(
    **PLOTLY_BASE,
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED, size=11)),
)


# ──────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def fmt_large_number(val: float) -> str:
    """Format large numbers with B/M/K suffixes."""
    if pd.isna(val) or val is None:
        return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val/1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val/1e3:.1f}K"
    else:
        return f"{sign}${val:.2f}"


def fmt_ratio(val: float, suffix: str = "x", decimals: int = 2) -> str:
    """Format a ratio value."""
    if pd.isna(val) or val is None:
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def fmt_pct(val: float, decimals: int = 1) -> str:
    """Format a percentage."""
    if pd.isna(val) or val is None:
        return "N/A"
    return f"{val:.{decimals}f}%"


def safe_div(numerator, denominator) -> Optional[float]:
    """Safe division returning None on error."""
    try:
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def safe_get(df: pd.DataFrame, row_labels: list, col_idx: int = 0):
    """Try multiple row label variants to find data in a financial statement."""
    if df is None or df.empty:
        return None
    for label in row_labels:
        if label in df.index:
            try:
                val = df.loc[label].iloc[col_idx]
                if pd.notna(val):
                    return float(val)
            except (IndexError, TypeError):
                continue
    return None


# ──────────────────────────────────────────────────────────────────
# 4. DATA FETCHING
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(ticker: str):
    """Fetch current price, market cap, and 1-year historical data."""
    t = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="1y")
    return info, hist


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_statements(ticker: str):
    """Fetch quarterly income statement, balance sheet, and cash flow."""
    t = yf.Ticker(ticker)
    inc_q = t.quarterly_income_stmt
    bs_q = t.quarterly_balance_sheet
    cf_q = t.quarterly_cashflow
    inc_a = t.income_stmt
    bs_a = t.balance_sheet
    return inc_q, bs_q, cf_q, inc_a, bs_a


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_sec_summary(ticker: str, company_name: str) -> dict:
    """
    Fetch Business Description and Risk Factors from the latest 10-K
    using the free SEC EDGAR EFTS API (no API key required).
    Falls back to yfinance info if SEC data is unavailable.
    """
    headers = {"User-Agent": "EquityResearchDashboard/1.0 (research@example.com)"}
    result = {"business_description": None, "risk_factors": None, "source": None, "filing_url": None}

    try:
        # Step 1: Get CIK from ticker
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(tickers_url, headers=headers, timeout=10)
        resp.raise_for_status()
        tickers_data = resp.json()

        cik = None
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break

        if not cik:
            raise ValueError(f"CIK not found for {ticker}")

        # Step 2: Get latest 10-K filing URL
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(submissions_url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])
        filing_dates = filings.get("filingDate", [])

        filing_url = None
        filing_date = None
        for i, form in enumerate(forms):
            if form == "10-K":
                accession = accession_numbers[i].replace("-", "")
                primary_doc = primary_docs[i]
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"
                filing_date = filing_dates[i]
                break

        if filing_url:
            result["filing_url"] = filing_url
            result["source"] = f"SEC EDGAR 10-K ({filing_date})"

            # Step 3: Fetch the 10-K document and extract sections
            try:
                doc_resp = requests.get(filing_url, headers=headers, timeout=20)
                doc_resp.raise_for_status()
                doc_text = html_mod.unescape(doc_resp.text)

                # Clean HTML tags for text extraction
                clean_text = re.sub(r'<[^>]+>', ' ', doc_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                # Extract Business Description (Item 1)
                # Find all matches and pick the one with substantial content
                biz_matches = list(re.finditer(
                    r'ITEM\s+1[.\s]*[\-\u2013\u2014]?\s*BUSINESS\s*(.+?)(?=ITEM\s+1\s*A)',
                    clean_text, re.IGNORECASE | re.DOTALL
                ))
                for m in biz_matches:
                    text = m.group(1).strip()
                    if len(text) > 200:  # Skip TOC entries
                        if len(text) > 2500:
                            text = text[:2500].rsplit('. ', 1)[0] + '.'
                        result["business_description"] = text
                        break

                # Extract Risk Factors (Item 1A)
                risk_matches = list(re.finditer(
                    r'Item\s+1\s*A[.\s]*[\-\u2013\u2014]?\s*Risk\s+Factors\s*(.+?)(?=Item\s+1\s*B|Item\s+1\s*C|Item\s+2\b)',
                    clean_text, re.IGNORECASE | re.DOTALL
                ))
                for m in risk_matches:
                    text = m.group(1).strip()
                    if len(text) > 200:
                        if len(text) > 2500:
                            text = text[:2500].rsplit('. ', 1)[0] + '.'
                        result["risk_factors"] = text
                        break

            except Exception:
                pass  # Fall through to yfinance fallback for content

    except Exception:
        pass

    # Fallback: use yfinance info for business summary
    if not result["business_description"]:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            result["business_description"] = info.get("longBusinessSummary", "No business description available.")
            if not result["source"]:
                result["source"] = "Yahoo Finance"
        except Exception:
            result["business_description"] = "Unable to retrieve business description."

    if not result["risk_factors"]:
        result["risk_factors"] = (
            "Risk factors could not be extracted automatically from the 10-K filing. "
            "Please review the full filing at SEC EDGAR for comprehensive risk disclosures."
        )

    return result


# ──────────────────────────────────────────────────────────────────
# 5. RATIO CALCULATIONS
# ──────────────────────────────────────────────────────────────────

def compute_quarterly_ratios(inc_q: pd.DataFrame, bs_q: pd.DataFrame, cf_q: pd.DataFrame, info: dict) -> pd.DataFrame:
    """
    Compute financial ratios for each of the last 4 quarters.
    Returns a DataFrame with quarters as rows and ratios as columns.
    """
    if inc_q is None or bs_q is None or inc_q.empty or bs_q.empty:
        return pd.DataFrame()

    # Use up to 4 most recent quarters
    n_quarters = min(4, len(inc_q.columns), len(bs_q.columns))
    if n_quarters == 0:
        return pd.DataFrame()

    records = []
    for i in range(n_quarters):
        q_date = inc_q.columns[i]
        q_label = pd.Timestamp(q_date).strftime("Q%q %Y") if hasattr(pd.Timestamp(q_date), 'quarter') else str(q_date)[:10]
        q_label = f"Q{(pd.Timestamp(q_date).month - 1) // 3 + 1} {pd.Timestamp(q_date).year}"

        # ── Income Statement items ──
        total_revenue = safe_get(inc_q, ["Total Revenue", "Operating Revenue", "TotalRevenue", "Revenue"], i)
        net_income = safe_get(inc_q, ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest", "NetIncome"], i)
        ebit = safe_get(inc_q, ["EBIT", "Operating Income", "Total Operating Income As Reported", "OperatingIncome"], i)
        interest_expense = safe_get(inc_q, ["Interest Expense", "Other Income Expense", "Other Non Operating Income Expenses", "InterestExpense"], i)
        ebitda = safe_get(inc_q, ["EBITDA", "Normalized EBITDA"], i)

        # ── Balance Sheet items ──
        bs_col = min(i, len(bs_q.columns) - 1)
        total_assets = safe_get(bs_q, ["Total Assets"], bs_col)
        total_liab = safe_get(bs_q, ["Total Liabilities Net Minority Interest"], bs_col)
        total_equity = safe_get(bs_q, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"], bs_col)
        current_assets = safe_get(bs_q, ["Current Assets"], bs_col)
        current_liab = safe_get(bs_q, ["Current Liabilities"], bs_col)
        inventory = safe_get(bs_q, ["Inventory", "Finished Goods"], bs_col)
        cash = safe_get(bs_q, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"], bs_col)
        total_debt = safe_get(bs_q, ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"], bs_col)

        # If total_liab not directly available, compute from assets - equity
        if total_liab is None and total_assets is not None and total_equity is not None:
            total_liab = total_assets - total_equity

        # ── Compute Ratios ──
        # Liquidity
        current_ratio = safe_div(current_assets, current_liab)
        quick_ratio = safe_div((current_assets - (inventory or 0)) if current_assets else None, current_liab)

        # Profitability
        roe = safe_div(net_income, total_equity) * 100 if safe_div(net_income, total_equity) is not None else None
        roa = safe_div(net_income, total_assets) * 100 if safe_div(net_income, total_assets) is not None else None
        net_margin = safe_div(net_income, total_revenue) * 100 if safe_div(net_income, total_revenue) is not None else None

        # Leverage
        debt_to_equity = safe_div(total_debt or total_liab, total_equity)
        interest_cov = safe_div(ebit, abs(interest_expense)) if interest_expense and interest_expense != 0 else None

        records.append({
            "Quarter": q_label,
            "Date": pd.Timestamp(q_date),
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "ROE (%)": roe,
            "ROA (%)": roa,
            "Net Margin (%)": net_margin,
            "Debt-to-Equity": debt_to_equity,
            "Interest Coverage": interest_cov,
            "Revenue": total_revenue,
            "Net Income": net_income,
            "Total Assets": total_assets,
            "Total Equity": total_equity,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_valuation_ratios(info: dict) -> dict:
    """Extract valuation ratios from yfinance info dict."""
    return {
        "P/E Ratio": info.get("trailingPE") or info.get("forwardPE"),
        "Forward P/E": info.get("forwardPE"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "Price-to-Book": info.get("priceToBook"),
        "EV/Revenue": info.get("enterpriseToRevenue"),
        "PEG Ratio": info.get("pegRatio"),
        "Enterprise Value": info.get("enterpriseValue"),
    }


# ──────────────────────────────────────────────────────────────────
# 6. CHART BUILDERS
# ──────────────────────────────────────────────────────────────────

def build_price_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """Build 1-year price chart with volume overlay."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist["Close"],
            mode="lines",
            name="Close",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(79, 152, 163, 0.08)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Volume bars
    colors = [GREEN if hist["Close"].iloc[i] >= hist["Open"].iloc[i] else RED
              for i in range(len(hist))]
    fig.add_trace(
        go.Bar(
            x=hist.index, y=hist["Volume"],
            marker_color=colors,
            opacity=0.5,
            name="Volume",
            hovertemplate="%{x|%b %d}: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=f"{ticker.upper()} — 1-Year Price & Volume", x=0.01),
        showlegend=False,
        height=440,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
        xaxis2=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED), tickprefix="$"),
        yaxis2=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def build_ratio_chart(df: pd.DataFrame, columns: list, title: str, y_suffix: str = "", y_format: str = ".2f") -> go.Figure:
    """Build a grouped bar chart for ratios over quarters."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title=title)
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED, size=14))
        return fig

    fig = go.Figure()
    for i, col in enumerate(columns):
        if col in df.columns:
            vals = df[col]
            color = CHART_COLORS[i % len(CHART_COLORS)]
            fig.add_trace(
                go.Bar(
                    x=df["Quarter"],
                    y=vals,
                    name=col,
                    marker_color=color,
                    marker_line_color=color,
                    marker_line_width=0,
                    hovertemplate=f"<b>{col}</b><br>%{{x}}: %{{y:{y_format}}}{y_suffix}<extra></extra>",
                )
            )

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=title, x=0.01),
        barmode="group",
        height=380,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED), ticksuffix=y_suffix),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED, size=11),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_waterfall_revenue(df: pd.DataFrame) -> go.Figure:
    """Build a revenue waterfall chart."""
    if df.empty or "Revenue" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="Revenue Trend")
        return fig

    fig = go.Figure(
        go.Waterfall(
            x=df["Quarter"],
            y=df["Revenue"],
            measure=["absolute"] + ["relative"] * (len(df) - 1),
            connector=dict(line=dict(color=BORDER)),
            increasing=dict(marker=dict(color=GREEN)),
            decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=ACCENT)),
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Quarterly Revenue", x=0.01),
        height=380,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED), tickprefix="$"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED, size=11)),
    )
    return fig


def build_valuation_gauge(label: str, value: float, min_val: float, max_val: float, suffix: str = "x") -> go.Figure:
    """Build a gauge chart for a valuation metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value if value is not None else 0,
        number=dict(suffix=suffix, font=dict(color=TEXT, size=28)),
        gauge=dict(
            axis=dict(range=[min_val, max_val], tickfont=dict(color=TEXT_MUTED, size=10)),
            bar=dict(color=ACCENT, thickness=0.7),
            bgcolor=CARD_BG,
            borderwidth=0,
            steps=[
                dict(range=[min_val, max_val * 0.33], color="rgba(109,170,69,0.15)"),
                dict(range=[max_val * 0.33, max_val * 0.66], color="rgba(255,197,83,0.15)"),
                dict(range=[max_val * 0.66, max_val], color="rgba(221,105,116,0.15)"),
            ],
        ),
    ))
    gauge_base = {k: v for k, v in PLOTLY_BASE.items() if k != 'margin'}
    fig.update_layout(
        **gauge_base,
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        title=dict(text=label, font=dict(size=12, color=TEXT_MUTED), x=0.5, y=0.95),
    )
    return fig


# ──────────────────────────────────────────────────────────────────
# 7. SIDEBAR
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size: 2rem; margin-bottom: 4px;">📊</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: {TEXT}; letter-spacing: 0.04em;">
            Equity Research Dashboard
        </div>
        <div style="font-size: 0.75rem; color: {TEXT_MUTED}; margin-top: 2px;">
            Financial Health & Valuation Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    ticker_input = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, TSLA",
        help="Enter a valid US-listed stock ticker symbol."
    )

    st.markdown(f"""
    <div style="font-size: 0.75rem; color: {TEXT_MUTED}; margin-top: -10px; margin-bottom: 16px;">
        Supports NYSE, NASDAQ, and AMEX listed equities
    </div>
    """, unsafe_allow_html=True)

    generate_btn = st.button(
        "⚡ Generate Report",
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")

    st.markdown(f"""
    <div style="font-size: 0.72rem; color: {TEXT_MUTED}; line-height: 1.6;">
        <b style="color: {ACCENT};">Data Sources</b><br>
        • Market data — Yahoo Finance<br>
        • Filings — SEC EDGAR (free API)<br>
        • Ratios — Computed from quarterly statements<br><br>
        <b style="color: {ACCENT};">Methodology</b><br>
        • Liquidity, profitability, and leverage ratios<br>
        &nbsp;&nbsp;derived from the 4 most recent 10-Q filings<br>
        • Valuation multiples use trailing 12-month data<br>
        • All figures in USD unless noted otherwise
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# 8. MAIN DASHBOARD
# ──────────────────────────────────────────────────────────────────

if generate_btn or ticker_input:
    ticker = ticker_input.strip().upper()

    if not ticker:
        st.warning("Please enter a stock ticker.")
        st.stop()

    # ── Loading state ──
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            info, hist = fetch_market_data(ticker)
        except Exception as e:
            st.error(f"Could not fetch market data for **{ticker}**. Verify the ticker symbol is correct.\n\n`{e}`")
            st.stop()

        if not info or "symbol" not in info:
            st.error(f"No data found for ticker **{ticker}**. Please check the symbol.")
            st.stop()

        try:
            inc_q, bs_q, cf_q, inc_a, bs_a = fetch_financial_statements(ticker)
        except Exception:
            inc_q = bs_q = cf_q = inc_a = bs_a = pd.DataFrame()

    # ── Company Header ──
    company_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector", "—")
    industry = info.get("industry", "—")
    exchange = info.get("exchange", "—")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or (hist["Close"].iloc[-1] if not hist.empty else None)
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    market_cap = info.get("marketCap")

    price_change = None
    price_change_pct = None
    if current_price and prev_close:
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100

    st.markdown(f"""
    <div style="display:flex; align-items:baseline; gap: 16px; margin-bottom: 4px;">
        <span style="font-size: 1.8rem; font-weight: 700; color: {TEXT};">{company_name}</span>
        <span style="font-size: 1.1rem; font-weight: 500; color: {ACCENT};">{ticker}</span>
        <span style="font-size: 0.8rem; color: {TEXT_MUTED};">{exchange}</span>
    </div>
    <div style="font-size: 0.82rem; color: {TEXT_MUTED}; margin-bottom: 20px;">
        {sector} &nbsp;·&nbsp; {industry}
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        delta_str = f"${price_change:+.2f} ({price_change_pct:+.1f}%)" if price_change is not None else None
        st.metric("Current Price", f"${current_price:,.2f}" if current_price else "N/A", delta=delta_str)
    with kpi2:
        st.metric("Market Cap", fmt_large_number(market_cap) if market_cap else "N/A")
    with kpi3:
        beta = info.get("beta")
        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
    with kpi4:
        div_yield = info.get("trailingAnnualDividendYield") or info.get("yield")
        if div_yield and div_yield < 1:
            st.metric("Div. Yield", f"{div_yield*100:.2f}%")
        elif div_yield:
            st.metric("Div. Yield", f"{div_yield:.2f}%")
        else:
            st.metric("Div. Yield", "N/A")
    with kpi5:
        wk52_high = info.get("fiftyTwoWeekHigh")
        wk52_low = info.get("fiftyTwoWeekLow")
        if wk52_high and wk52_low:
            st.metric("52-Wk Range", f"${wk52_low:,.0f} – ${wk52_high:,.0f}")
        else:
            st.metric("52-Wk Range", "N/A")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════

    tab_market, tab_health, tab_valuation, tab_sec = st.tabs([
        "📈 Market Data",
        "🏥 Financial Health",
        "💰 Valuation",
        "📄 SEC Summary",
    ])

    # ──────────────────────────────────────────────────────────
    # TAB 1: MARKET DATA
    # ──────────────────────────────────────────────────────────
    with tab_market:
        st.markdown(f'<div class="section-header">Price & Volume — Trailing 12 Months</div>', unsafe_allow_html=True)

        if not hist.empty:
            fig_price = build_price_chart(hist, ticker)
            st.plotly_chart(fig_price, use_container_width=True)

            # Performance metrics
            pm1, pm2, pm3, pm4 = st.columns(4)
            if len(hist) > 0:
                ytd_start = hist["Close"].iloc[0]
                ytd_end = hist["Close"].iloc[-1]
                ytd_return = ((ytd_end - ytd_start) / ytd_start) * 100

                high_52 = hist["Close"].max()
                low_52 = hist["Close"].min()
                avg_vol = hist["Volume"].mean()

                with pm1:
                    st.metric("1Y Return", f"{ytd_return:+.1f}%")
                with pm2:
                    st.metric("1Y High", f"${high_52:,.2f}")
                with pm3:
                    st.metric("1Y Low", f"${low_52:,.2f}")
                with pm4:
                    st.metric("Avg Daily Vol.", f"{avg_vol:,.0f}")
        else:
            st.info("No historical price data available.")

    # ──────────────────────────────────────────────────────────
    # TAB 2: FINANCIAL HEALTH
    # ──────────────────────────────────────────────────────────
    with tab_health:
        ratios_df = compute_quarterly_ratios(inc_q, bs_q, cf_q, info)

        if ratios_df.empty:
            st.warning("Quarterly financial data is not available for this ticker.")
        else:
            # ── Current-quarter snapshot KPIs ──
            latest = ratios_df.iloc[-1]

            st.markdown(f'<div class="section-header">Latest Quarter Snapshot — {latest["Quarter"]}</div>', unsafe_allow_html=True)

            l1, l2, l3, l4, l5, l6, l7 = st.columns(7)
            with l1:
                st.metric("Current Ratio", fmt_ratio(latest.get("Current Ratio")))
            with l2:
                st.metric("Quick Ratio", fmt_ratio(latest.get("Quick Ratio")))
            with l3:
                st.metric("ROE", fmt_pct(latest.get("ROE (%)")))
            with l4:
                st.metric("ROA", fmt_pct(latest.get("ROA (%)")))
            with l5:
                st.metric("Net Margin", fmt_pct(latest.get("Net Margin (%)")))
            with l6:
                st.metric("D/E Ratio", fmt_ratio(latest.get("Debt-to-Equity")))
            with l7:
                st.metric("Int. Coverage", fmt_ratio(latest.get("Interest Coverage")))

            st.markdown("---")

            # ── Liquidity Chart ──
            col_liq, col_prof = st.columns(2)

            with col_liq:
                st.markdown(f'<div class="section-header">Liquidity Ratios</div>', unsafe_allow_html=True)
                fig_liq = build_ratio_chart(
                    ratios_df,
                    ["Current Ratio", "Quick Ratio"],
                    "Current Ratio & Quick Ratio (Last 4 Quarters)",
                    y_suffix="x",
                )
                st.plotly_chart(fig_liq, use_container_width=True)

            with col_prof:
                st.markdown(f'<div class="section-header">Profitability Ratios</div>', unsafe_allow_html=True)
                fig_prof = build_ratio_chart(
                    ratios_df,
                    ["ROE (%)", "ROA (%)", "Net Margin (%)"],
                    "ROE, ROA & Net Margin (Last 4 Quarters)",
                    y_suffix="%",
                )
                st.plotly_chart(fig_prof, use_container_width=True)

            # ── Leverage Chart ──
            col_lev, col_rev = st.columns(2)

            with col_lev:
                st.markdown(f'<div class="section-header">Leverage Ratios</div>', unsafe_allow_html=True)
                fig_lev = build_ratio_chart(
                    ratios_df,
                    ["Debt-to-Equity", "Interest Coverage"],
                    "Debt-to-Equity & Interest Coverage (Last 4 Quarters)",
                    y_suffix="x",
                )
                st.plotly_chart(fig_lev, use_container_width=True)

            with col_rev:
                st.markdown(f'<div class="section-header">Revenue Trend</div>', unsafe_allow_html=True)
                if "Revenue" in ratios_df.columns and ratios_df["Revenue"].notna().any():
                    fig_rev = go.Figure()
                    fig_rev.add_trace(go.Bar(
                        x=ratios_df["Quarter"],
                        y=ratios_df["Revenue"],
                        marker_color=ACCENT,
                        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
                    ))
                    fig_rev.update_layout(
                        **PLOTLY_BASE,
                        title=dict(text="Quarterly Revenue", x=0.01),
                        height=380,
                        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
                        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_MUTED), tickprefix="$"),
                    )
                    st.plotly_chart(fig_rev, use_container_width=True)
                else:
                    st.info("Revenue data not available.")

            # ── Detailed Ratio Table ──
            with st.expander("📋 Detailed Ratio Table (All Quarters)", expanded=False):
                display_cols = ["Quarter", "Current Ratio", "Quick Ratio", "ROE (%)", "ROA (%)",
                                "Net Margin (%)", "Debt-to-Equity", "Interest Coverage"]
                available_cols = [c for c in display_cols if c in ratios_df.columns]
                styled_df = ratios_df[available_cols].copy()
                for col in available_cols:
                    if col != "Quarter":
                        styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────
    # TAB 3: VALUATION
    # ──────────────────────────────────────────────────────────
    with tab_valuation:
        st.markdown(f'<div class="section-header">Valuation Multiples — Trailing 12 Months</div>', unsafe_allow_html=True)

        val_ratios = compute_valuation_ratios(info)

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            pe = val_ratios.get("P/E Ratio")
            st.metric("P/E Ratio", fmt_ratio(pe) if pe else "N/A")
        with v2:
            fpe = val_ratios.get("Forward P/E")
            st.metric("Forward P/E", fmt_ratio(fpe) if fpe else "N/A")
        with v3:
            ev_ebitda = val_ratios.get("EV/EBITDA")
            st.metric("EV/EBITDA", fmt_ratio(ev_ebitda) if ev_ebitda else "N/A")
        with v4:
            ptb = val_ratios.get("Price-to-Book")
            st.metric("Price-to-Book", fmt_ratio(ptb) if ptb else "N/A")

        st.markdown("")

        v5, v6, v7, v8 = st.columns(4)
        with v5:
            ev_rev = val_ratios.get("EV/Revenue")
            st.metric("EV/Revenue", fmt_ratio(ev_rev) if ev_rev else "N/A")
        with v6:
            peg = val_ratios.get("PEG Ratio")
            st.metric("PEG Ratio", fmt_ratio(peg) if peg else "N/A")
        with v7:
            ev = val_ratios.get("Enterprise Value")
            st.metric("Enterprise Value", fmt_large_number(ev) if ev else "N/A")
        with v8:
            ps = info.get("priceToSalesTrailing12Months")
            st.metric("Price/Sales (TTM)", fmt_ratio(ps) if ps else "N/A")

        st.markdown("---")

        # ── Gauge Charts ──
        st.markdown(f'<div class="section-header">Valuation Gauges</div>', unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)
        with g1:
            fig_pe = build_valuation_gauge("P/E Ratio", val_ratios.get("P/E Ratio"), 0, 60, "x")
            st.plotly_chart(fig_pe, use_container_width=True)
        with g2:
            fig_ev = build_valuation_gauge("EV/EBITDA", val_ratios.get("EV/EBITDA"), 0, 40, "x")
            st.plotly_chart(fig_ev, use_container_width=True)
        with g3:
            fig_pb = build_valuation_gauge("Price-to-Book", val_ratios.get("Price-to-Book"), 0, 15, "x")
            st.plotly_chart(fig_pb, use_container_width=True)

        # ── Peer comparison placeholder ──
        with st.expander("📊 Sector Comparison Context", expanded=False):
            st.markdown(f"""
            <div style="font-size: 0.85rem; color: {TEXT_MUTED}; line-height: 1.7;">
                <b style="color: {ACCENT};">Interpretation Guide for Modeling</b><br><br>
                <b>P/E Ratio:</b> Compare against the sector median. Growth companies typically trade at
                higher P/E multiples. A P/E significantly above peers may indicate overvaluation or
                high growth expectations baked into the price.<br><br>
                <b>EV/EBITDA:</b> The preferred multiple for capital-intensive industries and M&A analysis.
                Removes distortions from capital structure, tax differences, and depreciation policies.
                Typical range: 8–15x for mature companies, 15–30x for high-growth tech.<br><br>
                <b>Price-to-Book:</b> Most meaningful for financial institutions, real estate, and
                asset-heavy businesses. A P/B below 1.0x may signal undervaluation or asset impairment risk.
                Less relevant for asset-light business models (SaaS, consulting).<br><br>
                <b>PEG Ratio:</b> Adjusts P/E for expected growth. A PEG of 1.0x suggests fair valuation
                relative to growth; below 1.0x is potentially undervalued. Use cautiously — relies on
                analyst growth estimates which may be stale or overly optimistic.
            </div>
            """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TAB 4: SEC SUMMARY
    # ──────────────────────────────────────────────────────────
    with tab_sec:
        st.markdown(f'<div class="section-header">SEC Filing Summary — Latest 10-K</div>', unsafe_allow_html=True)

        with st.spinner("Retrieving SEC filing data..."):
            sec_data = fetch_sec_summary(ticker, company_name)

        # Filing source
        source = sec_data.get("source", "N/A")
        filing_url = sec_data.get("filing_url")

        col_src, col_link = st.columns([3, 1])
        with col_src:
            st.markdown(f"""
            <div style="font-size: 0.82rem; color: {TEXT_MUTED};">
                Source: <span style="color: {ACCENT};">{source}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_link:
            if filing_url:
                st.markdown(f'<a href="{filing_url}" target="_blank" style="color: {ACCENT}; font-size: 0.82rem; text-decoration: none;">View Full 10-K Filing →</a>', unsafe_allow_html=True)

        st.markdown("")

        # Business Description
        with st.expander("🏢 Business Description", expanded=True):
            biz_desc = sec_data.get("business_description", "Not available.")
            st.markdown(f"""
            <div style="font-size: 0.88rem; color: {TEXT}; line-height: 1.75; max-height: 400px; overflow-y: auto;">
                {biz_desc}
            </div>
            """, unsafe_allow_html=True)

        # Risk Factors
        with st.expander("⚠️ Risk Factors", expanded=True):
            risk_text = sec_data.get("risk_factors", "Not available.")
            st.markdown(f"""
            <div style="font-size: 0.88rem; color: {TEXT}; line-height: 1.75; max-height: 400px; overflow-y: auto;">
                {risk_text}
            </div>
            """, unsafe_allow_html=True)

        # Additional yfinance info
        with st.expander("📋 Company Profile (Yahoo Finance)", expanded=False):
            profile_items = {
                "Full-Time Employees": info.get("fullTimeEmployees"),
                "Country": info.get("country"),
                "City": info.get("city"),
                "State": info.get("state"),
                "Website": info.get("website"),
                "Fiscal Year End": info.get("lastFiscalYearEnd"),
                "Most Recent Quarter": info.get("mostRecentQuarter"),
            }
            for k, v in profile_items.items():
                if v:
                    if k == "Website":
                        st.markdown(f"**{k}:** [{v}]({v})")
                    elif k in ("Full-Time Employees",):
                        st.markdown(f"**{k}:** {v:,}")
                    else:
                        val = str(v)
                        if k in ("Fiscal Year End", "Most Recent Quarter"):
                            try:
                                val = datetime.fromtimestamp(int(v)).strftime("%B %d, %Y")
                            except Exception:
                                pass
                        st.markdown(f"**{k}:** {val}")

    # ── Footer ──
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; font-size: 0.72rem; color: {TEXT_MUTED}; padding: 10px 0;">
        Equity Research Dashboard · Data as of {datetime.now().strftime('%B %d, %Y %I:%M %p')} ·
        Market data via Yahoo Finance · SEC filings via EDGAR EFTS API ·
        For educational and research purposes only — not investment advice
    </div>
    """, unsafe_allow_html=True)
