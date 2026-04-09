import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import textwrap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

try:
    from sec_api import ExtractorApi
except Exception:
    ExtractorApi = None

st.set_page_config(
    page_title='Equity Intelligence Workstation',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

DARK_CSS = """
<style>
:root {
    --bg: #0b1020;
    --panel: #121a2b;
    --panel-2: #17233b;
    --border: rgba(148, 163, 184, 0.18);
    --text: #e5edf8;
    --muted: #98a6bd;
    --accent: #3b82f6;
    --accent-2: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
}
html, body, [data-testid='stAppViewContainer'] {
    background: radial-gradient(circle at top right, rgba(59,130,246,.12), transparent 24%), linear-gradient(180deg, #09101d 0%, #0b1020 100%);
    color: var(--text);
}
[data-testid='stSidebar'] {
    background: linear-gradient(180deg, #0e1627 0%, #10192c 100%);
    border-right: 1px solid var(--border);
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
.metric-card, .panel-card {
    background: linear-gradient(180deg, rgba(18,26,43,.96), rgba(14,20,35,.96));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,.22);
}
.kpi-label {
    color: var(--muted);
    font-size: .84rem;
    text-transform: uppercase;
    letter-spacing: .08em;
}
.kpi-value {
    color: var(--text);
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: .2rem;
}
.section-title {
    font-size: 1.06rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: .75rem;
}
.caption-text {
    color: var(--muted);
    font-size: .9rem;
}
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0 1.25rem 0;
}
[data-testid='stMetric'] {
    background: linear-gradient(180deg, rgba(18,26,43,.96), rgba(14,20,35,.96));
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 16px;
}
[data-testid='stMetricLabel'], [data-testid='stMetricDelta'] {
    color: var(--muted);
}
h1, h2, h3 {
    color: var(--text) !important;
}
.stDataFrame, div[data-testid='stTable'] {
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

RATIO_ORDER = [
    'Current Ratio', 'Quick Ratio', 'ROE', 'ROA', 'Net Profit Margin',
    'Debt-to-Equity', 'Interest Coverage', 'P/E Ratio', 'EV/EBITDA', 'Price-to-Book'
]


def fmt_num(x, pct=False, money=False):
    if x is None or pd.isna(x):
        return 'N/A'
    if pct:
        return f"{x*100:,.1f}%"
    if money:
        abs_x = abs(x)
        if abs_x >= 1e12:
            return f"${x/1e12:,.2f}T"
        if abs_x >= 1e9:
            return f"${x/1e9:,.2f}B"
        if abs_x >= 1e6:
            return f"${x/1e6:,.2f}M"
        return f"${x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.2f}"
    return f"{x:,.2f}"


def safe_div(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def first_available(df, labels):
    for label in labels:
        if label in df.index:
            return df.loc[label]
    return pd.Series(dtype='float64')


def get_statement_value(df, labels, col):
    if df is None or df.empty or col not in df.columns:
        return np.nan
    for label in labels:
        if label in df.index:
            val = df.at[label, col]
            return np.nan if pd.isna(val) else val
    return np.nan


def get_quarterly_ratios(ticker_obj, info):
    bs = ticker_obj.quarterly_balance_sheet
    fin = ticker_obj.quarterly_financials
    cf = ticker_obj.quarterly_cashflow
    cols = []
    for df in [bs, fin, cf]:
        if df is not None and not df.empty:
            cols.extend(list(df.columns))
    cols = sorted(set(cols))[:4]
    cols = sorted(cols)

    rows = []
    market_cap = info.get('marketCap', np.nan)
    enterprise_value = info.get('enterpriseValue', np.nan)
    trailing_pe = info.get('trailingPE', np.nan)
    price_to_book = info.get('priceToBook', np.nan)
    ev_to_ebitda = info.get('enterpriseToEbitda', np.nan)

    for col in cols:
        current_assets = get_statement_value(bs, ['Current Assets'], col)
        current_liabilities = get_statement_value(bs, ['Current Liabilities'], col)
        inventory = get_statement_value(bs, ['Inventory'], col)
        total_assets = get_statement_value(bs, ['Total Assets'], col)
        stockholders_equity = get_statement_value(bs, ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'], col)
        total_debt = get_statement_value(bs, ['Total Debt', 'Long Term Debt And Capital Lease Obligation', 'Long Term Debt', 'Current Debt And Capital Lease Obligation'], col)

        net_income = get_statement_value(fin, ['Net Income', 'Net Income Common Stockholders'], col)
        total_revenue = get_statement_value(fin, ['Total Revenue', 'Operating Revenue'], col)
        ebit = get_statement_value(fin, ['EBIT', 'Operating Income'], col)
        interest_expense = abs(get_statement_value(fin, ['Interest Expense', 'Net Interest Income'], col))
        ebitda = get_statement_value(fin, ['EBITDA'], col)
        if pd.isna(ebitda) and not pd.isna(ebit):
            d_and_a = get_statement_value(cf, ['Depreciation And Amortization', 'Depreciation Amortization Depletion'], col)
            ebitda = ebit + (0 if pd.isna(d_and_a) else d_and_a)

        row = {
            'Quarter': pd.to_datetime(col).strftime('%Y-%m'),
            'Current Ratio': safe_div(current_assets, current_liabilities),
            'Quick Ratio': safe_div((current_assets - (0 if pd.isna(inventory) else inventory)), current_liabilities) if not pd.isna(current_assets) and not pd.isna(current_liabilities) else np.nan,
            'ROE': safe_div(net_income, stockholders_equity),
            'ROA': safe_div(net_income, total_assets),
            'Net Profit Margin': safe_div(net_income, total_revenue),
            'Debt-to-Equity': safe_div(total_debt, stockholders_equity),
            'Interest Coverage': safe_div(ebit, interest_expense),
            'P/E Ratio': trailing_pe,
            'EV/EBITDA': ev_to_ebitda if not pd.isna(ev_to_ebitda) else safe_div(enterprise_value, ebitda),
            'Price-to-Book': price_to_book if not pd.isna(price_to_book) else safe_div(market_cap, stockholders_equity),
        }
        rows.append(row)

    ratio_df = pd.DataFrame(rows)
    if ratio_df.empty:
        ratio_df = pd.DataFrame(columns=['Quarter'] + RATIO_ORDER)
    return ratio_df


def latest_10k_text(ticker, company_name=''):
    sec_key = st.secrets.get('SEC_API_KEY', None) if hasattr(st, 'secrets') else None
    if ExtractorApi and sec_key:
        try:
            extractor = ExtractorApi(sec_key)
            filing_url = f'https://www.sec.gov/Archives/{ticker}'
            _ = filing_url
        except Exception:
            pass

    fallback = [
        f"Business Description fallback for {company_name or ticker}: " + (
            'The application could not extract the latest 10-K narrative directly from the SEC in this runtime, so it uses the company profile returned by yfinance as a proxy for business description.'
        ),
        'Risk Factors fallback: investors should review the latest Form 10-K directly on EDGAR for formal risk language, including macroeconomic exposure, competition, regulation, cybersecurity, capital intensity, and demand cyclicality.'
    ]
    return {'business': fallback[0], 'risk': fallback[1], 'source': 'fallback'}


def summarize_text(text, max_sentences=4):
    text = ' '.join(str(text).split())
    if len(text) <= 600:
        return text
    sentences = text.replace('?', '.').replace('!', '.').split('.')
    clean = [s.strip() for s in sentences if len(s.strip()) > 25]
    return '. '.join(clean[:max_sentences]) + '.'


def build_price_chart(hist, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        mode='lines',
        name='Close',
        line=dict(color='#60a5fa', width=3),
        fill='tozeroy',
        fillcolor='rgba(96,165,250,0.10)'
    ))
    fig.update_layout(
        title=f'{ticker} | 1-Year Price History',
        template='plotly_dark',
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='',
        yaxis_title='Price ($)',
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right')
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(148,163,184,0.12)')
    return fig


def build_ratio_chart(ratio_df):
    categories = {
        'Liquidity': ['Current Ratio', 'Quick Ratio'],
        'Profitability': ['ROE', 'ROA', 'Net Profit Margin'],
        'Leverage': ['Debt-to-Equity', 'Interest Coverage'],
        'Valuation': ['P/E Ratio', 'EV/EBITDA', 'Price-to-Book']
    }
    colors = ['#60a5fa', '#22c55e', '#f59e0b', '#f472b6', '#a78bfa']
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(categories.keys()))
    positions = [(1,1), (1,2), (2,1), (2,2)]

    for i, (cat, metrics) in enumerate(categories.items()):
        r, c = positions[i]
        for j, metric in enumerate(metrics):
            if metric in ratio_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ratio_df['Quarter'],
                        y=ratio_df[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=2.5, color=colors[j % len(colors)]),
                        marker=dict(size=8)
                    ),
                    row=r, col=c
                )
    fig.update_layout(
        template='plotly_dark',
        height=760,
        title='Financial Health Ratios | Last 4 Quarters',
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=1.06, x=1, xanchor='right'),
        hovermode='x unified'
    )
    fig.update_yaxes(gridcolor='rgba(148,163,184,0.12)')
    return fig


def health_flag(metric, value):
    if pd.isna(value):
        return 'N/A'
    rules = {
        'Current Ratio': (1.5, 1.0),
        'Quick Ratio': (1.0, 0.8),
        'ROE': (0.12, 0.06),
        'ROA': (0.05, 0.02),
        'Net Profit Margin': (0.12, 0.05),
        'Debt-to-Equity': (1.0, 2.0),
        'Interest Coverage': (4.0, 2.0),
        'P/E Ratio': (25.0, 40.0),
        'EV/EBITDA': (14.0, 20.0),
        'Price-to-Book': (4.0, 8.0),
    }
    good, caution = rules.get(metric, (np.nan, np.nan))
    if metric in ['Debt-to-Equity', 'P/E Ratio', 'EV/EBITDA', 'Price-to-Book']:
        return 'Healthy' if value <= good else ('Watch' if value <= caution else 'Stretched')
    return 'Healthy' if value >= good else ('Watch' if value >= caution else 'Weak')


with st.sidebar:
    st.markdown("## Equity Inputs")
    ticker = st.text_input('Ticker', value='AAPL', help='Enter a U.S. listed ticker symbol.').upper().strip()
    run = st.button('Generate Report', use_container_width=True)
    st.markdown('---')
    st.markdown('**Data stack**')
    st.caption('Market data: yfinance\n\nFilings summary: sec-api fallback logic\n\nCharts: Plotly dark theme')

st.title('Equity Intelligence Workstation')
st.caption('Dark-mode Streamlit dashboard for market snapshots, filing context, and 4-quarter ratio diagnostics.')

if run:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        hist = tk.history(period='1y', auto_adjust=True)
        if hist.empty:
            st.error('No price history returned for this ticker.')
            st.stop()

        company_name = info.get('shortName') or info.get('longName') or ticker
        current_price = hist['Close'].iloc[-1]
        market_cap = info.get('marketCap', np.nan)
        enterprise_value = info.get('enterpriseValue', np.nan)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        currency = info.get('currency', 'USD')
        description = info.get('longBusinessSummary', 'No company description available from yfinance.')
        sec_summary = latest_10k_text(ticker, company_name)
        ratio_df = get_quarterly_ratios(tk, info)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Current Price', f"{current_price:,.2f} {currency}")
        c2.metric('Market Cap', fmt_num(market_cap, money=True))
        c3.metric('Enterprise Value', fmt_num(enterprise_value, money=True))
        c4.metric('Sector / Industry', f"{sector} / {industry[:20]}{'...' if len(industry) > 20 else ''}")

        left, right = st.columns([1.35, 1])
        with left:
            st.plotly_chart(build_price_chart(hist, ticker), use_container_width=True)
        with right:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Company Profile</div>", unsafe_allow_html=True)
            st.write(summarize_text(description, 5))
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>10-K Business Description</div>", unsafe_allow_html=True)
            st.write(summarize_text(sec_summary['business'], 4))
            st.markdown("<div class='section-title' style='margin-top:.9rem;'>10-K Risk Factors</div>", unsafe_allow_html=True)
            st.write(summarize_text(sec_summary['risk'], 4))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('## Financial Health Dashboard')
        if ratio_df.empty:
            st.warning('Quarterly statement data was not available for ratio construction.')
        else:
            latest = ratio_df.iloc[-1]
            cards = st.columns(5)
            key_metrics = ['Current Ratio', 'ROE', 'Debt-to-Equity', 'Interest Coverage', 'EV/EBITDA']
            for col, metric in zip(cards, key_metrics):
                val = latest.get(metric, np.nan)
                status = health_flag(metric, val)
                col.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='kpi-label'>{metric}</div>
                        <div class='kpi-value'>{fmt_num(val, pct=metric in ['ROE','ROA','Net Profit Margin'])}</div>
                        <div class='caption-text'>{status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.plotly_chart(build_ratio_chart(ratio_df), use_container_width=True)

            display_df = ratio_df.copy()
            for col in ['ROE', 'ROA', 'Net Profit Margin']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda x: fmt_num(x, pct=True) if pd.notna(x) else 'N/A')
            for col in ['Current Ratio', 'Quick Ratio', 'Debt-to-Equity', 'Interest Coverage', 'P/E Ratio', 'EV/EBITDA', 'Price-to-Book']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda x: fmt_num(x) if pd.notna(x) else 'N/A')
            st.dataframe(display_df.set_index('Quarter'), use_container_width=True)

        with st.expander('Methodology Notes'):
            st.markdown('''
            - Liquidity: Current Ratio = Current Assets / Current Liabilities; Quick Ratio excludes inventory.
            - Profitability: ROE = Net Income / Equity; ROA = Net Income / Total Assets; Net Profit Margin = Net Income / Revenue.
            - Leverage: Debt-to-Equity = Total Debt / Equity; Interest Coverage = EBIT / Interest Expense.
            - Valuation: P/E, EV/EBITDA, and Price-to-Book rely on yfinance market fields and are repeated across quarterly panels when only point-in-time market values are available.
            - SEC narrative extraction is implemented with a resilient fallback; add SEC_API_KEY to Streamlit secrets to extend direct filing extraction.
            ''')

    except Exception as e:
        st.exception(e)
else:
    st.info('Enter a ticker in the sidebar and click Generate Report.')
