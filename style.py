CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stMarkdown, p, label, h1, h2, h3, h4, input, textarea, button, [data-testid="stMetric"] {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
#MainMenu, footer, [data-testid="stHeaderActionElements"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1040px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1 { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.03em; }
h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
h2 { font-size: 1.35rem; margin-top: .4rem; }
h3 { font-size: 1.05rem; }
.brand { font-weight: 700; font-size: 1.25rem; letter-spacing: -0.03em; color: #111827; }
.brand span { color: #0F766E; }
.muted { color: #6B7280; font-size: .92rem; }
.tag { display: inline-block; padding: 2px 10px; margin: 0 6px 4px 0; border: 1px solid #E5E7EB; border-radius: 999px;
       font-size: .75rem; color: #4B5563; background: #fff; }
.title { font-weight: 600; font-size: 1.02rem; color: #111827; margin-bottom: 2px; }
.reason { font-size: .85rem; color: #0F766E; margin-top: 2px; }
hr { border: none; border-top: 1px solid #ECEEF2; margin: 1.2rem 0; }
[data-testid="stMetric"] { background: #F6F7F9; border: 1px solid #ECEEF2; border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricLabel"] p { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: #6B7280; }
[data-testid="stMetricValue"] { font-weight: 600; letter-spacing: -0.02em; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; border-color: #ECEEF2; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 8px; font-weight: 500; border: 1px solid #E5E7EB; box-shadow: none; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
  background: #0F766E; border-color: #0F766E; color: #fff; }
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stDateInput input { border-radius: 8px; }
[data-testid="stDataFrame"] { border: 1px solid #ECEEF2; border-radius: 12px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid #ECEEF2; border-radius: 12px; }
button[data-baseweb="tab"] { font-weight: 500; }
</style>
"""
