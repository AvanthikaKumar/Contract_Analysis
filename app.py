"""
app.py
------
Streamlit entry point for the Contract Intelligence System.
 
Run with:
    streamlit run app.py
"""
 
import sys
from pathlib import Path
 
# Guarantee project root is always on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
import streamlit as st
 
from ui.tab_upload import render_upload_tab
from ui.tab_query import render_query_tab
 
# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Contract Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)
 
# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------
st.title("📄 Contract Intelligence System")
st.markdown(
    "AI-powered contract analysis using **GraphRAG** — "
    "Azure OpenAI · Azure AI Search · Cosmos DB"
)
st.divider()
 
# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "📤  Contract Upload & Processing",
    "🔍  Contract Intelligence",
])
 
with tab1:
    render_upload_tab()
 
with tab2:
    render_query_tab()
 
