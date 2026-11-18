"""AlerteFraude IA – Plateforme Exécutive de Détection de Fraude Bancaire."""
import base64
import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AlerteFraude IA | Sécurité Bancaire",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Empêche la traduction automatique du navigateur (Chrome/Edge) : elle modifie le DOM
# et provoque « NotFoundError : removeChild » dans React/Streamlit.
_JS_NOTRANSLATE = """
const d = document;
d.documentElement.setAttribute('translate', 'no');
d.documentElement.setAttribute('lang', 'fr');
d.body.classList.add('notranslate');
if (!d.querySelector('meta[name="google"]')) {
    const m = d.createElement('meta'); m.name = 'google'; m.content = 'notranslate';
    d.head.appendChild(m);
}
"""
try:
    st.html(f"<script>{_JS_NOTRANSLATE}</script>", unsafe_allow_javascript=True)   # Streamlit récent
except TypeError:
    try:                                                                            # anciennes versions
        components.html(f"<script>{_JS_NOTRANSLATE.replace('document', 'window.parent.document')}</script>", height=0)
    except Exception:
        pass
except Exception:
    pass

# ------------------------------------------------------------------ PALETTE & CONSTANTES
BG, CARD, LINE = "#040b1e", "#0a1736", "#1a3166"
GOLD, GRN, RED, BLU, YEL, MUT = "#f0b429", "#10b981", "#ef4444", "#3b82f6", "#f59e0b", "#8ea4d2"

TOTAL, FRAUDES = 284_807, 492
NORMALES = TOTAL - FRAUDES
TEST_N, TEST_F = 85_443, 108

# Le modèle a été entraîné sur le dataset « Credit Card Fraud » où Amount est exprimé en EUROS
# (moyenne ≈ 88 €, écart-type ≈ 250 €, médiane ≈ 22 €).
# Parité fixe officielle (zone CFA) : 1 € = 655,957 FCFA.
TAUX_EUR_FCFA = 655.957
NIVEAUX = ["Élevé", "Moyen", "Faible"]
COUL_NIVEAU = {"Élevé": RED, "Moyen": GOLD, "Faible": GRN}

METRIQUES = pd.DataFrame(
    {
        "Modèle": ["Isolation Forest", "Régression Logistique (Retenu)", "Gradient Boosting"],
        "PR-AUC": [0.039, 0.705, 0.599],
        "ROC-AUC": [0.946, 0.973, 0.907],
        "Rappel@50": [0.000, 0.417, 0.324],
        "Rappel@200": [0.028, 0.815, 0.750],
        "Rappel@500": [0.231, 0.843, 0.769],
    }
)


# ------------------------------------------------------------------ COMPATIBILITÉ STREAMLIT
def _version_st():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


NEW_ST = _version_st() >= (1, 50)


def afficher_df(data, **kw):
    if NEW_ST:
        return st.dataframe(data, width="stretch", **kw)
    return st.dataframe(data, use_container_width=True, **kw)


def afficher_plot(fig, **kw):
    kw.setdefault("theme", None)   # le thème Streamlit impose un texte sombre aux légendes : on garde nos couleurs
    if NEW_ST:
        return st.plotly_chart(fig, width="stretch", **kw)
    return st.plotly_chart(fig, use_container_width=True, **kw)


# ------------------------------------------------------------------ UTILITAIRES IMAGE
def get_base64_image(image_path: Path) -> str:
    if image_path.exists():
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return ""


BASE_DIR = Path(__file__).parent
PHOTO_PATH = BASE_DIR / "WhatsApp Image 2026-10-10 at 01.11.09.jpeg"
PHOTO_B64 = get_base64_image(PHOTO_PATH)

AVATAR_HTML = (
    f'<img src="data:image/jpeg;base64,{PHOTO_B64}" class="avatar-img" alt="Ibrahima DIENG"/>'
    if PHOTO_B64
    else '<div class="avatar-fallback">ID</div>'
)

# ------------------------------------------------------------------ CSS PREMIUM FINTECH
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #f1f5f9;
}}

.stApp {{
    background:
        radial-gradient(900px 450px at 80% -5%, rgba(37, 99, 235, 0.25), transparent 70%),
        radial-gradient(700px 400px at 10% 20%, rgba(240, 180, 41, 0.08), transparent 60%),
        {BG};
}}

#MainMenu, footer {{ visibility: hidden; }}
/* En-tête transparent : on garde visibles les DEUX flèches de la barre latérale (« réduire » et » rouvrir) */
header[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ visibility: hidden; }}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] * ,
[data-testid="stExpandSidebarButton"] * {{ visibility: visible !important; }}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {{
    background: rgba(15, 29, 66, 0.92) !important; border: 1.5px solid {GOLD} !important;
    border-radius: 12px !important; box-shadow: 0 0 14px rgba(240, 180, 41, 0.35) !important; z-index: 999;
}}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *,
[data-testid="stSidebarCollapseButton"] * {{ color: {GOLD} !important; fill: {GOLD} !important; }}
.block-container {{ padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1420px; }}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #06122e 0%, #030919 100%);
    border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] .stRadio label {{
    padding: 11px 16px;
    border-radius: 12px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.08);
}}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background: linear-gradient(90deg, {GOLD} 0%, #ffd166 100%);
    box-shadow: 0 4px 20px rgba(240, 180, 41, 0.35);
}}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p {{
    color: #0f172a !important;
    font-weight: 800 !important;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {{
    display: none;
}}

/* TOPBAR */
.topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    padding: 8px 6px 18px;
    border-bottom: 1px solid {LINE};
    margin-bottom: 20px;
}}
.brand {{ display: flex; align-items: center; gap: 14px; }}
.shield-logo {{
    width: 48px; height: 54px;
    background: linear-gradient(145deg, #ffd166, #b47b09);
    clip-path: polygon(50% 0%, 100% 16%, 100% 62%, 50% 100%, 0% 62%, 0% 16%);
    display: grid; place-items: center;
    font-size: 22px;
    box-shadow: 0 0 25px rgba(240, 180, 41, 0.4);
}}
.brand-title {{ font-size: 26px; font-weight: 800; letter-spacing: 0.6px; line-height: 1.1; }}
.brand-title span {{ color: {GOLD}; }}
.brand-sub {{ color: {MUT}; font-size: 13px; margin-top: 2px; }}

.profile-zone {{ display: flex; align-items: center; gap: 22px; }}
.sys-status {{ text-align: right; font-size: 12.5px; color: {MUT}; }}
.sys-online {{ color: #e2e8f0; font-weight: 600; display: flex; align-items: center; justify-content: flex-end; gap: 7px; }}
.pulse-dot {{
    width: 9px; height: 9px; border-radius: 50%; background: {GRN};
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse 2s infinite;
}}
@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
    70% {{ box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
}}

.user-pill {{
    display: flex; align-items: center; gap: 12px;
    background: rgba(15, 29, 66, 0.85);
    border: 1px solid {LINE};
    padding: 6px 14px 6px 8px;
    border-radius: 50px;
}}
.avatar-img {{
    width: 44px; height: 44px; border-radius: 50%;
    object-fit: cover; object-position: top center;
    border: 2px solid {GOLD};
    box-shadow: 0 0 12px rgba(240, 180, 41, 0.4);
}}
.avatar-fallback {{
    width: 44px; height: 44px; border-radius: 50%;
    background: linear-gradient(135deg, {BLU}, #6366f1);
    display: grid; place-items: center; font-weight: 800; border: 2px solid {GOLD};
}}

/* HERO BANNER */
.hero-banner {{
    position: relative;
    border-radius: 20px;
    border: 1px solid #26458a;
    padding: 32px 38px;
    margin-bottom: 22px;
    overflow: hidden;
    background:
        linear-gradient(95deg, rgba(5, 13, 36, 0.94) 0%, rgba(10, 27, 68, 0.86) 55%, rgba(16, 42, 99, 0.78) 100%),
        url('https://images.unsplash.com/photo-1541354329998-f4d9a9f9297f?auto=format&fit=crop&w=1400&q=80') center/cover no-repeat;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
}}
.hero-text h1 {{
    margin: 0 0 10px;
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.15;
    color: #ffffff;
}}
.hero-text h1 span {{ color: {GOLD}; }}
.hero-text p {{
    margin: 0;
    color: #cbd5e1;
    font-size: 1.02rem;
    max-width: 650px;
}}
.hero-badge-box {{
    min-width: 145px; height: 155px;
    background: radial-gradient(circle at 30% 30%, rgba(59, 130, 246, 0.35), rgba(10, 23, 54, 0.9));
    border: 2px solid rgba(240, 180, 41, 0.65);
    border-radius: 24px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    box-shadow: 0 0 35px rgba(59, 130, 246, 0.3), inset 0 0 20px rgba(240, 180, 41, 0.15);
}}
.hero-badge-box .lock {{ font-size: 52px; filter: drop-shadow(0 4px 12px rgba(240, 180, 41, 0.5)); }}
.hero-badge-box span {{ font-size: 11px; font-weight: 700; color: {GOLD}; letter-spacing: 1px; margin-top: 6px; }}

/* KPI CARDS */
.kpi-card {{
    background: linear-gradient(165deg, #0d1f4a 0%, {CARD} 100%);
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
}}
.kpi-card.b {{ border-top: 3px solid {BLU}; }}
.kpi-card.g {{ border-top: 3px solid {GRN}; }}
.kpi-card.r {{ border-top: 3px solid {RED}; }}
.kpi-card.y {{ border-top: 3px solid {GOLD}; }}
.kpi-top {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
.kpi-icon {{
    width: 42px; height: 42px; border-radius: 12px;
    display: grid; place-items: center; font-size: 19px; font-weight: 700;
}}
.kpi-title {{ color: {MUT}; font-size: 0.85rem; font-weight: 600; }}
.kpi-val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 700; color: #fff; line-height: 1.1; }}
.kpi-delta {{ font-size: 0.8rem; font-weight: 700; margin-left: 6px; }}
.kpi-bar {{ height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 99px; overflow: hidden; margin-top: 10px; }}
.kpi-bar div {{ height: 100%; border-radius: 99px; }}

/* GLASS PANELS */
.panel {{
    background: linear-gradient(170deg, #0c1d45 0%, {CARD} 100%);
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 18px 20px;
    height: 100%;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
}}
.panel-hdr {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; font-weight: 700; font-size: 1rem; color: #f8fafc;
}}

/* ALERTS LIST */
.alert-item {{
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 11px 12px; border-radius: 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 9px;
}}
.alert-item:last-child {{ margin-bottom: 0; }}
.badge-pill {{
    padding: 3px 10px; border-radius: 99px; font-size: 0.74rem; font-weight: 700; display: inline-block;
}}
.b-ok {{ background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }}
.b-sus {{ background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
.b-mid {{ background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}

/* TABLE CUSTOM */
.txn-table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
.txn-table th {{
    text-align: left; color: {MUT}; font-weight: 600; padding: 10px 12px;
    border-bottom: 1px solid {LINE}; font-size: 0.8rem; text-transform: uppercase;
}}
.txn-table td {{ padding: 11px 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }}
.txn-table tr:hover td {{ background: rgba(59, 130, 246, 0.08); }}
.mono {{ font-family: 'JetBrains Mono', monospace; }}

/* STATUS BANNER */
.perf-banner {{
    background: linear-gradient(90deg, rgba(240, 180, 41, 0.16), rgba(16, 185, 129, 0.12));
    border: 1px solid rgba(240, 180, 41, 0.45);
    border-radius: 12px;
    padding: 11px 14px;
    display: flex; align-items: center; gap: 12px;
    margin-top: 10px;
}}

/* BUTTONS */
.stButton>button, .stDownloadButton>button {{
    background: linear-gradient(90deg, {GOLD}, #ffd166) !important;
    color: #0f172a !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 99px !important;
    padding: 0.55rem 1.3rem !important;
    box-shadow: 0 4px 15px rgba(240, 180, 41, 0.3);
}}
.github-btn {{
    display: block; text-align: center; text-decoration: none;
    background: linear-gradient(90deg, {GOLD}, #ffd166);
    color: #0f172a !important; font-weight: 800; font-size: 0.88rem;
    padding: 10px 16px; border-radius: 99px; margin-top: 14px;
    box-shadow: 0 4px 18px rgba(240, 180, 41, 0.35);
}}


/* CLOCHE DE NOTIFICATION (icône 🔔 toujours visible + compteur) */
[data-testid="stPopover"] button, button[data-testid="stPopoverButton"] {{
    background: linear-gradient(145deg, #0f2a63, #0a1736) !important;
    border: 1.5px solid {GOLD} !important;
    color: #ffffff !important;
    border-radius: 50px !important;
    min-height: 48px; min-width: 76px; width: auto !important;
    padding: 0 16px !important;
    display: inline-flex !important; align-items: center !important; justify-content: center !important; gap: 8px !important;
    box-shadow: 0 0 14px rgba(240, 180, 41, 0.28) !important;
    white-space: nowrap !important; overflow: visible !important;
}}
[data-testid="stPopover"] button::before, button[data-testid="stPopoverButton"]::before {{
    content: "🔔"; font-size: 1.5rem; line-height: 1; display: inline-block;
}}
[data-testid="stPopover"] button p {{
    color: #fff !important; font-size: 1.25rem !important; font-weight: 800 !important;
    margin: 0 !important; min-width: 0; overflow: visible !important;
}}
[data-testid="stPopover"] button [data-testid="stIconMaterial"] {{ display: none !important; }}


/* CENTRE DE NOTIFICATIONS (popover sombre, lisible, sans débordement) */
[data-testid="stPopoverBody"], div[data-baseweb="popover"] > div {{
    background: linear-gradient(170deg, #0c1d45 0%, #071330 100%) !important;
    border: 1px solid {LINE} !important;
    border-radius: 16px !important;
    color: #f1f5f9 !important;
    width: min(460px, 92vw) !important;
    max-width: 92vw !important;
    max-height: 72vh !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6) !important;
}}
[data-testid="stPopoverBody"] * {{ color: #f1f5f9; }}
[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] * {{ color: {MUT} !important; }}
.notif-title {{ font-weight: 800; font-size: 1.15rem; margin-bottom: 10px; color: #fff !important; }}
.notif-item {{
    display: flex; gap: 12px; align-items: flex-start;
    padding: 11px 12px; margin-bottom: 9px; border-radius: 12px;
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid {GRN};
    overflow-wrap: anywhere;
}}
.notif-item.new {{ background: rgba(239, 68, 68, 0.10); }}
.notif-ic {{ font-size: 1.4rem; line-height: 1.2; }}
.notif-main {{ min-width: 0; flex: 1; }}
.notif-h {{ font-weight: 800; font-size: 0.92rem; color: #fff !important; }}
.notif-new {{ background: {RED}; color: #fff !important; font-size: 0.64rem; font-weight: 800; padding: 2px 7px; border-radius: 99px; margin-left: 6px; }}
.notif-s {{ font-size: 0.78rem; color: #cbd5e1 !important; margin-top: 3px; }}
.notif-m {{ font-size: 0.72rem; color: {MUT} !important; margin-top: 3px; }}
.notif-empty {{ padding: 18px; text-align: center; color: #34d399 !important; font-weight: 700; }}

/* TOPBAR : aucun chevauchement */
.user-pill {{ max-width: 100%; white-space: nowrap; overflow: hidden; }}
.user-pill > div {{ min-width: 0; }}
.user-pill > div > div {{ overflow: hidden; text-overflow: ellipsis; }}
.avatar-img, .avatar-fallback {{ flex-shrink: 0; }}

/* PANNEAUX (conteneurs Streamlit) */
[class*="st-key-panel"] {{
    background: linear-gradient(170deg, #0c1d45 0%, {CARD} 100%);
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
    overflow: hidden;
}}
.panel {{ overflow: hidden; }}
.sec-title {{ font-weight: 800; font-size: 1.12rem; color: #f8fafc; margin: 6px 0 12px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.sec-title small {{ font-weight: 500; font-size: 0.8rem; color: {MUT}; }}

/* CARTES D'ALERTES + SCORE IA */
.score-card {{
    background: linear-gradient(170deg, #0c1d45 0%, {CARD} 100%);
    border: 1px solid {LINE}; border-left: 5px solid {GRN};
    border-radius: 14px; padding: 14px 16px; min-width: 0; height: 100%;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
}}
.score-card .sc-top {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
.score-card .sc-title {{ font-weight: 800; font-size: 0.95rem; }}
.score-card .sc-time {{ font-size: 0.75rem; color: {MUT}; white-space: nowrap; }}
.score-card .sc-ref {{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:{MUT}; overflow-wrap:anywhere; margin:4px 0 6px; }}
.score-card .sc-amt {{ font-size: 0.85rem; color:#e2e8f0; }}
.score-card .sc-lbl {{ font-size: 0.68rem; letter-spacing: 1px; color:{MUT}; font-weight:700; margin-top:10px; }}
.score-card .sc-score {{ font-family:'JetBrains Mono',monospace; font-weight:800; font-size:1.9rem; line-height:1.1; }}
.score-bar {{ height: 7px; background: rgba(255,255,255,0.09); border-radius: 99px; overflow:hidden; margin-top:6px; }}
.score-bar div {{ height:100%; border-radius:99px; }}
.score-chip {{
    display:inline-block; font-family:'JetBrains Mono',monospace; font-weight:800; font-size:0.9rem;
    padding: 3px 10px; border-radius: 10px; border: 1px solid; background: rgba(255,255,255,0.04); white-space: nowrap;
}}
.tbl-wrap {{ width:100%; overflow-x:auto; }}
.txn-table td, .txn-table th {{ white-space: nowrap; }}
@media (max-width: 900px) {{ .score-card .sc-score {{ font-size: 1.5rem; }} }}


/* NAVIGATION PRINCIPALE (toujours visible, sans dépendre de la barre latérale) */
[class*="st-key-nav_page"] [role="radiogroup"] {{ display: flex; flex-wrap: wrap; gap: 8px; }}
[class*="st-key-nav_page"] label {{
    background: rgba(15, 29, 66, 0.85); border: 1px solid {LINE}; border-radius: 99px;
    padding: 8px 18px; margin: 0 !important; cursor: pointer; transition: all .2s ease;
}}
[class*="st-key-nav_page"] label:hover {{ border-color: {GOLD}; background: rgba(240, 180, 41, 0.10); }}
[class*="st-key-nav_page"] label > div:first-child {{ display: none !important; }}
[class*="st-key-nav_page"] label p {{ font-weight: 700 !important; font-size: 0.92rem !important; color: #e2e8f0 !important; white-space: nowrap; }}
[class*="st-key-nav_page"] label:has(input:checked) {{
    background: linear-gradient(90deg, {GOLD} 0%, #ffd166 100%); border-color: {GOLD};
    box-shadow: 0 4px 18px rgba(240, 180, 41, 0.35);
}}
[class*="st-key-nav_page"] label:has(input:checked) p {{ color: #0f172a !important; font-weight: 800 !important; }}

/* FOOTER */
.footer-bar {{
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    padding-top: 16px; margin-top: 26px; border-top: 1px solid {LINE};
    color: {MUT}; font-size: 0.84rem;
}}

/* Libellés ciblés : blanc foncé, gras (le texte saisi dans les champs n'est pas modifié) */
.st-key-tx_client [data-testid="stWidgetLabel"], .st-key-tx_client [data-testid="stWidgetLabel"] *,
.st-key-tx_montant [data-testid="stWidgetLabel"], .st-key-tx_montant [data-testid="stWidgetLabel"] *,
.st-key-pay_carte [data-testid="stWidgetLabel"], .st-key-pay_carte [data-testid="stWidgetLabel"] *,
.st-key-pay_marchand [data-testid="stWidgetLabel"], .st-key-pay_marchand [data-testid="stWidgetLabel"] *,
.st-key-pay_montant [data-testid="stWidgetLabel"], .st-key-pay_montant [data-testid="stWidgetLabel"] *,
.lbl-blanc {{ color: #e8eefc !important; font-weight: 700 !important; font-size: 0.97rem !important; opacity: 1 !important; }}
.lbl-blanc {{ margin: 4px 0 6px; }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ CHARGEMENT MODÈLE
@st.cache_resource
def charger_modele(source):
    return joblib.load(source)


chemin_modele = BASE_DIR / "modele_fraude.joblib"
pack = charger_modele(chemin_modele) if chemin_modele.exists() else None

if pack is None:
    up = st.sidebar.file_uploader("📂 Charger modele_fraude.joblib", type="joblib")
    if up is not None:
        pack = charger_modele(up)

if pack is None:
    st.error("⚠️ Place `modele_fraude.joblib` à côté de `app.py` ou importe-le dans la barre latérale.")
    st.stop()

COLS = pack["colonnes"]
SCALER, MODELE = pack["scaler"], pack["modele"]


def scorer(df: pd.DataFrame) -> np.ndarray:
    """Probabilité de fraude. Amount DOIT être dans l'échelle du modèle (euros)."""
    X = df[COLS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(SCALER.mean_, index=COLS))
    X["Amount"] = X["Amount"].clip(lower=0)
    entree = X if hasattr(SCALER, "feature_names_in_") else X.to_numpy()
    return MODELE.predict_proba(SCALER.transform(entree))[:, 1]


def fmt_fcfa(x) -> str:
    return f"{float(x):,.0f}".replace(",", " ") + " FCFA"


def fmt_score(p) -> str:
    """Score IA en % lisible : jamais « 0.00 » (les très petits scores restent visibles)."""
    pct = float(p) * 100
    if pct >= 10:
        s = f"{pct:.1f}"
    elif pct >= 1:
        s = f"{pct:.2f}"
    elif pct >= 0.01:
        s = f"{pct:.3f}"
    else:
        return "< 0,01 %"
    return s.replace(".", ",") + " %"


JOURNAL_PATH = BASE_DIR / "journal_transactions.json"


def charger_journal() -> list:
    """Recharge les transactions analysées : elles survivent à l'actualisation de la page."""
    try:
        data = json.loads(JOURNAL_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def sauver_journal() -> None:
    try:
        JOURNAL_PATH.write_text(
            json.dumps(st.session_state.get("transactions_live", []), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except Exception:
        pass


def panneau(key: str):
    try:
        return st.container(key=key)
    except TypeError:
        return st.container(border=True)

if "transactions_live" not in st.session_state:
    st.session_state["transactions_live"] = charger_journal()
if "dernier_resultat_unitaire" not in st.session_state:
    st.session_state["dernier_resultat_unitaire"] = None
if "seuil_montant_alerte" not in st.session_state:
    st.session_state["seuil_montant_alerte"] = 1_000_000.0


# ------------------------------------------------------------------ NIVEAUX DE RISQUE (source unique)
def seuils():
    """(seuil Élevé, seuil Moyen) – modifiables dans la page Paramètres."""
    eleve = float(st.session_state.get("seuil_eleve", pack.get("seuil", 0.5)))
    moyen = float(st.session_state.get("seuil_moyen", 0.05))
    return eleve, min(moyen, eleve)


def planchers():
    """(plancher « montant négligeable », plancher « impact financier ») en FCFA."""
    nul = float(st.session_state.get("plancher_nul", 500.0))
    impact = float(st.session_state.get("plancher_impact", 5000.0))
    return nul, max(impact, nul)


def classer(probas, montants_fcfa):
    """Niveau de risque = score du modèle ET impact financier.
    - montant < plancher négligeable  -> au mieux « Faible » (ex. 0 FCFA : aucun enjeu financier)
    - montant < plancher d'impact     -> au pire  « Moyen »  (surveillance, pas de blocage)
    - sinon                           -> niveau donné par le score du modèle."""
    eleve, moyen = seuils()
    nul, impact = planchers()
    p = np.asarray(probas, dtype=float)
    m = np.asarray(montants_fcfa, dtype=float)
    brut = np.where(p >= eleve, 0, np.where(p >= moyen, 1, 2))      # 0 Élevé, 1 Moyen, 2 Faible
    plafond = np.where(m < nul, 2, np.where(m < impact, 1, 0))
    ordre = np.maximum(brut, plafond)
    motif = np.where(
        plafond > brut,
        np.where(plafond == 2, "Montant négligeable", "Montant faible (plafonné)"),
        np.where(brut == 0, "Score élevé", np.where(brut == 1, "Score intermédiaire", "Score faible")),
    )
    return np.array(NIVEAUX, dtype=object)[ordre], motif


def niveau_de(p: float, montant_fcfa: float) -> str:
    return str(classer([p], [montant_fcfa])[0][0])


def analyser_lot(brut: pd.DataFrame, unite: str = "EUR") -> pd.DataFrame:
    """Calcule score + rang + montant FCFA. `unite` = unité de la colonne Amount fournie."""
    df = brut.copy()
    montant = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).clip(lower=0)
    if unite == "FCFA":
        df["Montant (FCFA)"] = montant.round(0)
        df["Amount"] = montant / TAUX_EUR_FCFA          # retour à l'échelle du modèle (€)
    else:
        df["Amount"] = montant
        df["Montant (FCFA)"] = (montant * TAUX_EUR_FCFA).round(0)
    df["Probabilité"] = scorer(df)
    df["Rang"] = df["Probabilité"].rank(ascending=False, method="first").astype(int)
    return df


def finaliser(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute Niveau + Motif, puis classe : Élevé d'abord, puis Moyen, puis Faible (score décroissant)."""
    df = df.copy()
    df["Niveau"], df["Motif"] = classer(df["Probabilité"], df["Montant (FCFA)"])
    ordre_niv = df["Niveau"].map({n: i for i, n in enumerate(NIVEAUX)}).to_numpy()
    tri = np.lexsort((-df["Probabilité"].to_numpy(), ordre_niv))
    rang = np.empty(len(df), dtype=int)
    rang[tri] = np.arange(1, len(df) + 1)
    df["Rang"] = rang
    return df


def donnees_demo(n=2500, seed=42) -> pd.DataFrame:
    """Transactions SYNTHÉTIQUES calibrées sur l'échelle réelle du modèle
    (Amount en €, V1..V28 avec leurs vrais écarts-types) : niveaux de risque variés,
    indépendants du montant (le coefficient de Amount dans le modèle est très faible)."""
    rng = np.random.default_rng(seed)
    S = SCALER.scale_[:28]
    df = pd.DataFrame(rng.normal(size=(n, 28)) * S, columns=[f"V{i}" for i in range(1, 29)])
    # médiane ≈ 22 €, moyenne ≈ 88 € (≈ 14 000 FCFA / 58 000 FCFA), queue lourde comme le dataset réel
    df["Amount"] = np.exp(rng.normal(3.09, 1.6, n)).clip(3, 25_000).round(2)
    df["Class"] = 0
    # ~14 % de transactions atypiques d'intensité graduelle -> scores Faible / Moyen / Élevé
    nb = int(n * 0.14)
    ib = rng.choice(n, nb, replace=False)
    inten = rng.beta(1.2, 2.2, nb)
    for c, signe in {"V14": -1, "V10": -1, "V12": -1, "V4": 1, "V11": 1, "V9": -1, "V16": -1}.items():
        k = int(c[1:]) - 1
        df.loc[ib, c] += signe * inten * rng.uniform(1.5, 3.2, nb) * S[k] * 2.2
    df.loc[ib[inten > 0.6], "Class"] = 1
    df.insert(0, "Client", [f"Client_{rng.integers(1000, 9999)}" for _ in range(n)])
    df.insert(0, "ID", [f"#TXN-{78400 + i}" for i in range(n)])
    return df


@st.cache_data(show_spinner=False)
def lot_demo_score() -> pd.DataFrame:
    return analyser_lot(donnees_demo(), "EUR")


def alertes_recentes(df: pd.DataFrame, n: int = 8, niveau: str = "Tous") -> pd.DataFrame:
    """Alertes (Élevé + Moyen) les plus récentes d'abord : flux réaliste avec niveaux variés."""
    al = df[df["Niveau"] != "Faible"]
    if niveau != "Tous":
        al = al[al["Niveau"] == niveau]
    return al.iloc[::-1].head(n)


def lot_courant() -> pd.DataFrame:
    """Dernier lot analysé (page Analyse IA) ou lot de démonstration, avec niveaux à jour."""
    df = st.session_state.get("resultat")
    if df is None:
        df = lot_demo_score()
    return finaliser(df)


def style_plotly(fig, height=290):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Plus Jakarta Sans"),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=13, color="#f8fafc")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False, tickfont=dict(color="#cbd5e1"), title_font=dict(color="#cbd5e1")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False, tickfont=dict(color="#cbd5e1"), title_font=dict(color="#cbd5e1")),
        hoverlabel=dict(bgcolor="#0a1736", bordercolor=GOLD, font=dict(color="#ffffff", size=13)),
    )
    return fig


def sparkline_svg(points, color):
    w, h = 110, 32
    mn, mx = min(points), max(points)
    span = (mx - mn) or 1
    coords = " ".join(
        f"{i * (w / (len(points) - 1)):.1f},{h - ((v - mn) / span) * (h - 6) - 3:.1f}"
        for i, v in enumerate(points)
    )
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="overflow:visible">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" points="{coords}"/>'
        f"</svg>"
    )


def render_kpi(theme, icon, icon_bg, title, value, delta, delta_col, pct, bar_col, spark_pts):
    spark = sparkline_svg(spark_pts, bar_col)
    return f"""
    <div class="kpi-card {theme}">
        <div class="kpi-top">
            <div class="kpi-icon" style="background:{icon_bg}; color:{bar_col};">{icon}</div>
            <div class="kpi-title">{title}</div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:flex-end;">
            <div>
                <span class="kpi-val">{value}</span>
                <span class="kpi-delta" style="color:{delta_col};">{delta}</span>
            </div>
            {spark}
        </div>
        <div class="kpi-bar"><div style="width:{pct}%; background:{bar_col};"></div></div>
    </div>
    """



# ------------------------------------------------------------------ RAPPORT PDF (1 clic)
def generer_rapport_pdf() -> bytes:
    """Rapport de supervision PDF : indicateurs, graphiques, alertes, transactions, limites."""
    from io import BytesIO
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.legends import Legend

    NAVY, C_GOLD = colors.HexColor("#0a1736"), colors.HexColor("#f0b429")
    C_GRN, C_RED, C_YEL, C_BLU = colors.HexColor("#10b981"), colors.HexColor("#ef4444"), colors.HexColor("#f59e0b"), colors.HexColor("#3b82f6")
    GRIS, CLAIR, TXT = colors.HexColor("#64748b"), colors.HexColor("#f1f5f9"), colors.HexColor("#0f172a")
    coul = {"Élevé": C_RED, "Moyen": C_YEL, "Faible": C_GRN}

    s_norm = ParagraphStyle("n", fontName="Helvetica", fontSize=9, leading=13, textColor=TXT)
    s_small = ParagraphStyle("s", parent=s_norm, fontSize=7.5, leading=10)
    s_h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=NAVY, spaceBefore=14, spaceAfter=6, keepWithNext=1)
    s_kv = ParagraphStyle("kv", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=NAVY, alignment=TA_CENTER)
    s_kl = ParagraphStyle("kl", fontName="Helvetica", fontSize=7.5, leading=10, textColor=GRIS, alignment=TA_CENTER)
    s_th = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=colors.white)
    s_t1 = ParagraphStyle("t1", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.white)
    s_t2 = ParagraphStyle("t2", fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#cbd5e1"))

    def P(txt, st_=None):
        return Paragraph(escape(str(txt).replace("·", "-").replace("≥", ">=")), st_ or s_small)

    def titre(txt):
        return Paragraph(escape(txt), s_h)

    def kpi_row(items):
        cells = [[Paragraph(escape(str(v)), s_kv), Paragraph(escape(l), s_kl)] for v, l in items]
        w = 17.6 * cm / len(items)
        tb = Table([[[c[0], c[1]] for c in cells]], colWidths=[w] * len(items))
        tb.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CLAIR), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("LINEAFTER", (0, 0), (-2, -1), 0.5, colors.white), ("LINEABOVE", (0, 0), (-1, 0), 2.5, C_GOLD),
            ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return tb

    def tableau(entetes, lignes, widths, niv_col=None, niveaux=None):
        data = [[Paragraph(escape(h), s_th) for h in entetes]] + lignes
        tb = Table(data, colWidths=widths, repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLAIR]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if niv_col is not None and niveaux:
            for i, n in enumerate(niveaux, start=1):
                style.append(("BACKGROUND", (niv_col, i), (niv_col, i), coul.get(n, GRIS)))
                style.append(("TEXTCOLOR", (niv_col, i), (niv_col, i), colors.white))
        tb.setStyle(TableStyle(style))
        return tb

    def decor(c, d):
        w, h = A4
        c.saveState()
        c.setFillColor(NAVY); c.rect(0, h - 1.1 * cm, w, 1.1 * cm, stroke=0, fill=1)
        c.setFillColor(C_GOLD); c.rect(0, h - 1.18 * cm, w, 0.08 * cm, stroke=0, fill=1)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 8.5)
        c.drawString(1.5 * cm, h - 0.72 * cm, "FRAUDSHIELD AI  |  Rapport de supervision")
        c.setFont("Helvetica", 8); c.drawRightString(w - 1.5 * cm, h - 0.72 * cm, datetime.now().strftime("%d/%m/%Y %H:%M"))
        c.setFillColor(GRIS); c.setFont("Helvetica", 7.5)
        c.drawString(1.5 * cm, 0.9 * cm, "FraudShield AI - Projet de démonstration (données publiques) - Ibrahima DIENG, Junior Data Scientist & IA")
        c.drawRightString(w - 1.5 * cm, 0.9 * cm, f"Page {d.page}")
        c.restoreState()

    lot = lot_courant()
    n_lot = len(lot)
    comptes = lot["Niveau"].value_counts().reindex(NIVEAUX).fillna(0).astype(int)
    journal = st.session_state.get("transactions_live", [])
    eleve_s, moyen_s = seuils()
    vigilance = float(st.session_state.get("seuil_montant_alerte", 1_000_000.0))

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.9 * cm, bottomMargin=1.6 * cm,
                            title="FraudShield AI - Rapport de supervision", author="Ibrahima DIENG", subject="Détection de fraude bancaire")
    story = []

    # --- Bandeau de titre
    bandeau = Table([[[Paragraph("Rapport de supervision antifraude", s_t1),
                       Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} &nbsp;|&nbsp; Auteur : Ibrahima DIENG, Junior Data Scientist &amp; IA", s_t2)]]],
                    colWidths=[17.6 * cm])
    bandeau.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LINEBELOW", (0, 0), (-1, -1), 3, C_GOLD),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 16), ("TOPPADDING", (0, 0), (-1, -1), 16), ("BOTTOMPADDING", (0, 0), (-1, -1), 16)]))
    story += [bandeau, Spacer(1, 8)]

    # --- 1. Indicateurs du modèle
    ret = METRIQUES.iloc[1]
    story.append(titre("1. Performance du modèle retenu"))
    story.append(kpi_row([(f"{TOTAL:,}".replace(",", " "), "Transactions étudiées"), (f"{FRAUDES}", "Fraudes réelles (jeu de données)"),
                          (f"{ret['Rappel@200'] * 100:.1f} %", "Fraudes captées dans le Top 200"), (f"{ret['PR-AUC']:.3f}", "PR-AUC"), (f"{ret['ROC-AUC']:.3f}", "ROC-AUC")]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(escape(f"Évaluation hors-échantillon sur une période future : {TEST_N:,} transactions dont {TEST_F} fraudes. "
                                  "En n'auditant que les 200 alertes les plus prioritaires, l'équipe retrouve plus de huit fraudes sur dix.".replace(",", " ")), s_norm))

    # --- 2. Activité de la session
    story.append(titre("2. Activité analysée (session)"))
    if journal:
        probs = [float(x.get("Probabilité", 0)) for x in journal]
        n_al = sum(1 for x in journal if x.get("Niveau") in ("Élevé", "Moyen"))
        mt_tot = sum(float(x.get("Montant (FCFA)", 0)) for x in journal)
        mt_risque = sum(float(x.get("Montant (FCFA)", 0)) for x in journal if x.get("Niveau") == "Élevé")
        story.append(kpi_row([(f"{len(journal)}", "Transactions analysées"), (f"{n_al}", "Alertes (Élevé + Moyen)"),
                              (fmt_fcfa(mt_risque), "Montant à risque élevé"), (fmt_score(sum(probs) / len(probs)), "Score IA moyen"), (fmt_fcfa(mt_tot), "Montant total analysé")]))
    else:
        story.append(Paragraph("Aucune transaction n'a encore été analysée dans le simulateur. Les indicateurs d'activité apparaîtront ici après les premières analyses.", s_norm))

    # --- 3. Graphiques
    story.append(titre("3. Répartition des risques et capture des fraudes"))
    d = Drawing(17.6 * cm, 7.4 * cm)
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 0.4 * cm, 0.9 * cm, 5.2 * cm, 5.2 * cm
    pie.data = [max(int(comptes[n]), 0) or 0.0001 for n in NIVEAUX]
    pie.labels = None
    for i, n in enumerate(NIVEAUX):
        pie.slices[i].fillColor = coul[n]; pie.slices[i].strokeColor = colors.white; pie.slices[i].strokeWidth = 1
    d.add(pie)
    d.add(String(0.4 * cm, 6.9 * cm, "Répartition des risques (lot analysé)", fontName="Helvetica-Bold", fontSize=8.5, fillColor=NAVY))
    for i, n in enumerate(NIVEAUX):
        y = 4.2 * cm - i * 0.8 * cm
        d.add(String(6.3 * cm, y, f"{n} : {int(comptes[n])} ({comptes[n] / max(n_lot, 1) * 100:.1f} %)", fontName="Helvetica", fontSize=8, fillColor=TXT))
        d.add(Rect(5.7 * cm, y - 1, 0.4 * cm, 0.3 * cm, fillColor=coul[n], strokeColor=None))
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 11.6 * cm, 2.1 * cm, 5.6 * cm, 4.2 * cm
    bc.data = [[METRIQUES.iloc[i]["Rappel@50"] * 100, METRIQUES.iloc[i]["Rappel@200"] * 100, METRIQUES.iloc[i]["Rappel@500"] * 100] for i in range(len(METRIQUES))]
    bc.categoryAxis.categoryNames = ["Top 50", "Top 200", "Top 500"]
    bc.categoryAxis.labels.fontSize = 7
    bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep = 0, 100, 25
    bc.valueAxis.labels.fontSize = 7
    bc.barLabelFormat = "%.0f"; bc.barLabels.fontSize = 6; bc.barLabels.nudge = 6
    bc.barWidth = 6; bc.groupSpacing = 8
    for i, c_ in enumerate([GRIS, C_GOLD, C_BLU][: len(METRIQUES)]):
        bc.bars[i].fillColor = c_
    d.add(bc)
    d.add(String(11.2 * cm, 6.9 * cm, "Fraudes interceptées (%) selon le nombre d'alertes", fontName="Helvetica-Bold", fontSize=8.5, fillColor=NAVY))
    lg = Legend()
    lg.x, lg.y, lg.fontSize, lg.alignment = 11.6 * cm, 0.1 * cm, 6.5, "right"
    lg.colorNamePairs = [(c_, str(m).replace(" (Retenu)", "")) for c_, m in zip([GRIS, C_GOLD, C_BLU], METRIQUES["Modèle"])]
    lg.columnMaximum = 3; lg.deltax = 0; lg.deltay = 9; lg.dxTextSpace = 3; lg.boxAnchor = "sw"
    d.add(lg)
    story.append(d)

    # --- 4. Alertes
    story.append(titre("4. Alertes déclenchées"))
    alertes = [x for x in journal if x.get("Niveau") in ("Élevé", "Moyen")][-15:][::-1]
    if alertes:
        lignes = [[P(x.get("Date & Heure", "—")), P(x.get("Client", "—")), P(fmt_fcfa(x.get("Montant (FCFA)", 0))), P(fmt_score(x.get("Probabilité", 0))),
                   P(x.get("Niveau", "")), P(x.get("Motif", ""))] for x in alertes]
        for l in lignes:
            l[4].style = ParagraphStyle("lv", parent=s_small, textColor=colors.white, fontName="Helvetica-Bold")
        story.append(tableau(["Date & heure", "Client", "Montant", "Score IA", "Niveau", "Motif"], lignes,
                             [3.1 * cm, 2.6 * cm, 3.2 * cm, 2.0 * cm, 1.7 * cm, 5.0 * cm], niv_col=4, niveaux=[x.get("Niveau") for x in alertes]))
    else:
        story.append(Paragraph("Aucune alerte déclenchée pour le moment.", s_norm))

    # --- 5. Dernières transactions
    story.append(titre("5. Dernières transactions analysées"))
    if journal:
        recentes = journal[-15:][::-1]
        lignes = [[P(x.get("ID", "—")), P(x.get("Client", "—")), P(fmt_fcfa(x.get("Montant (FCFA)", 0))), P(fmt_score(x.get("Probabilité", 0))),
                   P(x.get("Niveau", "")), P(x.get("Action", ""))] for x in recentes]
        for l in lignes:
            l[4].style = ParagraphStyle("lv2", parent=s_small, textColor=colors.white, fontName="Helvetica-Bold")
        story.append(tableau(["ID transaction", "Client", "Montant", "Score IA", "Niveau", "Action recommandée"], lignes,
                             [3.6 * cm, 2.3 * cm, 2.9 * cm, 1.9 * cm, 1.6 * cm, 5.3 * cm], niv_col=4, niveaux=[x.get("Niveau") for x in recentes]))
    else:
        story.append(Paragraph("Aucune transaction analysée pour le moment.", s_norm))

    # --- 6. Méthodologie et limites
    story.append(KeepTogether([
        titre("6. Méthodologie et limites"),
        Paragraph(escape(
            f"Modèle : régression logistique avec StandardScaler, entraînée sur le jeu public « Credit Card Fraud » ({TOTAL:,} transactions dont {FRAUDES} fraudes, "
            "validation chronologique). ".replace(",", " ") +
            f"Seuils de risque actuels : Élevé à partir de {eleve_s * 100:.1f} % de score, Moyen à partir de {moyen_s * 100:.1f} %. "
            f"Règles métier : alerte jaune dès {fmt_fcfa(vigilance)}, alerte rouge dès 5 000 000 FCFA."), s_norm),
        Spacer(1, 4),
        Paragraph(escape(
            "Limites : FraudShield AI est un projet de démonstration. Le simulateur ne saisit que l'identifiant client et le montant ; les variables anonymisées V1 à V28 "
            "sont fixées à leurs moyennes d'apprentissage. Les scores doivent donc être interprétés comme illustratifs et non comme une décision bancaire réelle."), s_norm),
    ]))

    doc.build(story, onFirstPage=decor, onLaterPages=decor)
    return buf.getvalue()


def bouton_rapport_pdf(key: str) -> None:
    """Bouton de téléchargement du rapport PDF (génération à la volée)."""
    try:
        pdf = generer_rapport_pdf()
    except ModuleNotFoundError:
        st.info("Pour activer le rapport PDF, installez la bibliothèque : `pip install reportlab` (et ajoutez `reportlab` à requirements.txt).")
        return
    except Exception as exc:
        st.warning(f"Rapport PDF momentanément indisponible : {exc}")
        return
    st.download_button(
        "📄 Télécharger le rapport PDF",
        data=pdf,
        file_name=f"FraudShield_Rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        key=key,
        use_container_width=True,
    )


# ------------------------------------------------------------------ MODULE PAIEMENT CARTE (carte + score + 3-D Secure + reçu PDF)
import hashlib
import hmac
import os
import re
import secrets
import time
from io import BytesIO

LARGE = {"width": "stretch"} if NEW_ST else {"use_container_width": True}
SECRET_SIGNATURE = os.environ.get("FRAUDSHIELD_SECRET", "fraudshield-demo-key")   # clé HMAC des reçus (à mettre en variable d'env. en production)

# Cartes de TEST publiques (numéros valides Luhn, aucun compte réel) + clients fictifs
CARTES_DEMO = {
    "Aminata DIOP · Visa ·••• 1111": dict(client="CLT-0001", titulaire="AMINATA DIOP", numero="4111111111111111", exp="12/29", tel="+221 77 000 00 01"),
    "Mamadou SALL · Mastercard ·••• 4444": dict(client="CLT-0002", titulaire="MAMADOU SALL", numero="5555555555554444", exp="09/28", tel="+221 78 000 00 02"),
    "Fatou NDIAYE · Visa ·••• 5556": dict(client="CLT-0003", titulaire="FATOU NDIAYE", numero="4000056655665556", exp="03/30", tel="+221 76 000 00 03"),
}
CARTE_LIBRE = "➕ Saisir une autre carte"
MARCHANDS = ["Marché Central — Thiès", "E-Boutique Dakar", "Agence de Voyage Teranga", "Station Service Thiès", "Pharmacie Moderne", "Hôtel Teranga Palace"]


def luhn_ok(numero: str) -> bool:
    chiffres = [int(c) for c in numero if c.isdigit()]
    if not 13 <= len(chiffres) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(chiffres)):
        if i % 2 == 1:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def marque_carte(numero: str) -> str:
    n = re.sub(r"\D", "", numero)
    if n.startswith("4"):
        return "VISA"
    if re.match(r"^(5[1-5]|2[2-7])", n):
        return "MASTERCARD"
    return "CARTE"


def masquer_carte(numero: str) -> str:
    n = re.sub(r"\D", "", numero)
    return f"{n[:4]} •••• •••• {n[-4:]}" if len(n) >= 12 else n


def expiration_ok(exp: str) -> bool:
    m = re.match(r"^(0[1-9]|1[0-2])/(\d{2})$", exp.strip())
    if not m:
        return False
    mois, an = int(m.group(1)), 2000 + int(m.group(2))
    now = datetime.now()
    return (an, mois) >= (now.year, now.month)


def carte_html(numero_aff: str, titulaire: str, exp: str, marque: str) -> str:
    logo = {
        "VISA": '<span class="bk-visa">VISA</span>',
        "MASTERCARD": '<span class="bk-mc"><i></i><i></i></span>',
    }.get(marque, '<span class="bk-visa" style="font-size:15px">CARTE</span>')
    return (
        '<div translate="no" class="bk-wrap"><div class="bkcard"><div class="bk-in">'
        '<div class="bk-top"><span class="bk-bank">FRAUDSHIELD <b>SECURE</b></span><span class="bk-nfc">))) </span></div>'
        '<div class="bk-chip"></div>'
        f'<div class="bk-num">{numero_aff}</div>'
        '<div class="bk-bot"><div><div class="bk-lbl">TITULAIRE</div>'
        f'<div class="bk-val">{titulaire or "NOM DU TITULAIRE"}</div></div>'
        f'<div><div class="bk-lbl">EXPIRE</div><div class="bk-val">{exp or "MM/AA"}</div></div>{logo}</div>'
        '</div></div></div>'
    )


def numero_affichage_live(chiffres: str) -> str:
    d = re.sub(r"\D", "", chiffres)[:19]
    if len(d) >= 13:
        return masquer_carte(d)
    d = d.ljust(16, "•")
    return " ".join(d[i:i + 4] for i in range(0, 16, 4))


CSS_PAIEMENT = f"""
<style>
.bk-wrap {{ perspective: 1000px; margin: 4px 0 14px; }}
.bkcard {{ position: relative; width: 100%; max-width: 410px; aspect-ratio: 1.586; border-radius: 22px; overflow: hidden; color: #fff;
  background: linear-gradient(135deg, #0b2a6f 0%, #1d4ed8 48%, #071330 100%);
  box-shadow: 0 24px 55px rgba(0,0,0,.6), inset 0 0 0 1px rgba(255,255,255,.14); transform: rotateY(-7deg) rotateX(4deg);
  transition: transform .5s ease; animation: bkfloat 5s ease-in-out infinite; }}
.bkcard:hover {{ transform: rotateY(0) rotateX(0) scale(1.02); }}
.bkcard::before {{ content: ""; position: absolute; inset: -60%; background: conic-gradient(from 0deg, transparent 0 70%, rgba(240,180,41,.35) 82%, transparent 94%); animation: bkspin 7s linear infinite; }}
.bkcard::after {{ content: ""; position: absolute; right: -40px; top: -40px; width: 190px; height: 190px; border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,.16), transparent 70%); }}
.bk-in {{ position: relative; z-index: 2; padding: 20px 24px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box; }}
.bk-top {{ display: flex; justify-content: space-between; align-items: center; }}
.bk-bank {{ font-weight: 800; letter-spacing: 1px; font-size: 14px; }} .bk-bank b {{ color: {GOLD}; }}
.bk-nfc {{ font-size: 18px; opacity: .8; letter-spacing: -2px; }}
.bk-chip {{ width: 48px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #f7d774, #c9962b 55%, #f7d774); box-shadow: inset 0 0 0 1px rgba(0,0,0,.25); position: relative; }}
.bk-chip::after {{ content: ""; position: absolute; inset: 6px 10px; border: 1px solid rgba(0,0,0,.25); border-radius: 4px; }}
.bk-num {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: clamp(15px, 2.5vw, 22px); letter-spacing: 2px; text-shadow: 0 2px 6px rgba(0,0,0,.5); }}
.bk-bot {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; }}
.bk-lbl {{ font-size: 8px; letter-spacing: 1.5px; opacity: .65; }} .bk-val {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; }}
.bk-visa {{ font-style: italic; font-weight: 800; font-size: 24px; letter-spacing: -1px; }}
.bk-mc {{ display: inline-flex; }} .bk-mc i {{ width: 28px; height: 28px; border-radius: 50%; background: #eb001b; }}
.bk-mc i + i {{ background: #f79e1b; margin-left: -11px; mix-blend-mode: screen; opacity: .95; }}
@keyframes bkspin {{ to {{ transform: rotate(360deg); }} }}
@keyframes bkfloat {{ 0%,100% {{ translate: 0 0; }} 50% {{ translate: 0 -6px; }} }}

.pay-steps {{ display: flex; gap: 8px; margin: 6px 0 14px; }}
.pay-step {{ flex: 1; text-align: center; padding: 10px 4px; border-radius: 12px; font-size: 12px; font-weight: 700; border: 1px solid {LINE}; background: rgba(255,255,255,.03); color: {MUT}; }}
.pay-step.done {{ border-color: {GRN}; color: {GRN}; background: rgba(16,185,129,.10); }}
.pay-step.now {{ border-color: {GOLD}; color: {GOLD}; background: rgba(240,180,41,.10); animation: paypulse 1s ease-in-out infinite; }}
@keyframes paypulse {{ 50% {{ box-shadow: 0 0 16px rgba(240,180,41,.55); }} }}

.pay-hero {{ border-radius: 20px; padding: 22px; border: 1px solid var(--c); background: linear-gradient(160deg, color-mix(in srgb, var(--c) 18%, #0a1736), #0a1736); }}
.pay-hero h2 {{ margin: 8px 0 2px; font-size: 1.5rem; color: var(--c); }}
.pay-amount {{ font-family: 'JetBrains Mono', monospace; font-size: 2.1rem; font-weight: 800; }}
.pay-check {{ width: 74px; height: 74px; }}
.pay-check circle {{ fill: none; stroke: var(--c); stroke-width: 4; stroke-dasharray: 190; stroke-dashoffset: 190; animation: paydraw .8s ease forwards; }}
.pay-check path {{ fill: none; stroke: var(--c); stroke-width: 5; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 60; stroke-dashoffset: 60; animation: paydraw .5s .6s ease forwards; }}
@keyframes paydraw {{ to {{ stroke-dashoffset: 0; }} }}
.pay-kv {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px 18px; margin-top: 14px; }}
.pay-kv div span {{ display: block; font-size: 11px; color: {MUT}; text-transform: uppercase; letter-spacing: .8px; }}
.pay-kv div b {{ font-size: 14px; }}
.pay-code {{ font-family: 'JetBrains Mono', monospace; letter-spacing: 4px; color: {GOLD}; font-size: 1.6rem; font-weight: 800; }}

.sms-phone {{ max-width: 360px; border-radius: 26px; padding: 14px 14px 18px; background: #0b1220; border: 2px solid #26365e; box-shadow: 0 14px 34px rgba(0,0,0,.5); }}
.sms-head {{ text-align: center; font-size: 11px; color: {MUT}; margin-bottom: 10px; }}
.sms-bubble {{ background: #1e3a7a; border-radius: 16px 16px 16px 4px; padding: 12px 14px; font-size: 13px; line-height: 1.5; animation: smsin .6s ease; }}
.sms-bubble.alert {{ background: #5b1620; }}
.sms-time {{ text-align: right; font-size: 10px; color: {MUT}; margin-top: 6px; }}
@keyframes smsin {{ from {{ opacity: 0; transform: translateY(14px) scale(.96); }} to {{ opacity: 1; transform: none; }} }}
.ring {{ width: 120px; height: 120px; transform: rotate(-90deg); }}
.ring .bg {{ fill: none; stroke: rgba(255,255,255,.08); stroke-width: 10; }}
.ring .fg {{ fill: none; stroke: var(--c); stroke-width: 10; stroke-linecap: round; stroke-dasharray: 327; stroke-dashoffset: var(--off); animation: ringin 1.3s ease; }}
@keyframes ringin {{ from {{ stroke-dashoffset: 327; }} }}
</style>
"""


def evaluer_paiement(montant_fcfa: float) -> dict:
    """Même logique que le simulateur unitaire : score IA + niveau + règles métier de montant."""
    ligne = pd.DataFrame([dict(zip(COLS, SCALER.mean_))])
    ligne.loc[0, "Amount"] = float(montant_fcfa) / TAUX_EUR_FCFA
    p = float(scorer(ligne)[0])
    niv = niveau_de(p, float(montant_fcfa))
    vigilance = float(st.session_state.get("seuil_montant_alerte", 1_000_000.0))
    eleve = float(montant_fcfa) >= vigilance
    critique = float(montant_fcfa) >= 5_000_000.0
    if critique:
        niv = "Élevé"
    elif eleve and niv == "Faible":
        niv = "Moyen"
    motif = str(classer([p], [montant_fcfa])[1][0])
    if critique:
        motif += " · Montant ≥ 5 000 000 FCFA (règle métier)"
    elif eleve:
        motif += " · Montant supérieur au seuil de vigilance"
    action = {"Élevé": "Paiement bloqué — contrôle humain immédiat",
              "Moyen": "Authentification 3-D Secure (code SMS) requise",
              "Faible": "Paiement autorisé"}[niv]
    # Score de confiance client (0-100) = 100 x (1 - probabilité de fraude), plafonné selon le niveau décidé
    confiance = int(round(100 * (1 - p)))
    confiance = min(confiance, {"Faible": 99, "Moyen": 74, "Élevé": 29}[niv])
    return dict(p=p, niv=niv, motif=motif, action=action, eleve=eleve, critique=critique, confiance=confiance)


def signer(P: dict) -> str:
    msg = f"{P['ref']}|{P['horodatage']}|{P['carte']}|{P['montant']:.0f}|{P['auth'] or '-'}|{P['statut']}"
    return hmac.new(SECRET_SIGNATURE.encode(), msg.encode(), hashlib.sha256).hexdigest()[:24].upper()


def sms_texte(P: dict, banque: str) -> tuple[str, bool]:
    fin = P["carte"][-4:]
    if P["statut"] == "EFFECTUÉE":
        return (f"{banque} : Votre paiement de {fmt_fcfa(P['montant'])} chez {P['marchand']} par carte •••• {fin} a été EFFECTUÉ. "
                f"Code d'autorisation : {P['auth']}. Score de confiance : {P['confiance']}/100. Reçu disponible dans votre espace.", False)
    if P["statut"].startswith("REFUSÉE"):
        return (f"{banque} ALERTE : Paiement de {fmt_fcfa(P['montant'])} chez {P['marchand']} (carte •••• {fin}) BLOQUÉ par FraudShield AI "
                f"pour votre sécurité. Si c'est vous, contactez votre agence. Réf. {P['ref']}.", True)
    return (f"{banque} : Code 3-D Secure {P['otp']} pour valider {fmt_fcfa(P['montant'])} chez {P['marchand']}. "
            f"Ne le communiquez à personne.", False)


def _maj_journal(P: dict) -> None:
    """Crée ou met à jour l'entrée du journal d'audit (jamais le numéro complet de carte : PCI-DSS)."""
    entree = {
        "ID": P["ref"], "Date & Heure": P["horodatage"], "Client": f"{P['client']} · {P['titulaire'].title()}",
        "Montant (FCFA)": float(P["montant"]), "Probabilité": float(P["p"]), "Niveau": P["niv"], "Motif": P["motif"],
        "Action": P["action"], "Montant élevé": bool(P["eleve"]), "Alerte montant critique": bool(P["critique"]),
        "Notification non lue": P["niv"] in ("Élevé", "Moyen"),
        "Carte": P["carte"], "Marchand": P["marchand"], "Statut": P["statut"], "Code autorisation": P["auth"] or "",
    }
    journal = st.session_state["transactions_live"]
    for i, t in enumerate(journal):
        if t.get("ID") == P["ref"]:
            entree["Notification non lue"] = t.get("Notification non lue", entree["Notification non lue"])
            journal[i] = entree
            break
    else:
        journal.append(entree)
    sauver_journal()


def generer_recu_pdf(P: dict, banque: str) -> bytes:
    """Reçu bancaire PDF signé (HMAC) avec QR de vérification."""
    from reportlab.graphics import renderPDF
    from reportlab.graphics.barcode import qr
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas

    NAVY, OR, VERT, ROUGE = colors.HexColor("#0a1736"), colors.HexColor("#f0b429"), colors.HexColor("#10b981"), colors.HexColor("#ef4444")
    ok = P["statut"] == "EFFECTUÉE"
    couleur = VERT if ok else ROUGE
    W, H = A4
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"Reçu {P['ref']}")
    c.setAuthor("FraudShield AI")

    # Filigrane
    c.saveState()
    c.setFillColor(colors.HexColor("#0a1736"))
    c.setFillAlpha(0.045)
    c.setFont("Helvetica-Bold", 78)
    c.translate(W / 2, H / 2 - 1 * cm)
    c.rotate(35)
    c.drawCentredString(0, 0, "FRAUDSHIELD AI")
    c.restoreState()

    # En-tête
    c.setFillColor(NAVY)
    c.rect(0, H - 4.2 * cm, W, 4.2 * cm, fill=1, stroke=0)
    c.setFillColor(OR)
    c.rect(0, H - 4.35 * cm, W, 0.15 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(2 * cm, H - 2.0 * cm, banque.upper()[:34])
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#9db4e6"))
    c.drawString(2 * cm, H - 2.8 * cm, "REÇU DE TRANSACTION PAR CARTE BANCAIRE" if ok else "AVIS DE REFUS DE TRANSACTION")
    c.setFillColor(OR)
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(W - 2 * cm, H - 2.0 * cm, "FRAUDSHIELD AI")
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#9db4e6"))
    c.drawRightString(W - 2 * cm, H - 2.6 * cm, "Paiement sécurisé par intelligence artificielle")

    # Statut + montant
    y = H - 6.3 * cm
    c.setFillColor(couleur)
    c.roundRect(W / 2 - 4.8 * cm, y, 9.6 * cm, 1.1 * cm, 0.55 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(W / 2, y + 0.38 * cm, "TRANSACTION EFFECTUÉE" if ok else "TRANSACTION REFUSÉE")
    c.setFillColor(colors.HexColor("#64748b"))
    c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2, y - 0.9 * cm, "MONTANT DÉBITÉ" if ok else "MONTANT (NON DÉBITÉ)")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(W / 2, y - 2.1 * cm, fmt_fcfa(P["montant"]))

    # Détails
    auth3ds = {"Faible": "Non requise (risque faible)", "Moyen": "Validée par code SMS" if ok else "Non aboutie", "Élevé": "Non applicable"}[P["niv"]]
    lignes = [
        ("Référence", P["ref"]), ("Date et heure", P["horodatage"]), ("Client", f"{P['client']} - {P['titulaire'].title()}"),
        ("Carte bancaire", f"{P['marque']}  {P['carte']}"), ("Marchand", P["marchand"]),
        ("Code d'autorisation", P["auth"] or "—"), ("Authentification 3-D Secure", auth3ds),
    ]
    y0 = y - 3.6 * cm
    for i, (k, v) in enumerate(lignes):
        yy = y0 - i * 0.95 * cm
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica", 9.5)
        c.drawString(2 * cm, yy, k)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawRightString(W - 2 * cm, yy, str(v))
        c.setStrokeColor(colors.HexColor("#e2e8f0"))
        c.setLineWidth(0.5)
        c.line(2 * cm, yy - 0.3 * cm, W - 2 * cm, yy - 0.3 * cm)

    # Bloc anti-fraude
    by = y0 - len(lignes) * 0.95 * cm - 3.3 * cm
    c.setFillColor(colors.HexColor("#eef4ff"))
    c.setStrokeColor(colors.HexColor("#c7d7fb"))
    c.roundRect(2 * cm, by, W - 4 * cm, 3.0 * cm, 0.4 * cm, fill=1, stroke=1)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(2.5 * cm, by + 2.35 * cm, "ANALYSE ANTI-FRAUDE FRAUDSHIELD AI")
    cases = [("Score IA (probabilité de fraude)", fmt_score(P["p"])), ("Niveau de risque", P["niv"]), ("Score de confiance", f"{P['confiance']} / 100")]
    cw = (W - 5 * cm) / 3
    for i, (k, v) in enumerate(cases):
        x = 2.5 * cm + i * cw
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica", 8)
        c.drawString(x, by + 1.5 * cm, k)
        c.setFillColor({"Faible": VERT, "Moyen": colors.HexColor("#d97706"), "Élevé": ROUGE}[P["niv"]] if i else NAVY)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, by + 0.8 * cm, v)
    c.setFillColor(colors.HexColor("#475569"))
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2.5 * cm, by + 0.25 * cm, f"Décision : {P['action']}")

    # QR + signature
    sig = signer(P)
    d = Drawing(3.2 * cm, 3.2 * cm)
    w = qr.QrCodeWidget(f"FRAUDSHIELD|{P['ref']}|{P['montant']:.0f}|{P['statut']}|{sig}")
    x1, y1, x2, y2 = w.getBounds()
    d.scale(3.2 * cm / (x2 - x1), 3.2 * cm / (y2 - y1))
    d.add(w)
    renderPDF.draw(d, c, 2 * cm, 3.0 * cm)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(5.6 * cm, 5.5 * cm, "Authenticité du document")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(5.6 * cm, 4.85 * cm, "Signature numérique (HMAC-SHA256) :")
    c.setFont("Courier-Bold", 10)
    c.setFillColor(NAVY)
    c.drawString(5.6 * cm, 4.3 * cm, sig)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(5.6 * cm, 3.65 * cm, "Scannez le QR code pour vérifier ce reçu auprès de la banque.")

    # Pied de page
    c.setStrokeColor(OR)
    c.setLineWidth(1.2)
    c.line(2 * cm, 2.3 * cm, W - 2 * cm, 2.3 * cm)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawCentredString(W / 2, 1.8 * cm, "Document généré électroniquement, valable sans signature manuscrite. Le numéro complet de la carte n'est jamais conservé.")
    c.drawCentredString(W / 2, 1.4 * cm, "DÉMONSTRATION — FraudShield AI : aucune opération financière réelle n'a été effectuée.")
    c.showPage()
    c.save()
    return buf.getvalue()


def _etapes_html(actif: int, noms: list) -> str:
    cases = []
    for i, n in enumerate(noms):
        cls = "done" if i < actif else ("now" if i == actif else "")
        cases.append(f'<div class="pay-step {cls}">{"✓ " if i < actif else ""}{n}</div>')
    return f'<div translate="no" class="pay-steps">{"".join(cases)}</div>'


def _bulle_sms(P: dict, banque: str) -> None:
    texte, alerte = sms_texte(P, banque)
    st.markdown(
        '<div translate="no" class="sms-phone"><div class="sms-head">📱 SMS reçu par le client · '
        f'{P["tel"]}</div><div class="sms-bubble{" alert" if alerte else ""}">{texte}'
        f'<div class="sms-time">{datetime.now().strftime("%H:%M")} · FraudShield</div></div></div>',
        unsafe_allow_html=True,
    )


def _reset_paiement() -> None:
    st.session_state["paiement"] = None


def _preset_montant(v: float) -> None:
    st.session_state["pay_montant"] = float(v)


def _ecran_saisie(banque: str) -> None:
    c1, c2 = st.columns([1.05, 1.2], gap="large")
    with c1:
        choix = st.selectbox("💳 Carte du client", list(CARTES_DEMO) + [CARTE_LIBRE], key="pay_carte")
        if choix == CARTE_LIBRE:
            numero = st.text_input("Numéro de carte", placeholder="4111 1111 1111 1111", max_chars=23, key="pay_num")
            titulaire = st.text_input("Titulaire", placeholder="PRÉNOM NOM", max_chars=40, key="pay_titu").upper()
            a, b = st.columns(2)
            exp = a.text_input("Expiration (MM/AA)", placeholder="12/29", max_chars=5, key="pay_exp")
            cvv = b.text_input("Code CVV", type="password", max_chars=4, key="pay_cvv")
            client_id = st.text_input("Identifiant client", value="CLT-9999", max_chars=30, key="pay_cid")
            tel = "+221 70 000 00 00"
            chiffres = re.sub(r"\D", "", numero)
            st.markdown(carte_html(numero_affichage_live(numero), titulaire, exp, marque_carte(chiffres)), unsafe_allow_html=True)
        else:
            d = CARTES_DEMO[choix]
            numero, titulaire, exp, cvv, client_id, tel = d["numero"], d["titulaire"], d["exp"], "123", d["client"], d["tel"]
            chiffres = numero
            st.markdown(carte_html(masquer_carte(numero), titulaire, exp, marque_carte(numero)), unsafe_allow_html=True)
        st.caption("🔐 Le numéro complet et le CVV ne sont **jamais enregistrés** : seul le numéro masqué entre dans le journal d'audit (PCI-DSS).")

    with c2:
        marchand = st.selectbox("🏪 Marchand", MARCHANDS, key="pay_marchand")
        st.session_state.setdefault("pay_montant", 75_000.0)
        montant = st.number_input("💰 Montant du paiement (FCFA)", min_value=0.0, max_value=100_000_000.0, step=10_000.0, format="%.0f", key="pay_montant")
        st.markdown('<div translate="no" class="lbl-blanc">Scénarios de démonstration :</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.button("🟢 75 000", on_click=_preset_montant, args=(75_000,), key="pre1", **LARGE)
        b2.button("🟡 1 500 000", on_click=_preset_montant, args=(1_500_000,), key="pre2", **LARGE)
        b3.button("🔴 6 000 000", on_click=_preset_montant, args=(6_000_000,), key="pre3", **LARGE)
        payer = st.button("🔒 Payer en toute sécurité", type="primary", key="btn_payer", **LARGE)

    if not payer:
        return
    erreurs = []
    if choix == CARTE_LIBRE:
        if not luhn_ok(chiffres):
            erreurs.append("Numéro de carte invalide (contrôle Luhn).")
        if not titulaire.strip():
            erreurs.append("Nom du titulaire requis.")
        if not expiration_ok(exp):
            erreurs.append("Date d'expiration invalide ou dépassée (format MM/AA).")
        if not re.fullmatch(r"\d{3,4}", cvv or ""):
            erreurs.append("CVV invalide (3 ou 4 chiffres).")
    if montant <= 0:
        erreurs.append("Le montant doit être supérieur à 0.")
    if erreurs:
        for e in erreurs:
            st.error(e)
        return

    # Animation d'analyse en temps réel
    noms = ["Carte vérifiée", "Analyse IA", "Score de risque", "Décision"]
    zone = st.empty()
    for i in range(len(noms) + 1):
        zone.markdown(_etapes_html(i, noms), unsafe_allow_html=True)
        time.sleep(0.45)
    zone.empty()

    ev = evaluer_paiement(montant)
    maintenant_dt = datetime.now()
    P = dict(
        ref=f"RCP-{maintenant_dt.strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}",
        horodatage=maintenant_dt.strftime("%d/%m/%Y %H:%M:%S"),
        client=client_id.strip() or "CLT-0000", titulaire=titulaire.strip(), tel=tel,
        carte=masquer_carte(chiffres), marque=marque_carte(chiffres), exp=exp,
        marchand=marchand, montant=float(montant), tentatives=0, otp=None, auth=None, **ev,
    )
    if ev["niv"] == "Élevé":
        P["statut"], P["stage"] = "REFUSÉE (RISQUE ÉLEVÉ)", "resultat"
    elif ev["niv"] == "Moyen":
        P["statut"], P["stage"] = "EN ATTENTE 3-D SECURE", "otp"
        P["otp"] = f"{secrets.randbelow(10**6):06d}"
    else:
        P["statut"], P["stage"] = "EFFECTUÉE", "resultat"
        P["auth"] = secrets.token_hex(3).upper()
    st.session_state["paiement"] = P
    _maj_journal(P)
    if ev["niv"] in ("Élevé", "Moyen"):
        st.session_state["toast_alerte"] = f"Paiement {ev['niv']} · {P['client']} · {fmt_fcfa(montant)}"
    st.rerun()


def _ecran_otp(P: dict, banque: str) -> None:
    st.markdown(_etapes_html(2, ["Carte vérifiée", "Analyse IA", "3-D Secure", "Autorisation"]), unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, 1.2], gap="large")
    with c1:
        st.markdown(carte_html(P["carte"], P["titulaire"], P["exp"], P["marque"]), unsafe_allow_html=True)
        _bulle_sms(P, banque)
        st.caption("Mode démonstration : le code s'affiche ici. En production, il est envoyé par SMS au numéro enregistré du client.")
    with c2:
        st.markdown(
            f'<div translate="no" class="pay-hero" style="--c:{GOLD};"><div style="font-size:2rem;">🔐</div><h2>Vérification 3-D Secure</h2>'
            f'<div style="color:{MUT};">Le moteur IA a classé ce paiement en <b style="color:{GOLD};">risque moyen</b> ({fmt_score(P["p"])}). '
            f'Le client doit confirmer avec le code reçu par SMS.</div>'
            f'<div class="pay-kv"><div><span>Montant</span><b>{fmt_fcfa(P["montant"])}</b></div><div><span>Marchand</span><b>{P["marchand"]}</b></div></div></div>',
            unsafe_allow_html=True,
        )
        code = st.text_input("Code à 6 chiffres", max_chars=6, key="pay_otp", placeholder="••••••")
        v, a = st.columns(2)
        valider = v.button("✅ Valider le paiement", type="primary", key="btn_otp", **LARGE)
        annuler = a.button("✖ Annuler", key="btn_annul", **LARGE)
    if annuler:
        P.update(statut="REFUSÉE (ANNULÉE PAR LE CLIENT)", stage="resultat")
        _maj_journal(P)
        st.rerun()
    if valider:
        if code.strip() == P["otp"]:
            P.update(statut="EFFECTUÉE", stage="resultat", auth=secrets.token_hex(3).upper(), authentifie=True)
            _maj_journal(P)
            st.rerun()
        P["tentatives"] += 1
        if P["tentatives"] >= 3:
            P.update(statut="REFUSÉE (3-D SECURE)", stage="resultat")
            _maj_journal(P)
            st.rerun()
        st.error(f"Code incorrect. Tentatives restantes : {3 - P['tentatives']}")


def _ecran_resultat(P: dict, banque: str) -> None:
    ok = P["statut"] == "EFFECTUÉE"
    coul = GRN if ok else RED
    c1, c2 = st.columns([1.05, 1.2], gap="large")
    with c1:
        st.markdown(carte_html(P["carte"], P["titulaire"], P["exp"], P["marque"]), unsafe_allow_html=True)
        _bulle_sms(P, banque)
    with c2:
        icone = ('<svg class="pay-check" viewBox="0 0 72 72"><circle cx="36" cy="36" r="30"/><path d="M22 37 L32 47 L51 26"/></svg>' if ok else
                 '<svg class="pay-check" viewBox="0 0 72 72"><circle cx="36" cy="36" r="30"/><path d="M25 25 L47 47 M47 25 L25 47"/></svg>')
        titre = "Transaction effectuée" if ok else "Transaction refusée"
        off = 327 * (1 - P["confiance"] / 100)
        st.markdown(
            f'<div translate="no" class="pay-hero" style="--c:{coul};"><div style="display:flex;gap:18px;align-items:center;justify-content:space-between;">'
            f'<div>{icone}<h2>{titre}</h2><div class="pay-amount">{fmt_fcfa(P["montant"])}</div>'
            f'<div style="color:{MUT};">{P["marchand"]}</div></div>'
            f'<div style="text-align:center;"><svg class="ring" viewBox="0 0 120 120" style="--off:{off:.1f};"><circle class="bg" cx="60" cy="60" r="52"/><circle class="fg" cx="60" cy="60" r="52"/></svg>'
            f'<div style="margin-top:-78px;font-family:JetBrains Mono,monospace;font-size:1.7rem;font-weight:800;color:{coul};">{P["confiance"]}</div>'
            f'<div style="margin-top:42px;font-size:11px;color:{MUT};">SCORE DE CONFIANCE</div></div></div>'
            f'<div class="pay-kv"><div><span>Code d\'autorisation</span><span class="pay-code">{P["auth"] or "—"}</span></div>'
            f'<div><span>Référence</span><b>{P["ref"]}</b></div>'
            f'<div><span>Score IA (fraude)</span><b>{fmt_score(P["p"])}</b></div>'
            f'<div><span>Niveau de risque</span><b style="color:{COUL_NIVEAU[P["niv"]]};">{P["niv"]}</b></div>'
            f'<div><span>Statut</span><b>{P["statut"].title()}</b></div>'
            f'<div><span>Date</span><b>{P["horodatage"]}</b></div></div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        try:
            pdf = generer_recu_pdf(P, banque)
            st.download_button(
                "📄 Télécharger le reçu de la banque (PDF)" if ok else "📄 Télécharger l'avis de refus (PDF)",
                data=pdf, file_name=f"Recu_{P['ref']}.pdf", mime="application/pdf", key=f"dl_{P['ref']}", type="primary", **LARGE,
            )
        except ModuleNotFoundError:
            st.info("Installez `reportlab` pour générer le reçu : `pip install reportlab`.")
        st.button("➕ Nouveau paiement", on_click=_reset_paiement, key="btn_nouveau", **LARGE)
        st.caption(f"Signature numérique du reçu : `{signer(P)}`")
    if ok and not P.get("ballons"):
        st.balloons()
        P["ballons"] = True


def page_paiement() -> None:
    st.markdown(CSS_PAIEMENT, unsafe_allow_html=True)
    st.markdown("### 🏦 Paiement par Carte Bancaire — Autorisation intelligente en temps réel")
    st.caption("Carte du client → analyse IA → 3-D Secure si nécessaire → notification SMS → reçu bancaire signé.")
    st.session_state.setdefault("banque_nom", "Banque Partenaire")
    st.session_state.setdefault("paiement", None)
    with st.expander("🏛️ Personnaliser le reçu (nom de la banque)"):
        st.text_input("Nom de la banque affiché sur le reçu et les SMS", key="banque_nom", max_chars=40)
    banque = st.session_state["banque_nom"].strip() or "Banque Partenaire"
    P = st.session_state["paiement"]
    if P is None:
        _ecran_saisie(banque)
    elif P["stage"] == "otp":
        _ecran_otp(P, banque)
    else:
        _ecran_resultat(P, banque)


# ------------------------------------------------------------------ SIDEBAR
with st.sidebar:
    st.markdown(
        """
        <div class="brand" style="padding: 6px 4px 18px;">
            <div class="shield-logo">🔒</div>
            <div>
                <div style="font-weight:800; font-size:19px; letter-spacing:0.4px;">FRAUD<span>SHIELD</span></div>
                <div style="font-size:11px; color:#8ea4d2;">BANKING SECURITY AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top:26px; background:linear-gradient(160deg,#0e2352,#071330);
                    border:1px solid #1e3a7a; border-radius:16px; padding:18px 14px; text-align:center;">
            <div style="font-size:32px; margin-bottom:6px;">🛡️📈</div>
            <div style="font-weight:700; font-size:0.92rem; color:#f8fafc; line-height:1.35;">
                Ensemble, prévenons la fraude et renforçons la confiance bancaire.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ TOPBAR
maintenant = datetime.now().strftime("%d/%m/%Y • %H:%M")
transactions_live = st.session_state.get("transactions_live", [])
alertes_live = [t for t in transactions_live if t.get("Niveau") in ("Élevé", "Moyen") or t.get("Montant élevé")]
nb_alertes_non_lues = sum(1 for t in alertes_live if t.get("Notification non lue", False))
libelle_cloche = (str(nb_alertes_non_lues) if nb_alertes_non_lues < 10 else "9+") if nb_alertes_non_lues else "\u200b"
if nb_alertes_non_lues:
    st.markdown(
        '<style>[data-testid="stPopover"] button, button[data-testid="stPopoverButton"]{border-color:#ef4444 !important;'
        'background:linear-gradient(145deg,#7f1d1d,#3b0d0d) !important;box-shadow:0 0 18px rgba(239,68,68,.6) !important;}</style>',
        unsafe_allow_html=True,
    )
col_brand, col_status, col_bell, col_user = st.columns([3.4, 1.9, 1.0, 2.7], gap="medium")
with col_brand:
    st.markdown(f'<div class="brand" style="padding:8px 4px 14px;"><div class="shield-logo">🔒</div><div><div class="brand-title">FRAUD<span>SHIELD</span> AI</div><div class="brand-sub">Intelligence Artificielle au service de la sécurité bancaire</div></div></div>', unsafe_allow_html=True)
with col_status:
    st.markdown(f'<div class="sys-status" style="padding-top:8px;"><div class="sys-online"><span class="pulse-dot"></span>Système en ligne</div><div>Dernière analyse : {maintenant}</div></div>', unsafe_allow_html=True)
with col_bell:
    with st.popover(libelle_cloche):
        st.markdown('<div class="notif-title">🔔 Notifications FraudShield</div>', unsafe_allow_html=True)
        if alertes_live:
            blocs_notif = []
            for alerte in alertes_live[-8:][::-1]:
                critique_montant = bool(alerte.get("Alerte montant critique", False))
                niv_a = alerte.get("Niveau", "Moyen")
                icone = "🚨" if critique_montant else ("🔴" if niv_a == "Élevé" else "🟡")
                coul = RED if (critique_montant or niv_a == "Élevé") else GOLD
                nouvelle = bool(alerte.get("Notification non lue", False))
                etiquette = '<span class="notif-new">NOUVELLE</span>' if nouvelle else ""
                blocs_notif.append(
                    f'<div class="notif-item{" new" if nouvelle else ""}" style="border-left-color:{coul};">'
                    f'<div class="notif-ic">{icone}</div>'
                    f'<div class="notif-main">'
                    f'<div class="notif-h">{niv_a} — Client {alerte.get("Client", "—")}{etiquette}</div>'
                    f'<div class="notif-s">{fmt_fcfa(alerte.get("Montant (FCFA)", 0))} · Score IA <b style="color:{coul};">{fmt_score(alerte.get("Probabilité", 0))}</b></div>'
                    f'<div class="notif-m">{alerte.get("Date & Heure", "—")} · {alerte.get("Motif", "")}</div>'
                    f'</div></div>'
                )
            st.markdown("".join(blocs_notif), unsafe_allow_html=True)
            if nb_alertes_non_lues > 0:
                if st.button("✓ Marquer les notifications comme lues", key="marquer_notifications_lues", use_container_width=True):
                    for transaction in st.session_state.get("transactions_live", []):
                        transaction["Notification non lue"] = False
                    sauver_journal()
                    st.rerun()
        else:
            st.markdown('<div class="notif-empty">✅ Aucune alerte pour le moment.</div>', unsafe_allow_html=True)
with col_user:
    st.markdown(f'<div class="user-pill" style="margin:5px 0 12px;">{AVATAR_HTML}<div><div style="font-weight:700;font-size:0.9rem;color:#fff;">Ibrahima DIENG</div><div style="font-size:0.75rem;color:{GOLD};font-weight:600;">Junior Data Scientist &amp; IA</div></div></div>', unsafe_allow_html=True)

# Toast à la réception d'une nouvelle alerte
_msg_toast = st.session_state.pop("toast_alerte", None)
if _msg_toast:
    try:
        st.toast(_msg_toast, icon="🚨")
    except Exception:
        pass

# ------------------------------------------------------------------ NAVIGATION PRINCIPALE
with panneau("nav_page"):
    page = st.radio(
        "Navigation",
        [
            "🏠  Dashboard",
            "🧠  Analyse IA",
            "💳  Transactions",
            "🏦  Paiement Carte",
            "🔔  Alertes",
            "📊  Rapports",
            "⚙️  Paramètres",
            "ℹ️  À propos",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="radio_navigation",
    )
st.write("")

# ==============================================================================
# 1. DASHBOARD EXÉCUTIF
# ==============================================================================
if "Dashboard" in page:
    lot = lot_courant()
    repart = lot["Niveau"].value_counts(normalize=True).mul(100).reindex(NIVEAUX).fillna(0)
    journal_dashboard = st.session_state.get("transactions_live", [])
    # Le compteur critique suit les transactions réellement saisies lorsqu'il y en a.
    n_crit = (
        sum(1 for t in journal_dashboard if t.get("Niveau") == "Élevé")
        if journal_dashboard else int((lot["Niveau"] == "Élevé").sum())
    )
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-text">
                <h1>Détection Intelligente<br>des <span>Transactions Frauduleuses</span></h1>
                <p>Analyse prédictive par <b>Régression Logistique &amp; Scoring en Temps Réel</b> :
                81,5 % des fraudes interceptées en n’auditant que les 200 alertes prioritaires.</p>
            </div>
            <div class="hero-badge-box">
                <div class="lock">🔐</div>
                <span>SÉCURITÉ BANCAIRE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rp1, rp2 = st.columns([3, 1.3])
    with rp1:
        st.markdown('<div style="padding-top:6px;color:#cbd5e1;font-size:0.92rem;">📑 <b>Rapport de supervision</b> : indicateurs, graphiques, alertes et transactions de la session, prêts à partager.</div>', unsafe_allow_html=True)
    with rp2:
        bouton_rapport_pdf("btn_rapport_dashboard")
    st.write("")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        render_kpi("b", "💳", "rgba(59,130,246,0.18)", "Transactions Étudiées", f"{TOTAL:,}".replace(",", " "), "+12.5%", GRN, 88, BLU, [42, 48, 45, 56, 61, 68, 75]),
        unsafe_allow_html=True,
    )
    k2.markdown(
        render_kpi("g", "✔️", "rgba(16,185,129,0.18)", "Transactions Normales", f"{NORMALES:,}".replace(",", " "), "99.83%", GRN, 99.8, GRN, [80, 82, 81, 84, 85, 86, 88]),
        unsafe_allow_html=True,
    )
    k3.markdown(
        render_kpi("r", "⚠️", "rgba(239,68,68,0.18)", "Fraudes Réelles (Dataset)", f"{FRAUDES}", "0.173%", RED, 18, RED, [12, 19, 14, 24, 18, 29, 21]),
        unsafe_allow_html=True,
    )
    k4.markdown(
        render_kpi("y", "🎯", "rgba(240,180,41,0.18)", "Rappel @ 200 Alertes", "81.5%", "PR-AUC 0.705", GOLD, 81.5, GOLD, [41, 55, 63, 70, 76, 80, 81.5]),
        unsafe_allow_html=True,
    )

    # ---------- BANDE « SESSION EN DIRECT » (calculée sur les transactions réellement analysées)
    jr = st.session_state.get("transactions_live", [])
    st.markdown('<div class="sec-title" style="margin-top:14px;">⚡ Activité en direct <small>Calculée sur les transactions que vous analysez</small></div>', unsafe_allow_html=True)
    if jr:
        probs = [float(x.get("Probabilité", 0)) for x in jr]
        n_an = len(jr)
        n_al = sum(1 for x in jr if x.get("Niveau") in ("Élevé", "Moyen"))
        mt_bloque = sum(float(x.get("Montant (FCFA)", 0)) for x in jr if x.get("Niveau") == "Élevé")
        mt_total = sum(float(x.get("Montant (FCFA)", 0)) for x in jr)
        sp = (probs[-7:] if len(probs) > 1 else probs * 2)
        l1, l2, l3, l4 = st.columns(4)
        l1.markdown(render_kpi("b", "🧾", "rgba(59,130,246,0.18)", "Transactions analysées", f"{n_an}", f"{fmt_fcfa(mt_total)}", MUT, min(100, n_an * 10), BLU, sp), unsafe_allow_html=True)
        l2.markdown(render_kpi("y", "🔔", "rgba(240,180,41,0.18)", "Alertes déclenchées", f"{n_al}", f"{n_al / n_an * 100:.0f}% du flux", GOLD, n_al / n_an * 100, GOLD, sp), unsafe_allow_html=True)
        l3.markdown(render_kpi("r", "🛑", "rgba(239,68,68,0.18)", "Montant à risque élevé", fmt_fcfa(mt_bloque).replace(" FCFA", ""), "FCFA", RED, (mt_bloque / mt_total * 100) if mt_total else 0, RED, sp), unsafe_allow_html=True)
        l4.markdown(render_kpi("g", "🎯", "rgba(16,185,129,0.18)", "Score IA moyen", fmt_score(sum(probs) / n_an), f"max {fmt_score(max(probs))}", GRN, min(100, sum(probs) / n_an * 100), GRN, sp), unsafe_allow_html=True)
    else:
        st.info("Aucune transaction analysée pour l'instant — ouvrez **💳 Transactions** pour scorer votre première opération : les indicateurs en direct apparaîtront ici.")

    st.write("")

    # ---------- 1) ALERTES RÉCENTES (cartes avec Score IA bien visible)
    journal_alertes = st.session_state.get("transactions_live", [])
    elements_alertes = []
    if journal_alertes:
        for tr in journal_alertes[-3:][::-1]:
            niv = tr.get("Niveau", "Faible")
            heure = str(tr.get("Date & Heure", "—")).split()[-1]
            elements_alertes.append({
                "icone": {"Élevé": "🔴", "Moyen": "🟡", "Faible": "🟢"}.get(niv, "🟢"),
                "titre": {"Élevé": "Transaction suspecte", "Moyen": "Comportement inhabituel", "Faible": "Transaction normale"}.get(niv, "Transaction analysée"),
                "ref": str(tr.get("ID", "—")),
                "montant": fmt_fcfa(tr.get("Montant (FCFA)", 0)),
                "p": float(tr.get("Probabilité", 0)),
                "heure": heure,
                "couleur": COUL_NIVEAU.get(niv, GRN),
            })
        libelle_source_alertes = "Flux réel · 3 dernières analyses"
    else:
        for niv, icone, titre in [("Élevé", "🔴", "Transaction suspecte"), ("Moyen", "🟡", "Comportement inhabituel"), ("Faible", "🟢", "Transaction normale")]:
            cand = lot[lot["Niveau"] == niv].sort_values("Probabilité", ascending=False)
            if cand.empty:
                continue
            # Faible : valeur représentative (médiane) plutôt que le minimum, qui s'affichait « 0.00 ».
            ligne = cand.iloc[len(cand) // 2] if niv == "Faible" else cand.iloc[0]
            elements_alertes.append({
                "icone": icone, "titre": titre, "ref": str(ligne.get("ID", "—")),
                "montant": fmt_fcfa(ligne.get("Montant (FCFA)", 0)), "p": float(ligne.get("Probabilité", 0)),
                "heure": "Démo", "couleur": COUL_NIVEAU[niv],
            })
        libelle_source_alertes = "Exemples de démonstration · remplacés par vos analyses"

    st.markdown(
        f'<div class="sec-title">🔔 Alertes Récentes <span class="badge-pill b-sus">{n_crit} Critiques</span>'
        f'<small>{libelle_source_alertes}</small></div>',
        unsafe_allow_html=True,
    )
    if not elements_alertes:
        st.info("Aucune transaction à afficher pour le moment.")
    else:
        cols_al = st.columns(3)
        for col, al in zip(cols_al, elements_alertes):
            largeur = max(min(al["p"] * 100, 100), 3)
            col.markdown(
                f"""
                <div class="score-card" style="border-left-color:{al['couleur']};">
                    <div class="sc-top"><span class="sc-title">{al['icone']} {al['titre']}</span><span class="sc-time">{al['heure']}</span></div>
                    <div class="sc-ref">{al['ref']}</div>
                    <div class="sc-amt">Montant : <b>{al['montant']}</b></div>
                    <div class="sc-lbl">SCORE IA</div>
                    <div class="sc-score" style="color:{al['couleur']};">{fmt_score(al['p'])}</div>
                    <div class="score-bar"><div style="width:{largeur:.1f}%; background:{al['couleur']};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # ---------- 2) DERNIÈRES TRANSACTIONS (tableau pleine largeur, sans débordement)
    parts = [
        lot[lot["Niveau"] == "Élevé"].sample(min(1, (lot["Niveau"] == "Élevé").sum()), random_state=3),
        lot[lot["Niveau"] == "Moyen"].sample(min(1, (lot["Niveau"] == "Moyen").sum()), random_state=3),
        lot[lot["Niveau"] == "Faible"].sample(min(3, (lot["Niveau"] == "Faible").sum()), random_state=3),
    ]
    recents = pd.concat(parts).sort_values("ID", ascending=False).head(5)
    badge = {"Élevé": ("b-sus", "▲ Suspecte", RED), "Moyen": ("b-mid", "● À vérifier", GOLD), "Faible": ("b-ok", "✔ Normale", GRN)}
    lignes = ""
    journal_live = st.session_state.get("transactions_live", [])
    if journal_live:
        source_lignes = [(r["ID"], r["Date & Heure"], r["Client"], r["Montant (FCFA)"], r["Niveau"], r["Probabilité"]) for r in journal_live[-5:][::-1]]
    else:
        source_lignes = [
            (r["ID"], (datetime.now() - timedelta(minutes=2 * i + 1)).strftime("%d/%m/%Y %H:%M"), r.get("Client", "—"), r["Montant (FCFA)"], r["Niveau"], r["Probabilité"])
            for i, (_, r) in enumerate(recents.iterrows())
        ]
    for (rid, rdt, rcl, rmt, rniv, rp) in source_lignes:
        b_cls, b_txt, b_col = badge[rniv]
        lignes += (
            f'<tr><td class="mono"><b>{rid}</b></td><td>{rdt}</td><td>{rcl}</td>'
            f'<td class="mono">{fmt_fcfa(rmt)}</td>'
            f'<td><span class="badge-pill {b_cls}">{b_txt}</span></td>'
            f'<td><span class="score-chip" style="color:{b_col}; border-color:{b_col};">{fmt_score(rp)}</span></td></tr>'
        )
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-hdr">
                <span>🧾 Dernières Transactions Analysées</span>
                <span style="font-size:0.78rem; color:#8ea4d2;">Vert / Jaune / Rouge</span>
            </div>
            <div class="tbl-wrap">
            <table class="txn-table">
                <thead><tr><th>ID Transaction</th><th>Date &amp; Heure</th><th>Client</th><th>Montant (FCFA)</th><th>Statut</th><th>Score IA</th></tr></thead>
                <tbody>{lignes}</tbody>
            </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # ---------- 3) GRAPHIQUES (en bas, grands et lisibles)
    st.markdown('<div class="sec-title">📊 Analyse graphique</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1.5, 1])
    CFG = {"displayModeBar": False}

    with col_left:
        with panneau("panel_courbe"):
            st.markdown('<div class="panel-hdr"><span>📈 Courbe de Capture des Fraudes (Rappel@k)</span></div>', unsafe_allow_html=True)
            fig_evo = go.Figure()
            positions = ["top center", "top center", "bottom center"]   # Isolation / Régression en haut, Gradient en bas : aucune collision
            pct = lambda v, d=1: f"{v:.{d}f} %".replace(".", ",")
            for mod, c_mod, pos in zip(METRIQUES["Modèle"], [MUT, GOLD, BLU], positions):
                row = METRIQUES[METRIQUES["Modèle"] == mod].iloc[0]
                fig_evo.add_trace(
                    go.Scatter(
                        x=["Top 50 alertes", "Top 200 alertes", "Top 500 alertes"],
                        y=[row["Rappel@50"] * 100, row["Rappel@200"] * 100, row["Rappel@500"] * 100],
                        name=mod.replace(" (Retenu)", ""),
                        mode="lines+markers+text",
                        text=[f"<b>{pct(row['Rappel@50']*100)}</b>", f"<b>{pct(row['Rappel@200']*100)}</b>", f"<b>{pct(row['Rappel@500']*100)}</b>"],
                        textposition=pos,
                        textfont=dict(color=c_mod, size=14, family="JetBrains Mono"),
                        line=dict(color=c_mod, width=4),
                        marker=dict(size=12, line=dict(color="#ffffff", width=2)),
                        cliponaxis=False,
                    )
                )
            fig_evo = style_plotly(fig_evo, 400)
            fig_evo.update_layout(
                margin=dict(l=10, r=20, t=20, b=10),
                yaxis=dict(title="Fraudes interceptées (%)", range=[-8, 118], gridcolor="rgba(255,255,255,0.10)", zeroline=False, tickfont=dict(size=13, color="#e2e8f0")),
                xaxis=dict(range=[-0.25, 2.25], tickfont=dict(size=14, color="#f1f5f9"), gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center", font=dict(size=14, color="#f8fafc")),
            )
            fig_evo.add_annotation(
                x=1, y=112, xref="x", yref="y", showarrow=False,
                text="<b>Régression Logistique : 81,5 % des fraudes captées avec 200 alertes</b>",
                font=dict(size=13, color="#0a1736"), bgcolor=GOLD, borderpad=6, bordercolor=GOLD,
            )
            afficher_plot(fig_evo, key="fig_evo_dash", config=CFG)

    with col_right:
        with panneau("panel_donut"):
            st.markdown('<div class="panel-hdr"><span>🛡️ Répartition des Risques</span></div>', unsafe_allow_html=True)
            noms = [f"Risque Faible · {repart['Faible']:.1f}%", f"Risque Moyen · {repart['Moyen']:.1f}%", f"Risque Élevé · {repart['Élevé']:.1f}%"]
            fig_donut = go.Figure(
                go.Pie(
                    labels=noms,
                    values=[repart["Faible"], repart["Moyen"], repart["Élevé"]],
                    hole=0.66, sort=False,
                    marker=dict(colors=[GRN, GOLD, RED], line=dict(color=CARD, width=2)),
                    textinfo="none",
                )
            )
            fig_donut.add_annotation(
                text=f"<b>{repart['Élevé'] + repart['Moyen']:.1f}%</b><br><span style='font-size:12px;color:#8ea4d2'>Sous surveillance</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(size=24, color="#fff"),
            )
            fig_donut = style_plotly(fig_donut, 330)
            fig_donut.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            afficher_plot(fig_donut, key="fig_donut_dash", config=CFG)
            _lignes = "".join(
                f'<div style="display:flex;align-items:center;gap:12px;padding:9px 14px;margin:6px 0;border-radius:12px;'
                f'background:rgba(255,255,255,.05);border-left:5px solid {c};">'
                f'<span style="width:14px;height:14px;border-radius:50%;background:{c};box-shadow:0 0 10px {c};"></span>'
                f'<span style="flex:1;font-weight:700;font-size:15px;color:#f8fafc;">{lib}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-weight:800;font-size:18px;color:{c};">{v:.1f} %</span></div>'
                for lib, v, c in [("Risque Faible", repart["Faible"], GRN), ("Risque Moyen", repart["Moyen"], GOLD), ("Risque Élevé", repart["Élevé"], RED)]
            )
            st.markdown(f'<div translate="no">{_lignes}</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="perf-banner">
                    <div style="font-size:22px;">🏅</div>
                    <div>
                        <div style="font-weight:700; font-size:0.9rem; color:#ffd166;">Système performant !</div>
                        <div style="font-size:0.8rem; color:#cbd5e1;">ROC-AUC : 0,973 · Zéro fuite temporelle.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # ---------- 3 bis) HISTORIQUE DES SCORES IA (session)
    jr2 = st.session_state.get("transactions_live", [])
    if jr2:
        with panneau("panel_historique"):
            st.markdown('<div class="panel-hdr"><span>🕒 Historique des Scores IA (transactions analysées)</span></div>', unsafe_allow_html=True)
            dernieres = jr2[-20:]
            xs = [f"#{len(jr2) - len(dernieres) + i + 1} · {x.get('Client', '—')}" for i, x in enumerate(dernieres)]
            ys = [float(x.get("Probabilité", 0)) * 100 for x in dernieres]
            cs = [COUL_NIVEAU.get(x.get("Niveau", "Faible"), GRN) for x in dernieres]
            fig_h = go.Figure(go.Bar(x=xs, y=ys, marker_color=cs, text=[fmt_score(y / 100) for y in ys], textposition="outside", textfont=dict(size=12)))
            eleve_s, moyen_s = seuils()
            fig_h.add_hline(y=moyen_s * 100, line_dash="dot", line_color=GOLD, annotation_text="Seuil Moyen", annotation_font_color=GOLD)
            fig_h.add_hline(y=eleve_s * 100, line_dash="dot", line_color=RED, annotation_text="Seuil Élevé", annotation_font_color=RED)
            fig_h = style_plotly(fig_h, 340)
            fig_h.update_layout(
                margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
                yaxis=dict(title="Score IA (%)", range=[0, max(max(ys) * 1.35, eleve_s * 100 * 1.15, 10)], gridcolor="rgba(255,255,255,0.08)"),
                xaxis=dict(tickfont=dict(size=11)),
            )
            afficher_plot(fig_h, key="fig_hist_dash", config=CFG)
        st.write("")

    # ---------- 4) À PROPOS (tout en bas)
    st.markdown(
        """
        <div class="panel">
            <div class="panel-hdr"><span>💡 À propos du Projet</span></div>
            <div style="display:flex; flex-wrap:wrap; gap:24px; align-items:center; justify-content:space-between;">
                <p style="flex:2 1 320px; font-size:0.86rem; color:#cbd5e1; line-height:1.55; margin:0;">
                    <b>FRAUDSHIELD AI</b> déploie une architecture de Machine Learning calibrée sur données bancaires
                    fortement déséquilibrées (284 807 opérations) avec validation chronologique stricte.
                </p>
                <div style="flex:1 1 260px; font-size:0.84rem; line-height:1.85; color:#e2e8f0;">
                    ✅ Machine Learning Supervisé<br>
                    ✅ Régression Logistique &amp; StandardScaler<br>
                    ✅ Optimisation Rappel@k &amp; PR-AUC<br>
                    ✅ Scoring temps réel &amp; export d'audit
                </div>
                <a href="https://github.com/ibrahimadieng997-boop" target="_blank" class="github-btn" style="flex:0 1 260px; margin-top:0;">
                    Découvrir le projet sur GitHub →
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 2. ANALYSE IA (SCORING BATCH CSV / DÉMO)
# ==============================================================================
elif "Analyse IA" in page:
    st.markdown("### 🧠 Analyse IA & Scoring d'un lot de transactions")
    c_up, c_sl = st.columns([2, 1])
    with c_up:
        f = st.file_uploader("Importer un fichier CSV (colonnes V1 à V28 et Amount)", type="csv", key="uploader_csv")
        unite_lbl = st.radio(
            "Unité de la colonne Amount du CSV",
            ["Euro (échelle du modèle)", "FCFA"],
            horizontal=True,
            key="unite_csv",
            help="Le modèle a été entraîné avec des montants en euros. Si votre CSV est en FCFA, il est converti automatiquement (1 € = 655,957 FCFA).",
        )
    with c_sl:
        k = st.slider("Capacité d'audit (Top k alertes)", 10, 500, 100, 10, key="slider_top_k")

    if f is not None:
        try:
            brut = pd.read_csv(f)
        except Exception as e:
            st.error(f"Impossible de lire le CSV : {e}")
            st.stop()
        if "ID" not in brut.columns:
            brut.insert(0, "ID", [f"#TXN-{70000 + i}" for i in range(len(brut))])
        manque = [c for c in COLS if c not in brut.columns]
        if manque:
            st.error(f"Colonnes manquantes dans le CSV : {', '.join(manque[:6])}…")
            st.stop()
        df = analyser_lot(brut, "FCFA" if unite_lbl == "FCFA" else "EUR")
        source = f"Fichier client importé (montants en {'FCFA' if unite_lbl == 'FCFA' else 'euros'})"
    else:
        df = lot_demo_score()
        source = "Flux de démonstration synthétique (2 500 transactions, montants à l'échelle du modèle)"

    k = min(k, len(df))
    st.session_state["resultat"], st.session_state["k"] = df, k
    df = finaliser(df)
    st.caption(f"Source active : **{source}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions analysées", f"{len(df):,}".replace(",", " "))
    m2.metric("Alertes prioritaires (Top k)", k)
    m3.metric("Taux de contrôle", f"{k / len(df) * 100:.1f} %")
    if "Class" in df.columns and df["Class"].sum() > 0:
        trouve = int(df.loc[df["Rang"] <= k, "Class"].sum())
        m4.metric("Fraudes interceptées", f"{trouve} / {int(df['Class'].sum())}")

    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown("#### File prioritaire pour l'équipe conformité")
        cols_vue = [c for c in ["Rang", "ID", "Client", "Montant (FCFA)", "Probabilité", "Niveau", "Motif"] if c in df.columns]
        vue = df.sort_values("Rang").head(k)[cols_vue]
        afficher_df(
            vue,
            hide_index=True,
            height=370,
            key="table_alertes_ia",
            column_config={
                "Probabilité": st.column_config.ProgressColumn("Score de Risque", format="%.3f", min_value=0, max_value=1),
                "Montant (FCFA)": st.column_config.NumberColumn("Montant (FCFA)", format="%,.0f FCFA"),
            },
        )
        st.download_button("⬇️ Exporter les alertes (CSV)", vue.to_csv(index=False).encode(), "alertes_fraudshield.csv", "text/csv", key="btn_dl_csv")

    with g2:
        st.markdown("#### Segmentation des risques")
        cnt = df["Niveau"].value_counts().reindex(NIVEAUX).dropna().astype(int)
        fig = go.Figure(
            go.Pie(
                labels=list(cnt.index),
                values=list(cnt.values),
                hole=0.62,
                sort=False,
                marker_colors=[COUL_NIVEAU[n] for n in cnt.index],
            )
        )
        afficher_plot(style_plotly(fig, 370), key="chart_segmentation_risques")

    with st.expander("📐 Niveaux de risque par tranche de montant"):
        bornes = [-1, 50_000, 200_000, 1_000_000, np.inf]
        noms = ["< 50 000", "50 000 – 200 000", "200 000 – 1 M", "> 1 M"]
        tranche = pd.cut(df["Montant (FCFA)"], bins=bornes, labels=noms)
        tab = pd.crosstab(tranche, df["Niveau"]).reindex(columns=NIVEAUX, fill_value=0)
        tab.index.name = "Tranche (FCFA)"
        afficher_df(tab.reset_index(), hide_index=True, key="table_tranches")
        st.caption("Le score dépend surtout des composantes V1–V28 ; le montant n'a qu'un poids très faible dans le modèle, "
                   "d'où des niveaux variés dans chaque tranche.")

# ==============================================================================
# 3. SIMULATEUR DE TRANSACTION UNITAIRE
# ==============================================================================
elif "Transactions" in page:
    st.markdown("### 💳 Simulateur de Scoring Unitaire en Temps Réel")
    st.caption("Saisissez l'identifiant du client et le montant de sa transaction.")
    with st.form("form_scoring_unitaire"):
        c1, c2 = st.columns(2)
        with c1:
            client_id = st.text_input("🪪 Identifiant du client", placeholder="Ex. CLIENT-001", max_chars=60, key="tx_client")
        with c2:
            montant_fcfa = st.number_input("💰 Montant (FCFA)", min_value=0.0, max_value=100_000_000.0, value=250_000.0, step=10_000.0, format="%.0f", key="tx_montant")
        analyser = st.form_submit_button("🔍 Analyser la transaction", use_container_width=True)

    if analyser:
        if not client_id.strip():
            st.error("Veuillez saisir l'identifiant du client.")
        else:
            ligne = pd.DataFrame([dict(zip(COLS, SCALER.mean_))])
            ligne.loc[0, "Amount"] = float(montant_fcfa) / TAUX_EUR_FCFA
            p = float(scorer(ligne)[0])
            niv = niveau_de(p, float(montant_fcfa))
            seuil_vigilance = float(st.session_state.get("seuil_montant_alerte", 1_000_000.0))
            montant_eleve = float(montant_fcfa) >= seuil_vigilance
            alerte_montant_critique = float(montant_fcfa) >= 5_000_000.0

            # Règle métier de démonstration : à partir de 5 000 000 FCFA,
            # la transaction déclenche une alerte rouge, indépendamment du score ML.
            if alerte_montant_critique:
                niv = "Élevé"
            elif montant_eleve and niv == "Faible":
                niv = "Moyen"

            motif = str(classer([p], [montant_fcfa])[1][0])
            if alerte_montant_critique:
                motif += " · Alerte critique : montant ≥ 5 000 000 FCFA (règle métier)"
            elif montant_eleve:
                motif += " · Montant supérieur au seuil de vigilance"

            action = {
                "Élevé": "Blocage préventif recommandé — contrôle humain immédiat",
                "Moyen": "Vérification complémentaire / 2FA",
                "Faible": "Transaction autorisée",
            }[niv]
            transaction = {
                "ID": f"TX-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(st.session_state['transactions_live'])+1:03d}",
                "Date & Heure": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "Client": client_id.strip(),
                "Montant (FCFA)": float(montant_fcfa),
                "Probabilité": p,
                "Niveau": niv,
                "Motif": motif,
                "Action": action,
                "Montant élevé": montant_eleve,
                "Alerte montant critique": alerte_montant_critique,
                "Notification non lue": niv in ("Élevé", "Moyen"),
            }
            st.session_state["transactions_live"].append(transaction)
            st.session_state["dernier_resultat_unitaire"] = transaction
            if niv in ("Élevé", "Moyen"):
                st.session_state["toast_alerte"] = f"Nouvelle alerte {niv} · {client_id.strip()} · {fmt_fcfa(montant_fcfa)}"
            sauver_journal()
            # Relancer immédiatement pour que la cloche affiche le nouveau compteur (ex. 🔔 1).
            st.rerun()

    resultat = st.session_state.get("dernier_resultat_unitaire")
    if resultat:
        niv = resultat["Niveau"]
        couleur = COUL_NIVEAU[niv]
        badge_cls = {"Élevé": "b-sus", "Moyen": "b-mid", "Faible": "b-ok"}[niv]
        st.markdown(f'<div class="panel" style="margin-top:16px;"><div style="color:{MUT};">Résultat pour le client {resultat["Client"]}</div><div style="font-family:JetBrains Mono,monospace;font-size:2.6rem;font-weight:800;color:{couleur};margin:8px 0;">{fmt_score(resultat["Probabilité"])}</div><span class="badge-pill {badge_cls}">Risque {niv}</span><p style="margin-top:12px;">Montant : <b>{fmt_fcfa(resultat["Montant (FCFA)"])}</b></p><p style="color:{MUT};">Motif : {resultat["Motif"]}</p><p>Action recommandée : <b>{resultat["Action"]}</b></p></div>', unsafe_allow_html=True)
        p_aff = float(resultat["Probabilité"]) * 100
        eleve_g, moyen_g = seuils()
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=p_aff, number=dict(suffix=" %", font=dict(size=34, color=couleur)),
            title=dict(text="Jauge du Score IA", font=dict(size=14, color=MUT)),
            gauge=dict(
                axis=dict(range=[0, max(100 * eleve_g * 2, 20)], tickcolor=MUT),
                bar=dict(color=couleur),
                bgcolor="rgba(255,255,255,0.04)",
                steps=[dict(range=[0, moyen_g * 100], color="rgba(16,185,129,0.35)"),
                       dict(range=[moyen_g * 100, eleve_g * 100], color="rgba(240,180,41,0.35)"),
                       dict(range=[eleve_g * 100, max(100 * eleve_g * 2, 20)], color="rgba(239,68,68,0.35)")],
            ),
        ))
        afficher_plot(style_plotly(fig_g, 260), key="fig_gauge_unitaire", config={"displayModeBar": False})
        if resultat.get("Alerte montant critique", False):
            st.error("🚨 ALERTE ROUGE : montant égal ou supérieur à 5 000 000 FCFA. Une nouvelle notification 🔔 a été créée.")
        elif niv == "Élevé":
            st.error("🚨 ALERTE CRITIQUE : transaction à risque élevé selon le modèle. Vérification humaine recommandée.")
        elif niv == "Moyen":
            st.warning("⚠️ NOTIFICATION : transaction à surveiller. Vérification complémentaire recommandée.")
        else:
            st.success("✅ Transaction classée à risque faible par le modèle.")
        st.info("Limite technique : V1 à V28 ne sont pas saisies dans ce formulaire. Le modèle utilise leurs moyennes apprises. L'ID client sert au suivi et n'est pas une variable du modèle.")

    with st.expander("⚙️ Seuil de montant élevé"):
        seuil = st.number_input("Alerter à partir de (FCFA)", min_value=0.0, max_value=100_000_000.0, value=float(st.session_state.get("seuil_montant_alerte", 1_000_000.0)), step=100_000.0, key="seuil_montant_widget")
        st.session_state["seuil_montant_alerte"] = seuil
        st.caption("Le seuil de vigilance est configurable. Par défaut : 🟡 jaune dès 1 000 000 FCFA, 🔴 rouge dès 5 000 000 FCFA. Il s’agit d’une règle métier de démonstration, pas d’un seuil réglementaire bancaire.")

    st.markdown("#### 🧾 Dernières transactions analysées")
    journal = st.session_state.get("transactions_live", [])
    if journal:
        journal_df = pd.DataFrame(journal[::-1])
        journal_df["Score IA"] = journal_df["Probabilité"].map(fmt_score)
        afficher_df(journal_df[["ID", "Date & Heure", "Client", "Montant (FCFA)", "Score IA", "Niveau", "Motif", "Action"]], hide_index=True, key="journal_transactions_live")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "⬇️ Exporter le journal d'audit (CSV)",
                data=journal_df.drop(columns=["Notification non lue"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
                file_name=f"fraudshield_journal_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True,
            )
        with ex2:
            if st.button("🗑️ Réinitialiser le journal", key="reset_journal", use_container_width=True):
                st.session_state["transactions_live"] = []
                st.session_state["dernier_resultat_unitaire"] = None
                sauver_journal()
                st.rerun()
    else:
        st.info("Aucune transaction analysée pour le moment.")

# ==============================================================================
# 4. CENTRE D'ALERTES
# ==============================================================================
elif "Alertes" in page:
    st.markdown("### 🔔 Centre de Supervision des Alertes")

    alertes_transactions_live = [
        t for t in st.session_state.get("transactions_live", [])
        if t.get("Niveau") in ("Élevé", "Moyen") or t.get("Montant élevé")
    ]
    if alertes_transactions_live:
        st.markdown("#### 🚨 Alertes issues du simulateur en temps réel")
        for alerte in alertes_transactions_live[::-1]:
            if alerte.get("Alerte montant critique", False):
                st.error(
                    f"🚨 ALERTE ROUGE — Client {alerte['Client']} · "
                    f"{fmt_fcfa(alerte['Montant (FCFA)'])} · {alerte['Date & Heure']}"
                )
            elif alerte.get("Niveau") == "Élevé":
                st.error(
                    f"🔴 Risque élevé — Client {alerte['Client']} · "
                    f"{fmt_fcfa(alerte['Montant (FCFA)'])} · {alerte['Date & Heure']}"
                )
            else:
                st.warning(
                    f"🟡 Risque moyen — Client {alerte['Client']} · "
                    f"{fmt_fcfa(alerte['Montant (FCFA)'])} · {alerte['Date & Heure']}"
                )
        st.divider()

    df = lot_courant()

    fa, fb = st.columns([1, 1])
    with fa:
        filtre = st.radio("Niveau", ["Tous", "Élevé", "Moyen"], horizontal=True, key="filtre_alertes")
    with fb:
        ordre = st.radio("Tri", ["Plus récentes", "Plus critiques"], horizontal=True, key="tri_alertes")
    if ordre == "Plus récentes":
        liste = alertes_recentes(df, 8, filtre)
    else:
        liste = df[df["Niveau"] != "Faible"]
        if filtre != "Tous":
            liste = liste[liste["Niveau"] == filtre]
        liste = liste.sort_values("Rang").head(8)
    if liste.empty:
        st.info("Aucune alerte pour ce filtre avec les seuils actuels.")

    blocs = []
    for _, r in liste.iterrows():
        niv = r["Niveau"]
        ic = {"Élevé": "🔴", "Moyen": "🟡", "Faible": "🟢"}[niv]
        b_cls = {"Élevé": "b-sus", "Moyen": "b-mid", "Faible": "b-ok"}[niv]
        blocs.append(
            f"""<div class="alert-item">
                <div style="display:flex; gap:12px; align-items:center;">
                    <div style="font-size:22px;">{ic}</div>
                    <div>
                        <b class="mono">{r["ID"]}</b> · Montant : <b>{fmt_fcfa(r["Montant (FCFA)"])}</b>
                        <div style="font-size:0.8rem; color:{MUT};">Score IA : <b style="color:#fff;">{fmt_score(r["Probabilité"])}</b> · Priorité #{int(r["Rang"])} · {r["Motif"]}</div>
                    </div>
                </div>
                <span class="badge-pill {b_cls}">{niv}</span>
            </div>"""
        )
    if blocs:
        st.markdown('<div translate="no">' + "".join(blocs) + "</div>", unsafe_allow_html=True)

# ==============================================================================
# 5. RAPPORTS & BENCHMARK
# ==============================================================================
elif "Rapports" in page:
    st.markdown("### 📊 Benchmark Scientifique & Métriques Métier")
    st.caption(f"Évaluation hors-échantillon (période future) : {TEST_N:,} transactions dont {TEST_F} fraudes.".replace(",", " "))
    afficher_df(
        METRIQUES.style.format({c: "{:.3f}" for c in METRIQUES.columns[1:]}),
        hide_index=True,
        key="table_benchmark_metriques",
    )
    st.download_button("⬇️ Télécharger le benchmark (CSV)", METRIQUES.to_csv(index=False).encode(), "benchmark_fraudshield.csv", "text/csv", key="btn_dl_bench")
    st.markdown("#### 📄 Rapport de supervision complet")
    st.caption("Un PDF professionnel : performance du modèle, activité, graphiques, alertes, dernières transactions, méthodologie et limites.")
    bouton_rapport_pdf("btn_rapport_page")

# ==============================================================================
# 6. PARAMÈTRES
# ==============================================================================
elif "Paramètres" in page:
    st.markdown("### ⚙️ Configuration du Moteur de Risque")
    eleve0, moyen0 = seuils()
    eleve = st.slider("Seuil Risque Élevé — blocage préventif (Probabilité ≥)", 0.05, 0.95, eleve0, 0.05, key="slider_seuil_critique")
    moyen = st.slider("Seuil Risque Moyen — vérification 2FA (Probabilité ≥)", 0.01, 0.50, min(moyen0, 0.50), 0.01, key="slider_seuil_moyen")
    if moyen >= eleve:
        st.warning("Le seuil Moyen doit rester inférieur au seuil Élevé : il est ramené sous le seuil Élevé.")
    # Mémorisation (les widgets non affichés sont oubliés par Streamlit d'une page à l'autre)
    st.session_state["seuil_eleve"], st.session_state["seuil_moyen"] = eleve, moyen
    st.caption(f"Faible : < {min(moyen, eleve):.2f}  ·  Moyen : {min(moyen, eleve):.2f} à {eleve:.2f}  ·  Élevé : ≥ {eleve:.2f}")
    st.markdown("**Règle d'impact financier** (évite d'alerter fortement sur des montants sans enjeu)")
    pa, pb = st.columns(2)
    with pa:
        p_nul = st.number_input("Montant négligeable (FCFA) → au mieux « Faible »", 0.0, 100_000.0,
                                planchers()[0], step=100.0, key="num_plancher_nul")
    with pb:
        p_imp = st.number_input("Montant à faible impact (FCFA) → au pire « Moyen »", 0.0, 1_000_000.0,
                                planchers()[1], step=500.0, key="num_plancher_impact")
    st.session_state["plancher_nul"], st.session_state["plancher_impact"] = p_nul, p_imp
    st.caption("Mettez les deux valeurs à 0 pour classer uniquement d'après le score du modèle.")
    seuil_montant = st.number_input("Seuil de montant élevé pour notification (FCFA)", min_value=0.0, max_value=100_000_000.0, value=float(st.session_state.get("seuil_montant_alerte", 1_000_000.0)), step=100_000.0, key="param_seuil_montant")
    st.session_state["seuil_montant_alerte"] = seuil_montant
    st.checkbox("Activer le journal d'audit réglementaire (BCEAO / PCI-DSS)", value=True, key="chk_audit")
    st.checkbox("Notifications temps réel vers l'équipe Conformité", value=True, key="chk_notif")
    st.success("Configuration active sur le modèle `modele_fraude.joblib` (Scikit-Learn).")

# ==============================================================================
# 6 bis. PAIEMENT CARTE
# ==============================================================================
elif "Paiement" in page:
    page_paiement()

# ==============================================================================
# 7. À PROPOS
# ==============================================================================
else:
    st.markdown("### ℹ️ Architecture Technique — AlerteFraude IA")
    st.markdown(
        """
        <div class="panel">
            <p style="line-height:1.65;">
                <b>FraudShield AI</b> est une solution de détection de fraude bancaire conçue par <b>Ibrahima DIENG</b> (Junior Data Scientist &amp; IA).
                Contrairement aux approches naïves biaisées par l'accuracy (99,83 % en prédisant toujours « normal »), ce système optimise
                directement la capacité opérationnelle des analystes fraude via la <b>PR-AUC (0,705)</b> et le <b>Rappel@200 (81,5 %)</b>.
            </p>
            <span class="badge-pill b-ok">Python</span>
            <span class="badge-pill b-ok">Scikit-Learn</span>
            <span class="badge-pill b-ok">StandardScaler</span>
            <span class="badge-pill b-ok">Streamlit</span>
            <span class="badge-pill b-ok">Plotly</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ FOOTER
st.markdown(
    f"""
    <div class="footer-bar">
        <div><b>AlerteFraude IA</b> &nbsp;|&nbsp; Projet Data Science &amp; IA — <b>Ibrahima DIENG</b>, Junior Data Scientist &amp; IA</div>
        <div>🔐 Données traitées localement · Journal d'audit exportable · v2.0</div>
        <div style="color:{GOLD}; font-style:italic;">L’IA pour une finance plus sécurisée</div>
    </div>
    """,
    unsafe_allow_html=True,
)
