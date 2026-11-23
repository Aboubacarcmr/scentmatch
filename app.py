"""Application Streamlit ScentMatch. Lancer avec : streamlit run app.py"""
from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from scentmatch_ml import family_display, normalize_text

st.set_page_config(page_title="ScentMatch | Famille olfactive", page_icon="✦", layout="wide")
st.markdown("""
<style>
:root { color-scheme:dark; --ink:#f7f1e8; --muted:#d5c6b4; --panel:#292824; --border:#756b5d; --gold:#d8b77f; }
.stApp { background:radial-gradient(ellipse at top left,#302b24 0%,#1d1e1c 48%,#191a19 100%); color:var(--ink); }
.block-container { width:100%; max-width:1120px; padding:clamp(1.25rem,4vw,2.5rem) clamp(1rem,4vw,2rem) 2rem; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#25231f; border-right:1px solid #494237; }
[data-testid="stSidebar"] * { color:var(--ink); }
h1,h2,h3,p,label,[data-testid="stMarkdownContainer"] { color:var(--ink); }
h1 { letter-spacing:.04em; font-family:Georgia,serif; }
[data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px; }
[data-testid="stMetricLabel"],[data-testid="stMetricValue"] { color:var(--ink); }
input,textarea,[data-baseweb="select"] > div,[data-baseweb="input"] > div { color:var(--ink) !important; background:#242522 !important; border-color:#8c7a61 !important; }
input::placeholder,textarea::placeholder { color:#c6b9a8 !important; opacity:1 !important; }
[data-baseweb="select"] *,[role="listbox"],[role="option"] { color:var(--ink) !important; background-color:#292824; }
[role="option"]:hover,[aria-selected="true"] { background-color:#514637 !important; }
[data-testid="stFileUploader"] section { background:#292824; border:1px dashed #9b8668; border-radius:12px; }
[data-testid="stFileUploader"] section, [data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span, [data-testid="stFileUploader"] section p { color:#e2d5c3 !important; }
[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] { color:#e2d5c3 !important; }
[data-testid="stFileUploader"] button { color:var(--ink) !important; border:1px solid #a38b68 !important; background:#38332b !important; }
.stButton > button { color:#211b14; background:var(--gold); border:1px solid var(--gold); border-radius:10px; font-weight:700; }
.stButton > button:hover { color:#211b14; background:#e7c993; border-color:#e7c993; }
.stButton > button:focus { outline:2px solid #f7e1b7; outline-offset:2px; }
[data-testid="stAlert"],[data-baseweb="notification"] { color:#FFFFFF !important; background:#292724 !important; border:1px solid #756b5d; }
[data-testid="stAlert"] *,[data-baseweb="notification"] * { color:#FFFFFF !important; background-color:transparent !important; }
.sm-notice { box-sizing:border-box; width:100%; padding:0.85rem 1rem; margin:0.25rem 0 0.75rem; border:1px solid #756b5d; border-left:4px solid #d8b77f; border-radius:0.6rem; background:#292724; color:#FFFFFF; line-height:1.5; }
[data-testid="stDataFrame"],[data-testid="stTable"] { color:var(--ink); }
small { color:var(--muted); }
@media (max-width:700px) { .block-container { padding:1.25rem 1rem 1.5rem; } [data-testid="stMetric"] { padding:12px; } }
</style>
""", unsafe_allow_html=True)
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "scentmatch_model.joblib"


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def get_tfidf_vectorizer(estimator):
    """Retrouve le TF-IDF même lorsque le pipeline est encapsulé/calibré."""
    pending = [estimator]
    visited = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        steps = getattr(current, "named_steps", {})
        if "tfidf" in steps:
            return steps["tfidf"]
        wrapped = getattr(current, "estimator", None)
        if wrapped is not None:
            pending.append(wrapped)
    raise AttributeError("Le vectoriseur TF-IDF est introuvable dans le modèle chargé.")


def render_notice(message: str) -> None:
    """Affiche une information ou un avertissement avec un contraste maîtrisé."""
    st.markdown(
        f'<div class="sm-notice" role="status">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


st.title("ScentMatch")
st.write("**Quelle famille olfactive correspond à votre description ?**")
st.write("Décrivez les notes de votre parfum ou importez un texte. L'application retourne une famille et les probabilités estimées pour toutes les familles du modèle.")

if not MODEL_PATH.exists():
    st.error("Le modèle entraîné est absent. Exécutez d'abord `python train.py` depuis le dossier ScentMatch.")
    st.code("python -m pip install -r requirements.txt\npython train.py\nstreamlit run app.py", language="bash")
    st.stop()

try:
    bundle = load_model(str(MODEL_PATH))
    model = bundle["model"]
    summary = bundle["summary"]
except Exception as exc:
    st.error(f"Impossible de charger le modèle : {exc}")
    st.stop()

with st.sidebar:
    st.subheader("À propos du projet")
    st.write(f"**Source :** {summary['dataset']}")
    st.write(f"**Modèle retenu :** {summary['modele_retenu']}")
    st.write(f"**Familles apprises :** {len(model.classes_)}")
    st.write(f"**Parfums exploités :** {summary['observations']:,}")
    st.metric("F1 macro sur le test", f"{summary['scores_test']['f1_macro']:.1%}")
    st.caption("Ce score concerne le jeu de test. La famille proposée reste une estimation statistique.")
    st.divider()
    st.markdown("**Conseil de saisie**")
    st.write("Le dataset original est en anglais. Les notes olfactives courantes en français sont partiellement reconnues ; pour le meilleur résultat, utilisez une description en anglais.")

left, right = st.columns([1.08, 1], gap="large")
with left:
    content = st.text_area(
        "Description du parfum ou des notes olfactives",
        value="", height=190,
        placeholder="Exemple : Je cherche un parfum boisé, avec de la vanille et du cèdre…",
        help="Évitez d'écrire directement le nom de la famille à prédire si vous voulez tester le raisonnement du modèle.",
    )
    uploaded = st.file_uploader("Ou importer une description en .txt", type=["txt"])
    if uploaded is not None:
        raw = uploaded.read()
        if len(raw) > 200000:
            st.error("Le fichier est trop volumineux : limite de 200 Ko.")
            st.stop()
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("Le fichier doit être encodé en UTF-8.")
            st.stop()
        st.caption(f"Fichier chargé : {uploaded.name} ({len(content)} caractères)")
    go = st.button("Analyser le parfum", type="primary", use_container_width=True)

with right:
    st.subheader("Résultat de la classification")
    if go:
        clean = normalize_text(content, translate_notes=True)
        if not clean:
            render_notice("Saisissez une description avant de lancer l'analyse.")
        else:
            vectorizer = get_tfidf_vectorizer(model)
            analyzer = vectorizer.build_analyzer()
            recognized_terms = [term for term in analyzer(clean) if term in vectorizer.vocabulary_]
            if not recognized_terms:
                render_notice("Aucun terme de cette description ne correspond au vocabulaire appris par le modèle. Essayez des notes olfactives reconnues, par exemple rose, woody, vanilla ou citrus.")
            else:
                if len(clean.split()) <= 3:
                    st.caption("Une description plus détaillée améliore généralement la qualité de la prédiction.")
                vector = model.predict_proba([clean])[0]
                cls = [str(c) for c in model.classes_]
                probas = pd.DataFrame({"Famille": [family_display(c) for c in cls], "Probabilité": vector})
                probas = probas.sort_values("Probabilité", ascending=False).reset_index(drop=True)
                top = probas.iloc[0]
                st.metric("Famille prédite", top["Famille"])
                st.metric("Probabilité estimée", f"{top['Probabilité']:.1%}")
                st.write("**Vecteur complet de probabilités**")
                chart_rows = probas.sort_values("Probabilité", ascending=True).reset_index(drop=True)
                labels = ["\n".join([name[i:i + 22] for i in range(0, len(name), 22)]) for name in chart_rows["Famille"]]
                values = chart_rows["Probabilité"].to_numpy()
                fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor="#1d1e1c")
                ax.set_facecolor("#1d1e1c")
                bars = ax.barh(range(len(labels)), values, color=["#b89a6b", "#d8b77f", "#c8a874", "#e0c795", "#b89a6b", "#d8b77f", "#c8a874"], height=0.66)
                ax.set_yticks(range(len(labels)), labels=labels, color="#f7f1e8", fontsize=9)
                ax.set_xlim(0, max(float(values.max()) * 1.2, 0.12))
                ax.set_xticks([])
                ax.tick_params(axis="y", length=0, pad=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(axis="x", color="#756b5d", alpha=0.28, linewidth=0.7)
                ax.set_axisbelow(True)
                for bar, value in zip(bars, values):
                    ax.text(bar.get_width() + max(float(values.max()) * 0.025, 0.006), bar.get_y() + bar.get_height() / 2,
                            f"{value:.1%}", va="center", ha="left", color="#f7f1e8", fontsize=9, fontweight="bold")
                fig.subplots_adjust(left=0.48, right=0.97, top=0.97, bottom=0.04)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.dataframe(probas.style.format({"Probabilité": "{:.2%}"}), hide_index=True, use_container_width=True)
                st.caption("Les valeurs totalisent 100 %. Ce sont des estimations du modèle, pas des garanties sur l'odeur réelle.")
    else:
        render_notice("Renseignez une description à gauche, puis cliquez sur « Analyser le parfum ».")

st.divider()
