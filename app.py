"""Thera Bank - Moteur de décision IA (Streamlit)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analyse_marketing

try:
    import shap

    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ============================================================
# CONSTANTES
# ============================================================
BASE_DIR = Path(__file__).parent
# Fichier chargé directement depuis ce chemin ; repli sur le dossier de l'application
FICHIER_ANALYSE = Path("D:/Projets/Projet_Bank_Taux_de_reconvertion/bank.xlsx")
if not FICHIER_ANALYSE.exists():
    FICHIER_ANALYSE = BASE_DIR / "bank.xlsx"
DEVISE = "€"  # À aligner avec l'unité des données d'entraînement
EDUCATION = ["Licence", "Master", "Formation professionnelle", "Autre"]

# Ordre exact des colonnes attendu par le Pipeline (identique à l'entraînement)
FEATURES = [
    "Age", "Revenu", "Family", "Depenses_Mensuelle", "Prêt_Hypothecaire",
    "A_Pret_Hypothecaire", "Education", "Compte_Titre", "Certificat_Depot",
    "Services_Ligne", "CreditCard", "Categorie_Revenu", "Categorie_Age",
]
DERIVED = ["A_Pret_Hypothecaire", "Categorie_Revenu", "Categorie_Age"]
RAW_COLS = [c for c in FEATURES if c not in DERIVED]
BIN_COLS = ["Compte_Titre", "Certificat_Depot", "Services_Ligne", "CreditCard"]
NUM_COLS = ["Age", "Revenu", "Family", "Depenses_Mensuelle", "Prêt_Hypothecaire"]

# ============================================================
# CONFIGURATION DE LA PAGE ET STYLE
# ============================================================
st.set_page_config(
    page_title="Thera Bank | Moteur de décision IA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = BASE_DIR / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DES ARTEFACTS
# ============================================================
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_artifacts():
    return (
        joblib.load(BASE_DIR / "modele_thera_bank.pkl"),
        joblib.load(BASE_DIR / "seuils_revenu.pkl"),
        joblib.load(BASE_DIR / "seuils_age.pkl"),
        joblib.load(BASE_DIR / "seuils_pret_hypothecaire.pkl"),
    )


try:
    model, revenu_bins, age_bins, pret_seuil = load_artifacts()
except Exception as exc:  # noqa: BLE001
    st.error(f"Impossible de charger le modèle : {exc}")
    st.stop()


def load_metrics() -> dict | None:
    path = BASE_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ============================================================
# LOGIQUE MÉTIER (feature engineering, segments, explicabilité)
# ============================================================
def to_bin(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin({"1", "1.0", "oui", "yes", "true", "y"}).astype(int)


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Transforme les données brutes en features du modèle (individuel et lot)."""
    df = raw.copy()
    for col in BIN_COLS:
        df[col] = to_bin(df[col])
    df["A_Pret_Hypothecaire"] = (df["Prêt_Hypothecaire"] > pret_seuil).astype(int)
    df["Categorie_Revenu"] = np.select(
        [df["Revenu"] <= revenu_bins[1], df["Revenu"] <= revenu_bins[2], df["Revenu"] <= revenu_bins[3]],
        ["Faible", "Moyen", "Élevé"],
        default="Très élevé",
    )
    df["Categorie_Age"] = np.select(
        [df["Age"] <= age_bins[1], df["Age"] <= age_bins[2]],
        ["Jeune", "Adulte"],
        default="Agée",
    )
    return df[FEATURES]


def get_segment(p: float, seuil: float) -> tuple[str, str, str]:
    if p >= 0.80:
        return "Très forte probabilité", "🔥", "Contacter prioritairement le client."
    if p >= 0.60:
        return "Forte probabilité", "📈", "Proposer une offre personnalisée."
    if p >= seuil:
        return "Probabilité moyenne", "⚖️", "Inclure dans une campagne de relance."
    return "Faible probabilité", "❄️", "Ne pas cibler prioritairement."


def feature_names(m) -> list[str]:
    return [n.split("__")[-1] for n in m[:-1].get_feature_names_out()]


def global_importances(m) -> pd.DataFrame | None:
    est = m[-1]  # dernière étape du Pipeline
    if not hasattr(est, "feature_importances_"):
        return None
    df = pd.DataFrame({"Variable": feature_names(m), "Importance": est.feature_importances_})
    return df.sort_values("Importance", ascending=False).head(12).iloc[::-1]


@st.cache_resource
def get_explainer(_m):
    return shap.TreeExplainer(_m[-1])


def local_contributions(m, client: pd.DataFrame) -> pd.DataFrame:
    xt = m[:-1].transform(client)
    if hasattr(xt, "toarray"):
        xt = xt.toarray()
    sv = get_explainer(m).shap_values(xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    df = pd.DataFrame({"Variable": feature_names(m), "Contribution": sv[0] * 100})
    df["abs"] = df["Contribution"].abs()
    return df.sort_values("abs", ascending=False).head(8).drop(columns="abs").iloc[::-1]


# ============================================================
# GRAPHIQUES
# ============================================================
def gauge(p: float, seuil: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": " %", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#0B2545"},
            "steps": [
                {"range": [0, seuil * 100], "color": "#E2E8F0"},
                {"range": [seuil * 100, 60], "color": "#BFDBFE"},
                {"range": [60, 80], "color": "#93C5FD"},
                {"range": [80, 100], "color": "#60A5FA"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 3}, "value": seuil * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, colors, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=df[x], y=df[y], orientation="h", marker_color=colors))
    fig.update_layout(
        title=title, height=340, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================
# BARRE LATÉRALE : paramètres + formulaire
# ============================================================
PAGES = ["🎯 Prédiction individuelle", "📦 Prédiction en lot", "📈 Performance du modèle", "ℹ️ À propos",
         "📊 Analyse marketing des données"]

st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("resultat", None)
st.session_state.setdefault("historique", [])


def clear_history() -> None:
    st.session_state["historique"] = []


def go_to_prediction() -> None:
    st.session_state["page"] = PAGES[0]

with st.sidebar:
    st.title("🏦 Profil client")
    seuil = st.slider(
        "Seuil de décision", 0.30, 0.60, 0.50, 0.05,
        help="Probabilité minimale pour considérer qu'un client acceptera le prêt. "
             "À choisir selon le coût d'un faux positif vs d'un faux négatif.",
    )
    st.divider()

    with st.form("form_client"):
        st.subheader("👤 Démographie & revenus")
        age = st.number_input("Âge", 18, 100, 35)
        revenu = st.number_input(f"Revenu annuel ({DEVISE})", 0.0, value=50000.0, step=1000.0)
        family = st.number_input("Membres de la famille", 1, 20, 2)
        education = st.selectbox("Niveau d'éducation", EDUCATION)

        st.subheader("💰 Situation financière")
        depenses = st.number_input(f"Dépenses mensuelles ({DEVISE})", 0.0, value=1500.0, step=100.0)
        pret = st.number_input(f"Prêt hypothécaire ({DEVISE})", 0.0, value=0.0, step=1000.0)

        st.subheader("🏦 Produits bancaires")
        compte_titre = st.selectbox("Compte titre", ["Non", "Oui"])
        certificat = st.selectbox("Certificat de dépôt", ["Non", "Oui"])
        services = st.selectbox("Services en ligne", ["Non", "Oui"])
        carte = st.selectbox("Carte de crédit", ["Non", "Oui"])

        submitted = st.form_submit_button(
            "🚀 Lancer la prédiction", type="primary", width="stretch", on_click=go_to_prediction
        )

if submitted:
    if revenu <= 0:
        st.sidebar.error("Le revenu annuel doit être supérieur à 0.")
    else:
        raw = pd.DataFrame([{
            "Age": age, "Revenu": revenu, "Family": family, "Depenses_Mensuelle": depenses,
            "Prêt_Hypothecaire": pret, "Education": education, "Compte_Titre": compte_titre,
            "Certificat_Depot": certificat, "Services_Ligne": services, "CreditCard": carte,
        }])
        client = engineer(raw)
        proba = float(model.predict_proba(client)[0, 1])
        alertes = []
        if depenses * 12 > revenu:
            alertes.append("Les dépenses annuelles (mensuelles × 12) dépassent le revenu annuel : vérifiez la saisie.")
        st.session_state["resultat"] = {"client": client, "proba": proba, "alertes": alertes}
        st.session_state["historique"].append({
            "Heure": datetime.now().strftime("%H:%M:%S"), "Âge": age, "Revenu": revenu,
            "Éducation": education, "Probabilité": round(proba, 4),
        })
        st.toast("Prédiction terminée", icon="✅")

# ============================================================
# EN-TÊTE ET ONGLETS
# ============================================================
st.markdown(
    """
<div class="app-header">
<div class="app-eyebrow">Intelligence artificielle · Marketing bancaire</div>
<div class="app-title">🏦 Thera Bank <span>· Moteur de décision IA</span></div>
<div class="app-accent"></div>
<p>Cette plateforme a pour objectif d'aider <strong>Thera Bank</strong> à identifier les clients les plus susceptibles d'accepter une offre de prêt personnel. À partir du profil d'un client (âge, revenu, situation familiale, produits bancaires détenus…), un modèle de machine learning estime sa probabilité d'acceptation et recommande l'action commerciale la plus adaptée. Elle permet ainsi aux équipes marketing de <strong>cibler les bons clients, de réduire le coût des campagnes et d'améliorer le taux de conversion</strong>, tout en expliquant chaque score pour faciliter la prise de décision. Les clients peuvent être évalués un par un ou en lot, à partir d'un fichier CSV.</p>
</div>
""",
    unsafe_allow_html=True,
)

page = st.radio("Navigation", PAGES, key="page", horizontal=True, label_visibility="collapsed")

# ------------------------------------------------------------
# Onglet 1 : prédiction individuelle
# ------------------------------------------------------------
if page == PAGES[0]:
    res = st.session_state["resultat"]
    if res is None:
        st.info("Renseignez le profil dans la barre latérale, puis cliquez sur **Lancer la prédiction**.")
    else:
        proba, client = res["proba"], res["client"]
        decision = proba >= seuil
        label, emoji, action = get_segment(proba, seuil)

        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.plotly_chart(gauge(proba, seuil), width="stretch")
            st.caption(f"Trait noir : seuil de décision ({seuil:.0%}).")
        with col2:
            st.metric("Probabilité d'acceptation", f"{proba:.1%}")
            if decision:
                st.success("✅ **Le client est susceptible d'accepter le prêt.**")
            else:
                st.warning("⚠️ **Le client est peu susceptible d'accepter le prêt.**")
            st.markdown("#### 🎯 Recommandation marketing")
            st.info(f"{emoji} **Segment : {label}**\n\n👉 *Action : {action}*")

        for alerte in res["alertes"]:
            st.warning(alerte)

        st.markdown("#### 🔎 Pourquoi ce score ?")
        if SHAP_OK:
            try:
                contrib = local_contributions(model, client)
                colors = ["#0F766E" if v > 0 else "#B91C1C" for v in contrib["Contribution"]]
                st.plotly_chart(
                    bar_chart(contrib, "Contribution", "Variable", colors, "Contribution à la probabilité (points de %)"),
                    width="stretch",
                )
                st.caption("Vert : la variable augmente la probabilité ; rouge : elle la diminue.")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Explication locale indisponible : {exc}")
        else:
            st.caption("Installez `shap` (voir requirements.txt) pour activer l'explication locale.")

        rapport = client.assign(
            Probabilite=round(proba, 4),
            Decision="Acceptation probable" if decision else "Peu probable",
            Segment=label,
        )
        st.download_button(
            "⬇️ Télécharger le rapport client (CSV)",
            rapport.to_csv(index=False, sep=";").encode("utf-8-sig"),
            "rapport_client.csv", "text/csv",
        )

        with st.expander("🔍 Détails techniques"):
            st.markdown("**Données envoyées au modèle**")
            st.dataframe(client.T.rename(columns={0: "Valeur"}).astype(str), width="stretch")

        with st.expander("🕑 Historique de la session"):
            if not st.session_state["historique"]:
                st.caption("Aucune analyse enregistrée pour le moment.")
            else:
                hist = pd.DataFrame(st.session_state["historique"])
                st.dataframe(hist, width="stretch")
                b1, b2 = st.columns(2)
                b1.download_button(
                    "⬇️ Exporter l'historique", hist.to_csv(index=False, sep=";").encode("utf-8-sig"),
                    "historique.csv", "text/csv", width="stretch",
                )
                b2.button("🗑️ Vider l'historique", on_click=clear_history, width="stretch")

# ------------------------------------------------------------
# Onglet 2 : prédiction en lot
# ------------------------------------------------------------
if page == PAGES[1]:
    st.markdown("Scorez plusieurs clients d'un coup à partir d'un fichier CSV.")
    template = pd.DataFrame([{
        "Age": 35, "Revenu": 50000, "Family": 2, "Depenses_Mensuelle": 1500, "Prêt_Hypothecaire": 0,
        "Education": "Licence", "Compte_Titre": "Non", "Certificat_Depot": "Non",
        "Services_Ligne": "Oui", "CreditCard": "Non",
    }])
    st.download_button(
        "📄 Télécharger le modèle de fichier", template.to_csv(index=False, sep=";").encode("utf-8-sig"),
        "modele_clients.csv", "text/csv",
    )
    fichier = st.file_uploader("Fichier clients (CSV)", type="csv")

    if fichier is not None:
        try:
            try:
                data = pd.read_csv(fichier, sep=None, engine="python", encoding="utf-8-sig")
            except UnicodeDecodeError:  # fichier enregistré par Excel en ANSI
                fichier.seek(0)
                data = pd.read_csv(fichier, sep=None, engine="python", encoding="cp1252")
            data.columns = data.columns.str.strip()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Lecture impossible : {exc}")
            st.stop()

        manquantes = [c for c in RAW_COLS if c not in data.columns]
        if manquantes:
            st.error(f"Colonnes manquantes : {', '.join(manquantes)}")
        else:
            data[NUM_COLS] = data[NUM_COLS].apply(pd.to_numeric, errors="coerce")
            invalides = data[RAW_COLS].isna().any(axis=1).sum()
            inconnues = sorted(set(data["Education"].dropna()) - set(EDUCATION))
            if invalides:
                st.error(f"{invalides} ligne(s) contiennent des valeurs manquantes ou non numériques.")
            elif inconnues:
                st.error(f"Modalités d'éducation inconnues : {inconnues}. Attendu : {EDUCATION}")
            else:
                with st.spinner("Calcul des scores…"):
                    probas = model.predict_proba(engineer(data[RAW_COLS]))[:, 1]
                sortie = data.copy()
                sortie["Probabilité"] = probas.round(4)
                sortie["Décision"] = np.where(probas >= seuil, "Acceptation probable", "Peu probable")
                sortie["Segment"] = [get_segment(p, seuil)[0] for p in probas]

                m1, m2, m3 = st.columns(3)
                m1.metric("Clients analysés", f"{len(sortie):,}".replace(",", " "))
                m2.metric("Acceptation probable", f"{(probas >= seuil).mean():.1%}")
                m3.metric("Probabilité moyenne", f"{probas.mean():.1%}")

                ordre = ["Très forte probabilité", "Forte probabilité", "Probabilité moyenne", "Faible probabilité"]
                comptes = sortie["Segment"].value_counts().reindex(ordre, fill_value=0)
                fig_seg = go.Figure(go.Bar(
                    x=comptes.index, y=comptes.values, text=comptes.values, textposition="outside",
                    marker_color=["#0B2545", "#1D4ED8", "#60A5FA", "#CBD5E1"],
                ))
                fig_seg.update_layout(
                    title="Répartition des clients par segment", height=320, yaxis_title="Clients",
                    margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_seg, width="stretch")
                st.dataframe(
                    sortie.sort_values("Probabilité", ascending=False),
                    width="stretch",
                    column_config={"Probabilité": st.column_config.ProgressColumn(
                        "Probabilité", min_value=0.0, max_value=1.0, format="%.1f")},
                )
                st.download_button(
                    "⬇️ Télécharger les résultats (CSV)",
                    sortie.to_csv(index=False, sep=";").encode("utf-8-sig"),
                    "resultats_scoring.csv", "text/csv",
                )

# ------------------------------------------------------------
# Onglet 3 : performance du modèle
# ------------------------------------------------------------
if page == PAGES[2]:
    metrics = load_metrics()
    if metrics is None:
        st.info(
            "Aucun fichier `metrics.json` trouvé. Générez-le depuis votre notebook "
            "(AUC, précision, rappel, F1, matrice de confusion, courbe ROC) et placez-le "
            "à côté de `app.py` pour afficher ici les performances sur le jeu de test."
        )
    else:
        cols = st.columns(4)
        for col, (cle, nom) in zip(cols, [("auc", "AUC"), ("precision", "Précision"),
                                          ("recall", "Rappel"), ("f1", "F1-score")]):
            if cle in metrics:
                col.metric(nom, f"{metrics[cle]:.3f}")
        st.caption(
            f"Version : {metrics.get('model_version', 'n/c')} · entraîné le {metrics.get('trained_on', 'n/c')} "
            f"· seuil d'évaluation : {metrics.get('threshold', 'n/c')}"
        )
        g1, g2 = st.columns(2)
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Refus prédit", "Acceptation prédite"], y=["Refus réel", "Acceptation réelle"],
                text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
            fig.update_layout(title="Matrice de confusion", height=340, margin=dict(l=10, r=10, t=40, b=10))
            g1.plotly_chart(fig, width="stretch")
        if "roc" in metrics:
            fig = go.Figure()
            fig.add_scatter(x=metrics["roc"]["fpr"], y=metrics["roc"]["tpr"], name="Modèle", line_color="#1D4ED8")
            fig.add_scatter(x=[0, 1], y=[0, 1], name="Hasard", line=dict(dash="dash", color="grey"))
            fig.update_layout(title="Courbe ROC", height=340, xaxis_title="Taux de faux positifs",
                              yaxis_title="Taux de vrais positifs", margin=dict(l=10, r=10, t=40, b=10))
            g2.plotly_chart(fig, width="stretch")

    st.markdown("#### Importance globale des variables")
    imp = global_importances(model)
    if imp is None:
        st.caption("Ce modèle ne fournit pas d'importance des variables.")
    else:
        st.plotly_chart(bar_chart(imp, "Importance", "Variable", "#1D4ED8", "Top 12 des variables"), width="stretch")

# ------------------------------------------------------------
# Onglet 4 : à propos
# ------------------------------------------------------------
if page == PAGES[3]:
    st.markdown(f"""
**Objectif** : estimer la probabilité qu'un client de Thera Bank accepte une offre de prêt personnel,
afin de prioriser les actions marketing.

**Modèle** : `{type(model[-1]).__name__}` dans un Pipeline scikit-learn (prétraitement + classifieur).

**Utilisation** : la décision « acceptation probable » repose sur un seuil unique, réglable dans la barre latérale.
Les segments marketing (très forte, forte, moyenne, faible) en découlent.

**Limites** :
- Les probabilités dépendent de la calibration du modèle et de la représentativité des données d'entraînement.
- Le score est une aide à la décision commerciale ; il ne doit pas fonder seul une décision de crédit.
- Les montants sont exprimés en {DEVISE} : ils doivent correspondre à l'unité des données d'entraînement.
""")

# ------------------------------------------------------------
# Onglet 5 : analyse marketing des données
# ------------------------------------------------------------
if page == PAGES[4]:
    analyse_marketing.afficher(FICHIER_ANALYSE)

# ============================================================
# PIED DE PAGE
# ============================================================
st.divider()
st.markdown(
    f"<div class='footer'>Thera Bank © {datetime.now().year} · Les prédictions reposent sur un modèle "
    "de machine learning et ne constituent pas un conseil financier.</div>",
    unsafe_allow_html=True,
)