"""Analyse marketing des données Thera Bank (onglet de l'application Streamlit).

Aucun nettoyage ni modélisation : le fichier Excel est lu tel quel et la page se limite à des
agrégations descriptives (taux de conversion par segment) pour répondre aux questions business.
"""
from __future__ import annotations

import unicodedata
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.stats import chi2_contingency
except ImportError:  # scipy est optionnel (test de significativité)
    chi2_contingency = None

NUIT, BLEU, CIEL, ARDOISE, VERT, ROUGE = "#0B2545", "#1D4ED8", "#93C5FD", "#CBD5E1", "#0F766E", "#B91C1C"
ECHELLE = [[0, "#BFDBFE"], [1, NUIT]]

REQUISES = ["Age", "Revenu", "Family", "Education", "Prêt_Hypothecaire", "Accepte_Pret",
            "Compte_Titre", "Certificat_Depot", "Services_Ligne", "CreditCard"]
NUMERIQUES = ["Age", "Experience", "Revenu", "Family", "Depenses_Mensuelle", "Prêt_Hypothecaire"]
BINAIRES = {"Services en ligne": "Services_Ligne", "Certificat de dépôt": "Certificat_Depot",
            "Compte titre": "Compte_Titre", "Carte de crédit": "CreditCard"}
PRODUITS = ["Services en ligne", "Certificat de dépôt", "Compte titre", "Carte de crédit", "Prêt hypothécaire"]
DIMS = ["Tranche de revenu", "Tranche d'âge", "Éducation", "Taille de la famille"] + PRODUITS

QUESTIONS = [
    "Quels clients sont les plus susceptibles d'accepter un prêt personnel ? (question principale)",
    "Les clients à revenus élevés acceptent-ils davantage les prêts ?",
    "L'éducation influence-t-elle l'acceptation ?",
    "Les clients utilisant les services en ligne sont-ils plus susceptibles d'accepter ?",
    "La détention d'un certificat de dépôt est-elle associée à l'acceptation ?",
    "Les clients ayant déjà un prêt hypothécaire sont-ils plus susceptibles d'accepter ?",
    "L'âge influence-t-il la probabilité d'acceptation ?",
    "Les clients avec une famille plus nombreuse sont-ils plus réceptifs ?",
    "Quels profils de clients sont les plus intéressants pour le marketing ?",
]


# ============================================================
# Outils de format
# ============================================================
def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f} %".replace(".", ",")


def nombre(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def style(fig: go.Figure, hauteur: int = 380, titre: str | None = None) -> go.Figure:
    fig.update_layout(
        height=hauteur, separators=", ", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60 if titre else 20, b=10), font=dict(size=13),
        legend=dict(orientation="h", y=1.1, x=0), hoverlabel=dict(bgcolor="white"),
    )
    if titre:
        fig.update_layout(title=dict(text=titre, x=0, font=dict(size=15)))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)")
    return fig


def fois(x: float) -> str:
    return f"×{x:.1f}".replace(".", ",")


def lecture(texte: str) -> None:
    st.info("💡 **Lecture marketing.** " + texte)


def comparer(nom_a: str, a: float, nom_b: str, b: float) -> str:
    if b <= 0 or a <= 0:
        return f"{nom_a} : {pct(a)} ; {nom_b} : {pct(b)}."
    r = a / b
    if 0.95 <= r <= 1.05:
        return f"{nom_a} ({pct(a)}) et {nom_b} ({pct(b)}) convertissent à un niveau comparable."
    sens = "plus" if r > 1 else "moins"
    rapport = f"{r:.2f}".replace(".", ",")
    return f"{nom_a} convertissent {sens} ({pct(a)}) que {nom_b} ({pct(b)}), soit un rapport de ×{rapport}."


def norm(c) -> str:
    """Nom de colonne sans accents, en minuscules, avec _ à la place des espaces."""
    t = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode()
    return t.strip().lower().replace(" ", "_")


# ============================================================
# Lecture et préparation minimale (agrégation uniquement)
# ============================================================
@st.cache_data(show_spinner="Chargement des données…")
def charger(chemin: str, horodatage: float) -> pd.DataFrame:
    """Lit le classeur Excel et repère la table clients (feuille + ligne d'en-tête) sans rien modifier."""
    requises = {norm(c) for c in REQUISES}
    renommage = {norm(c): c for c in REQUISES + ["Experience", "Depenses_Mensuelle"]}
    feuilles = pd.read_excel(chemin, sheet_name=None, header=None)
    apercu = []
    for nom, brut in feuilles.items():
        for i in range(min(len(brut), 30)):
            ligne = brut.iloc[i].astype(str)
            if requises <= {norm(v) for v in ligne}:
                df = brut.iloc[i + 1:].copy()
                df.columns = [renommage.get(norm(c), str(c).strip()) for c in ligne]
                df = df.loc[:, df.columns != "nan"].dropna(how="all").reset_index(drop=True)
                return df.infer_objects()
        non_vide = brut.dropna(how="all").head(1)
        valeurs = [str(v) for v in non_vide.iloc[0].dropna()][:12] if len(non_vide) else []
        apercu.append(f"feuille « {nom} » ({brut.shape[0]} lignes × {brut.shape[1]} colonnes), première ligne : {valeurs}")
    raise ValueError("aucune feuille ne contient les colonnes attendues. " + " ; ".join(apercu))


def oui(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"oui", "1", "1.0", "yes", "true", "y"})


def tranches(s: pd.Series, q: int) -> pd.Series:
    """Découpe en tranches de taille égale (quantiles) avec des libellés lisibles."""
    _, bornes = pd.qcut(s, q, retbins=True, duplicates="drop")
    noms = [f"T{i + 1} · {nombre(bornes[i])}–{nombre(bornes[i + 1])}" for i in range(len(bornes) - 1)]
    return pd.qcut(s, q, labels=noms, duplicates="drop")


@st.cache_data(show_spinner=False)
def preparer(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["y"] = oui(df["Accepte_Pret"]).astype(int)
    for col in NUMERIQUES:
        if col in df.columns:
            d[col] = pd.to_numeric(df[col], errors="coerce")

    d["Tranche de revenu"] = tranches(d["Revenu"], 5)
    d["Tranche d'âge"] = pd.cut(d["Age"], [0, 29, 39, 49, 59, 200],
                                labels=["< 30 ans", "30–39 ans", "40–49 ans", "50–59 ans", "60 ans et +"])
    edu = df["Education"].astype(str)
    ordre = [e for e in ["Licence", "Master", "Formation professionnelle", "Autre"] if e in set(edu)]
    ordre += sorted(set(edu) - set(ordre))
    d["Éducation"] = pd.Categorical(edu, categories=ordre, ordered=True)
    cats = [str(int(v)) for v in sorted(d["Family"].dropna().unique())]
    d["Taille de la famille"] = pd.Categorical(
        d["Family"].map(lambda v: str(int(v)) if pd.notna(v) else None), categories=cats, ordered=True)
    for nom, col in BINAIRES.items():
        d[nom] = pd.Categorical(np.where(oui(df[col]), "Oui", "Non"), categories=["Non", "Oui"], ordered=True)
    d["Prêt hypothécaire"] = pd.Categorical(
        np.where(d["Prêt_Hypothecaire"] > 0, "Oui", "Non"), categories=["Non", "Oui"], ordered=True)
    nb = sum((d[n] == "Oui").astype(int) for n in BINAIRES)
    d["Nombre de produits"] = pd.Categorical(nb.astype(str), categories=[str(i) for i in range(5)], ordered=True)
    return d


def agreger(d: pd.DataFrame, col: str, base: float | None = None) -> pd.DataFrame:
    base = d["y"].mean() if base is None else base
    g = d.groupby(col, observed=True)["y"].agg(effectif="size", acceptants="sum").reset_index()
    g = g.rename(columns={col: "Modalité"})
    g["Modalité"] = g["Modalité"].astype(str)
    g["taux"] = g["acceptants"] / g["effectif"]
    g["lift"] = g["taux"] / base if base else np.nan
    return g


def tableau(g: pd.DataFrame, nom: str) -> None:
    t = g.rename(columns={"Modalité": nom, "effectif": "Clients", "acceptants": "Acceptants"}).copy()
    t["Taux de conversion"] = t.pop("taux") * 100
    t["Indice"] = t.pop("lift")
    st.dataframe(
        t, hide_index=True, width="stretch",
        column_config={
            "Clients": st.column_config.NumberColumn(format="%d"),
            "Acceptants": st.column_config.NumberColumn(format="%d"),
            "Taux de conversion": st.column_config.ProgressColumn(
                "Taux de conversion", min_value=0.0, max_value=float(max(t["Taux de conversion"].max() * 1.15, 1.0)),
                format="%.1f%%"),
            "Indice": st.column_config.NumberColumn("Indice vs moyenne", format="×%.2f"),
        },
    )


def signif(d: pd.DataFrame, col: str) -> None:
    if chi2_contingency is None:
        return
    tab = pd.crosstab(d[col], d["y"])
    tab = tab.loc[tab.sum(axis=1) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return
    try:
        p = chi2_contingency(tab)[1]
    except ValueError:
        return
    verdict = "significative" if p < 0.05 else "non significative"
    st.caption(f"Test du χ² d'indépendance : p = {p:.3g} → association {verdict} au seuil de 5 %.")


# ============================================================
# Graphiques réutilisables
# ============================================================
def graphique_taux(g: pd.DataFrame, titre: str, taux_global: float, hauteur: int = 380) -> go.Figure:
    textes = [f"{pct(t)}<br>n = {nombre(n)}" for t, n in zip(g["taux"], g["effectif"])]
    fig = go.Figure(go.Bar(
        x=g["Modalité"], y=g["taux"], text=textes, textposition="outside",
        marker=dict(color=g["taux"], colorscale=ECHELLE, cmin=0, cmax=max(g["taux"].max(), 0.01), showscale=False),
        customdata=np.stack([g["effectif"], g["acceptants"], g["lift"]], axis=-1),
        hovertemplate=("<b>%{x}</b><br>Taux de conversion : %{y:.1%}<br>Clients : %{customdata[0]:,}"
                       "<br>Acceptants : %{customdata[1]:,}<br>Indice vs moyenne : ×%{customdata[2]:.2f}<extra></extra>"),
    ))
    fig.add_hline(y=taux_global, line_dash="dash", line_color=ROUGE,
                  annotation_text=f"Moyenne : {pct(taux_global)}", annotation_position="top left")
    fig.update_yaxes(tickformat=".0%", range=[0, max(g["taux"].max(), taux_global, 0.02) * 1.4],
                     title="Taux de conversion")
    fig.update_xaxes(type="category")
    return style(fig, hauteur, titre)


def graphique_volumes(g: pd.DataFrame, titre: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=g["Modalité"], y=g["acceptants"], name="Acceptants", marker_color=BLEU,
                hovertemplate="%{x}<br>Acceptants : %{y:,}<extra></extra>")
    fig.add_bar(x=g["Modalité"], y=g["effectif"] - g["acceptants"], name="Non-acceptants", marker_color=ARDOISE,
                hovertemplate="%{x}<br>Non-acceptants : %{y:,}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Nombre de clients")
    fig.update_xaxes(type="category")
    return style(fig, 380, titre)


def distribution(d: pd.DataFrame, col: str, etiquette: str) -> tuple[go.Figure, go.Figure]:
    t = d[[col, "y"]].assign(Groupe=np.where(d["y"] == 1, "Acceptants", "Non-acceptants"))
    couleurs = {"Acceptants": BLEU, "Non-acceptants": ARDOISE}
    h = px.histogram(t, x=col, color="Groupe", barmode="overlay", histnorm="percent", nbins=30,
                     opacity=0.7, color_discrete_map=couleurs, labels={col: etiquette})
    h.update_yaxes(title="Part du groupe (%)")
    b = px.box(t, x="Groupe", y=col, color="Groupe", color_discrete_map=couleurs, points="outliers",
               labels={col: etiquette})
    b.update_layout(showlegend=False)
    return style(h, 360, f"Distribution : {etiquette}"), style(b, 360, f"Dispersion : {etiquette}")


def courbe_gains(d: pd.DataFrame, col: str, etiquette: str):
    tri = d.sort_values(col, ascending=False)["y"].to_numpy()
    total = tri.sum()
    if total == 0:
        return None, None
    cum = np.cumsum(tri) / total
    part = np.arange(1, len(tri) + 1) / len(tri)
    idx = np.unique(np.linspace(0, len(tri) - 1, min(len(tri), 300)).astype(int))
    k = max(int(0.3 * len(tri)), 1)
    gain30 = float(cum[k - 1])
    fig = go.Figure()
    fig.add_scatter(x=part[idx], y=cum[idx], mode="lines", name=f"Ciblage par {etiquette} décroissant",
                    line=dict(color=BLEU, width=3), fill="tozeroy", fillcolor="rgba(29,78,216,0.10)",
                    hovertemplate="Clients contactés : %{x:.0%}<br>Acceptants captés : %{y:.0%}<extra></extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ciblage aléatoire", line=dict(color=ROUGE, dash="dash"))
    fig.add_scatter(x=[0.3], y=[gain30], mode="markers+text", showlegend=False, textposition="bottom right",
                    text=[f"30 % ciblés → {pct(gain30, 0)} des acceptants"], marker=dict(size=12, color=NUIT))
    fig.update_xaxes(tickformat=".0%", title="Part des clients contactés")
    fig.update_yaxes(tickformat=".0%", title="Part des acceptants captés")
    return style(fig, 400, f"Courbe de gain : contacter d'abord les clients au {etiquette} le plus élevé"), gain30


def indices_profil(d: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    libelles = {"Age": "Âge moyen", "Experience": "Expérience moyenne", "Revenu": "Revenu moyen",
                "Family": "Taille moyenne de la famille", "Depenses_Mensuelle": "Dépenses mensuelles moyennes",
                "Prêt_Hypothecaire": "Prêt hypothécaire moyen"}
    for col, nom in libelles.items():
        if col in d.columns:
            a, n, t = d.loc[d.y == 1, col].mean(), d.loc[d.y == 0, col].mean(), d[col].mean()
            lignes.append((nom, "Moyenne", a, n, t, a / t * 100 if t else np.nan))
    for dim in PRODUITS:
        a = (d.loc[d.y == 1, dim] == "Oui").mean() * 100
        n = (d.loc[d.y == 0, dim] == "Oui").mean() * 100
        t = (d[dim] == "Oui").mean() * 100
        lignes.append((f"Détention : {dim}", "% de clients", a, n, t, a / t * 100 if t else np.nan))
    return pd.DataFrame(lignes, columns=["Indicateur", "Type", "Acceptants", "Non-acceptants", "Ensemble", "Indice"])


def ecarts_variables(d: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    lignes = []
    for dim in DIMS:
        g = agreger(d, dim)
        g = g[g["effectif"] >= min_n]
        if len(g) < 2:
            continue
        haut, bas = g.loc[g["taux"].idxmax()], g.loc[g["taux"].idxmin()]
        lignes.append({"Variable": dim, "Écart (points)": (haut["taux"] - bas["taux"]) * 100,
                       "Meilleure modalité": haut["Modalité"], "Taux max": haut["taux"],
                       "Modalité la plus faible": bas["Modalité"], "Taux min": bas["taux"]})
    return pd.DataFrame(lignes).sort_values("Écart (points)", ascending=False)


@st.cache_data(show_spinner="Calcul des profils clients…")
def profils(d: pd.DataFrame, min_n: int) -> pd.DataFrame:
    morceaux = []
    for a, b in combinations(DIMS, 2):
        g = d.groupby([a, b], observed=True)["y"].agg(effectif="size", acceptants="sum").reset_index()
        g = g[g["effectif"] >= min_n]
        g["Profil"] = a + " : " + g[a].astype(str) + "  |  " + b + " : " + g[b].astype(str)
        morceaux.append(g[["Profil", "effectif", "acceptants"]])
    out = pd.concat(morceaux, ignore_index=True)
    out["taux"] = out["acceptants"] / out["effectif"]
    out["lift"] = out["taux"] / d["y"].mean()
    return out.sort_values(["taux", "effectif"], ascending=False).reset_index(drop=True)


# ============================================================
# Onglets de l'analyse
# ============================================================
def onglet_vue(d: pd.DataFrame) -> None:
    n, acc = len(d), int(d["y"].sum())
    tg = acc / n
    ra, rn = d.loc[d.y == 1, "Revenu"].mean(), d.loc[d.y == 0, "Revenu"].mean()
    aa, an = d.loc[d.y == 1, "Age"].mean(), d.loc[d.y == 0, "Age"].mean()
    k = st.columns(5)
    k[0].metric("Clients analysés", nombre(n))
    k[1].metric("Acceptants du prêt", nombre(acc))
    k[2].metric("Taux de conversion global", pct(tg))
    k[3].metric("Revenu moyen des acceptants", nombre(ra), f"{ra / rn - 1:+.0%} vs non-acceptants", delta_color="off")
    k[4].metric("Âge moyen des acceptants", f"{aa:.1f} ans".replace(".", ","),
                f"{aa - an:+.1f} an(s) vs non-acceptants".replace(".", ","), delta_color="off")

    st.markdown("### Qui sont les clients qui acceptent le prêt ?")
    c1, c2 = st.columns([2, 3])
    donut = go.Figure(go.Pie(labels=["Acceptants", "Non-acceptants"], values=[acc, n - acc], hole=0.65,
                             marker=dict(colors=[BLEU, ARDOISE]), textinfo="label+percent", sort=False))
    donut.add_annotation(text=f"<b>{pct(tg)}</b><br>de conversion", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    donut.update_layout(showlegend=False)
    c1.plotly_chart(style(donut, 380, "Répartition des clients"), width="stretch")

    ind = indices_profil(d).sort_values("Indice")
    barres = go.Figure(go.Bar(
        x=ind["Indice"], y=ind["Indicateur"], orientation="h",
        marker_color=[BLEU if v >= 100 else "#94A3B8" for v in ind["Indice"]],
        text=[f"{v:.0f}" for v in ind["Indice"]], textposition="outside",
        hovertemplate="%{y}<br>Indice : %{x:.0f}<extra></extra>"))
    barres.add_vline(x=100, line_dash="dash", line_color=ROUGE, annotation_text="Clientèle moyenne = 100")
    barres.update_xaxes(range=[0, max(ind["Indice"].max() * 1.15, 120)], title="Indice (base 100 = ensemble des clients)")
    c2.plotly_chart(style(barres, 380, "Profil des acceptants comparé à la clientèle moyenne"), width="stretch")

    st.markdown("**Profil comparé : acceptants vs non-acceptants**")
    st.dataframe(
        indices_profil(d), hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.1f")
                       for c in ["Acceptants", "Non-acceptants", "Ensemble", "Indice"]})
    lecture("Un indice supérieur à 100 signale une caractéristique **sur-représentée** chez les acceptants : "
            "c'est un critère de ciblage pertinent. Un indice inférieur à 100 indique une caractéristique plus rare chez eux.")

    st.markdown("### Quelles variables discriminent le plus l'acceptation ?")
    ec = ecarts_variables(d)
    fig = go.Figure(go.Bar(
        x=ec["Écart (points)"][::-1], y=ec["Variable"][::-1], orientation="h", marker_color=BLEU,
        text=[f"{v:.1f} pts".replace(".", ",") for v in ec["Écart (points)"][::-1]], textposition="outside",
        customdata=np.stack([ec["Meilleure modalité"][::-1], ec["Taux max"][::-1] * 100,
                             ec["Modalité la plus faible"][::-1], ec["Taux min"][::-1] * 100], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Meilleure : %{customdata[0]} (%{customdata[1]:.1f} %)"
                       "<br>Plus faible : %{customdata[2]} (%{customdata[3]:.1f} %)<extra></extra>")))
    fig.update_xaxes(title="Écart de conversion entre la meilleure et la moins bonne modalité (points de %)")
    st.plotly_chart(style(fig, 420, "Pouvoir discriminant des variables"), width="stretch")
    st.dataframe(ec, hide_index=True, width="stretch", column_config={
        "Écart (points)": st.column_config.NumberColumn(format="%.1f"),
        "Taux max": st.column_config.NumberColumn(format="percent"),
        "Taux min": st.column_config.NumberColumn(format="percent")})
    lecture("Plus l'écart est grand, plus la variable permet de **séparer** les clients très réceptifs des clients peu réceptifs. "
            "Les variables en haut du classement sont les premiers critères à retenir pour cibler une campagne.")


def onglet_revenu_age(d: pd.DataFrame) -> None:
    tg = d["y"].mean()
    st.markdown("### Q1 · Les clients à revenus élevés acceptent-ils davantage les prêts ?")
    g = agreger(d, "Tranche de revenu")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(graphique_taux(g, "Taux de conversion par tranche de revenu (quintiles)", tg), width="stretch")
    with c2:
        tableau(g, "Tranche de revenu")
        signif(d, "Tranche de revenu")
    h, b = distribution(d, "Revenu", "Revenu")
    c3, c4 = st.columns(2)
    c3.plotly_chart(h, width="stretch")
    c4.plotly_chart(b, width="stretch")
    fig, gain = courbe_gains(d, "Revenu", "revenu")
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    haut, bas, best = g.iloc[-1], g.iloc[0], g.loc[g["taux"].idxmax()]
    texte = (f"La conversion passe de **{pct(bas['taux'])}** (tranche la plus basse) à **{pct(haut['taux'])}** "
             f"(tranche la plus haute) ; la meilleure tranche est **{best['Modalité']}** ({pct(best['taux'])}).")
    if gain is not None:
        texte += f" Contacter en priorité les 30 % de clients aux revenus les plus élevés permettrait de capter **{pct(gain, 0)}** des acceptants."
    lecture(texte)

    st.divider()
    st.markdown("### Q6 · L'âge influence-t-il la probabilité d'acceptation ?")
    ga = agreger(d, "Tranche d'âge")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(graphique_taux(ga, "Taux de conversion par tranche d'âge", tg), width="stretch")
    with c2:
        tableau(ga, "Tranche d'âge")
        signif(d, "Tranche d'âge")
    h, b = distribution(d, "Age", "Âge")
    c3, c4 = st.columns(2)
    c3.plotly_chart(h, width="stretch")
    c4.plotly_chart(b, width="stretch")
    gb = ga[ga["effectif"] >= 30]
    best = gb.loc[gb["taux"].idxmax()] if len(gb) else ga.loc[ga["taux"].idxmax()]
    lecture(f"La tranche d'âge la plus réceptive est **{best['Modalité']}** ({pct(best['taux'])}, "
            f"soit {fois(best['lift'])} la moyenne).")


def onglet_education_famille(d: pd.DataFrame) -> None:
    tg = d["y"].mean()
    st.markdown("### Q2 · L'éducation influence-t-elle l'acceptation ?")
    g = agreger(d, "Éducation")
    c1, c2 = st.columns(2)
    c1.plotly_chart(graphique_taux(g, "Taux de conversion par niveau d'éducation", tg), width="stretch")
    c2.plotly_chart(graphique_volumes(g, "Acceptants et non-acceptants par niveau d'éducation"), width="stretch")
    tableau(g, "Niveau d'éducation")
    signif(d, "Éducation")
    best, worst = g.loc[g["taux"].idxmax()], g.loc[g["taux"].idxmin()]
    lecture(f"Le niveau **{best['Modalité']}** convertit le mieux ({pct(best['taux'])}) et **{worst['Modalité']}** le moins ({pct(worst['taux'])}). "
            "Le volume de clients de chaque niveau (graphique de droite) permet d'arbitrer entre rendement et taille de cible.")

    st.divider()
    st.markdown("### Q7 · Les clients avec une famille plus nombreuse sont-ils plus réceptifs ?")
    gf = agreger(d, "Taille de la famille")
    c1, c2 = st.columns(2)
    c1.plotly_chart(graphique_taux(gf, "Taux de conversion selon la taille de la famille", tg), width="stretch")
    c2.plotly_chart(graphique_volumes(gf, "Acceptants et non-acceptants selon la taille de la famille"), width="stretch")
    tableau(gf, "Membres de la famille")
    signif(d, "Taille de la famille")
    best = gf.loc[gf["taux"].idxmax()]
    lecture(f"La conversion est la plus forte pour les foyers de **{best['Modalité']} personne(s)** ({pct(best['taux'])}). "
            "Vérifiez si la tendance est régulière avec la taille du foyer avant d'en faire un critère de ciblage.")


def bloc_produit(d: pd.DataFrame, dim: str, question: str, sujet: str) -> None:
    tg = d["y"].mean()
    st.markdown(f"### {question}")
    g = agreger(d, dim)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(graphique_taux(g, f"Taux de conversion selon : {dim}", tg, 340), width="stretch")
    with c2:
        tableau(g, dim)
        signif(d, dim)
    o, n = g[g["Modalité"] == "Oui"], g[g["Modalité"] == "Non"]
    if len(o) and len(n):
        lecture(comparer(sujet, float(o["taux"].iloc[0]), "les autres clients", float(n["taux"].iloc[0])))


def onglet_produits(d: pd.DataFrame) -> None:
    tg = d["y"].mean()
    st.markdown("### Vue d'ensemble : la détention d'un produit est-elle associée à l'acceptation ?")
    lignes = []
    for dim in PRODUITS:
        for _, r in agreger(d, dim).iterrows():
            lignes.append({"Produit": dim, "Détention": r["Modalité"], "taux": r["taux"], "effectif": r["effectif"]})
    t = pd.DataFrame(lignes)
    fig = go.Figure()
    for modalite, couleur in [("Non", ARDOISE), ("Oui", BLEU)]:
        s = t[t["Détention"] == modalite]
        fig.add_bar(x=s["Produit"], y=s["taux"], name=f"Détention : {modalite}", marker_color=couleur,
                    text=[pct(v) for v in s["taux"]], textposition="outside", customdata=s["effectif"],
                    hovertemplate="%{x}<br>Taux : %{y:.1%}<br>Clients : %{customdata:,}<extra></extra>")
    fig.add_hline(y=tg, line_dash="dash", line_color=ROUGE, annotation_text=f"Moyenne : {pct(tg)}",
                  annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", range=[0, t["taux"].max() * 1.35], title="Taux de conversion")
    st.plotly_chart(style(fig, 400, "Taux de conversion selon la détention de chaque produit"), width="stretch")

    st.divider()
    bloc_produit(d, "Services en ligne", "Q3 · Les clients utilisant les services en ligne sont-ils plus susceptibles d'accepter ?",
                 "Les clients utilisant les services en ligne")
    st.divider()
    bloc_produit(d, "Certificat de dépôt", "Q4 · La détention d'un certificat de dépôt est-elle associée à l'acceptation ?",
                 "Les détenteurs d'un certificat de dépôt")
    st.divider()
    bloc_produit(d, "Prêt hypothécaire", "Q5 · Les clients ayant déjà un prêt hypothécaire sont-ils plus susceptibles d'accepter ?",
                 "Les clients ayant un prêt hypothécaire")
    dd = d[d["Prêt_Hypothecaire"] > 0]
    if len(dd) >= 40:
        tmp = dd.assign(Tranche=tranches(dd["Prêt_Hypothecaire"], 4))
        g2 = agreger(tmp, "Tranche", base=tg)
        c1, c2 = st.columns(2)
        c1.plotly_chart(graphique_taux(g2, "Conversion selon le montant du prêt hypothécaire (détenteurs)", tg),
                        width="stretch")
        c2.plotly_chart(graphique_volumes(g2, "Clients détenteurs par tranche de montant"), width="stretch")

    st.divider()
    st.markdown("### Autres produits et effet cumulé")
    c1, c2 = st.columns(2)
    with c1:
        bloc_produit(d, "Compte titre", "Compte titre", "Les détenteurs d'un compte titre")
    with c2:
        bloc_produit(d, "Carte de crédit", "Carte de crédit", "Les détenteurs d'une carte de crédit")
    gn = agreger(d, "Nombre de produits")
    st.plotly_chart(graphique_taux(gn, "Taux de conversion selon le nombre de produits détenus (hors prêt hypothécaire)", tg),
                    width="stretch")
    lecture("Si la conversion augmente avec le nombre de produits détenus, les clients **déjà équipés** sont une cible naturelle "
            "(vente croisée) ; sinon, privilégiez les critères les plus discriminants.")


def onglet_profils(d: pd.DataFrame) -> None:
    tg = d["y"].mean()
    st.markdown("### Question principale · Quels clients sont les plus susceptibles d'accepter ?")
    lignes = []
    for dim in DIMS:
        g = agreger(d, dim)
        g = g[g["effectif"] >= 30]
        if len(g):
            b = g.loc[g["taux"].idxmax()]
            lignes.append({"Critère": dim, "Modalité la plus convertissante": b["Modalité"],
                           "Clients": int(b["effectif"]), "Taux": b["taux"] * 100, "Indice": b["lift"]})
    fiche = pd.DataFrame(lignes).sort_values("Taux", ascending=False)
    st.markdown("**Fiche du client à fort potentiel** (meilleure modalité de chaque critère)")
    st.dataframe(fiche, hide_index=True, width="stretch", column_config={
        "Taux": st.column_config.ProgressColumn("Taux de conversion", min_value=0.0, max_value=float(fiche["Taux"].max() * 1.15),
                                                format="%.1f%%"),
        "Indice": st.column_config.NumberColumn("Indice vs moyenne", format="×%.2f")})

    st.divider()
    st.markdown("### Q8 · Quels profils sont les plus intéressants pour le marketing ?")
    st.markdown("**Carte de chaleur interactive : croisez deux critères**")
    c1, c2, c3 = st.columns(3)
    a = c1.selectbox("Critère en lignes", DIMS, index=0, key="am_a")
    b = c2.selectbox("Critère en colonnes", DIMS, index=2, key="am_b")
    seuil = c3.slider("Effectif minimum par case", min_value=5, max_value=100, value=20, key="am_cell")
    if a == b:
        st.warning("Choisissez deux critères différents.")
    else:
        n = pd.crosstab(d[a], d[b])
        taux = d.pivot_table(index=a, columns=b, values="y", aggfunc="mean", observed=True)
        n, taux = n.reindex(index=taux.index, columns=taux.columns), taux.where(n.reindex(index=taux.index, columns=taux.columns) >= seuil)
        taux.index, taux.columns = taux.index.astype(str), taux.columns.astype(str)
        n.index, n.columns = n.index.astype(str), n.columns.astype(str)
        fig = px.imshow(taux, text_auto=".0%", aspect="auto", color_continuous_scale=[[0, "#EFF6FF"], [1, NUIT]],
                        labels=dict(color="Taux"))
        fig.update_traces(customdata=n.values, hovertemplate="%{y} × %{x}<br>Taux : %{z:.1%}<br>Clients : %{customdata}<extra></extra>")
        fig.update_coloraxes(colorbar_tickformat=".0%")
        st.plotly_chart(style(fig, 440, f"Taux de conversion : {a} × {b}"), width="stretch")
        st.caption("Les cases vides ont un effectif inférieur au seuil choisi (résultat peu fiable).")

    st.divider()
    st.markdown("**Classement des profils (combinaison de deux critères)**")
    c1, c2 = st.columns(2)
    min_n = c1.slider("Effectif minimum du profil", min_value=10, max_value=200, value=30, key="am_min")
    top_n = c2.slider("Nombre de profils affichés", min_value=5, max_value=25, value=10, key="am_top")
    p = profils(d, min_n)
    if p.empty:
        st.warning("Aucun profil ne dépasse cet effectif minimum.")
        return
    top = p.head(top_n)
    st.dataframe(top.rename(columns={"effectif": "Clients", "acceptants": "Acceptants", "lift": "Indice"}).assign(
        Taux=top["taux"] * 100).drop(columns="taux"), hide_index=True, width="stretch", column_config={
        "Taux": st.column_config.ProgressColumn("Taux de conversion", min_value=0.0, max_value=float(top["taux"].max() * 115),
                                                format="%.1f%%"),
        "Indice": st.column_config.NumberColumn("Indice vs moyenne", format="×%.2f")})
    bulles = px.scatter(p.head(40), x="effectif", y="taux", size="acceptants", color="lift", hover_name="Profil",
                        color_continuous_scale=[[0, "#BFDBFE"], [1, NUIT]], size_max=38,
                        labels={"effectif": "Taille du profil (clients)", "taux": "Taux de conversion", "lift": "Indice"})
    bulles.update_yaxes(tickformat=".0%")
    bulles.add_hline(y=tg, line_dash="dash", line_color=ROUGE, annotation_text=f"Moyenne : {pct(tg)}")
    st.plotly_chart(style(bulles, 440, "Rendement vs volume des 40 meilleurs profils"), width="stretch")
    best = top.iloc[0]
    lecture(f"Le profil le plus réceptif est **{best['Profil']}** : **{pct(best['taux'])}** de conversion sur {nombre(best['effectif'])} clients "
            f"({fois(best['lift'])} la moyenne). Les bulles hautes **et** grosses sont les cibles idéales : fort rendement et volume suffisant.")


def onglet_synthese(d: pd.DataFrame) -> None:
    tg = d["y"].mean()
    st.markdown("### Synthèse des enseignements")
    points = []
    for dim, nom in [("Tranche de revenu", "Revenu"), ("Tranche d'âge", "Âge"), ("Éducation", "Éducation"),
                     ("Taille de la famille", "Taille de la famille")]:
        g = agreger(d, dim)
        g = g[g["effectif"] >= 30]
        if len(g) >= 2:
            hi, lo = g.loc[g["taux"].idxmax()], g.loc[g["taux"].idxmin()]
            points.append(f"**{nom}** : meilleure conversion pour *{hi['Modalité']}* ({pct(hi['taux'])}, {fois(hi['lift'])}), "
                          f"la plus faible pour *{lo['Modalité']}* ({pct(lo['taux'])}).")
    rapports = {}
    for dim in PRODUITS:
        g = agreger(d, dim).set_index("Modalité")
        if {"Oui", "Non"} <= set(g.index) and g.loc["Non", "taux"] > 0:
            rapports[dim] = (g.loc["Oui", "taux"], g.loc["Non", "taux"], g.loc["Oui", "taux"] / g.loc["Non", "taux"])
            points.append(f"**{dim}** : {pct(g.loc['Oui', 'taux'])} de conversion pour les détenteurs contre {pct(g.loc['Non', 'taux'])} pour les autres.")
    st.markdown(f"Taux de conversion global : **{pct(tg)}**.")
    for p in points:
        st.markdown("- " + p)

    st.markdown("### Recommandations marketing")
    recos = []
    pr = profils(d, 30)
    if not pr.empty:
        recos.append(f"**Cibler en priorité** le profil *{pr.iloc[0]['Profil']}* ({pct(pr.iloc[0]['taux'])} de conversion).")
    if rapports:
        dim, (a, b, r) = max(rapports.items(), key=lambda kv: kv[1][2])
        if r > 1.05:
            recos.append(f"**Exploiter la vente croisée** : les détenteurs de « {dim} » convertissent {fois(r)} mieux que les autres.")
    _, gain = courbe_gains(d, "Revenu", "revenu")
    if gain is not None:
        recos.append(f"**Réduire les coûts** : en contactant d'abord les 30 % de clients aux revenus les plus élevés, on capte {pct(gain, 0)} des acceptants.")
    recos.append("**Éviter** de cibler massivement les segments dont l'indice est inférieur à 1 : ils consomment du budget pour un faible retour.")
    for r in recos:
        st.markdown("- " + r)
    st.warning("Ces résultats décrivent des **associations observées** dans les données ; ils n'établissent pas de causalité. "
               "Validez les ciblages par un test de campagne sur un échantillon avant un déploiement large.")


# ============================================================
# Point d'entrée
# ============================================================
def afficher(chemin) -> None:
    chemin = Path(chemin)
    st.markdown("## 📊 Analyse marketing des données")
    st.caption("Analyse descriptive du fichier clients : aucun nettoyage ni modélisation, uniquement des agrégations.")
    if not chemin.exists():
        st.error(f"Fichier introuvable : {chemin}")
        return
    try:
        df = charger(str(chemin), chemin.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lecture du fichier impossible : {exc}")
        return
    manquantes = [c for c in REQUISES if c not in df.columns]
    if manquantes:
        st.error(f"Colonnes manquantes : {', '.join(manquantes)}. Colonnes trouvées : {', '.join(map(str, df.columns))}")
        return
    d = preparer(df)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### 🎯 Problématique")
        st.info("Identifier les clients de Thera Bank ayant la plus forte probabilité d'accepter un prêt personnel, "
                "afin de cibler les campagnes marketing et d'améliorer le taux de conversion tout en réduisant les coûts.")
    with c2:
        with st.expander("❓ Questions business traitées"):
            for q in QUESTIONS:
                st.markdown("- " + q)
    with st.expander("📋 Aperçu des données et statistiques descriptives"):
        st.markdown(f"**{nombre(len(df))} clients · {df.shape[1]} variables**")
        st.dataframe(df.head(20), hide_index=True, width="stretch")
        st.dataframe(df.describe().T, width="stretch")

    sections = {
        "📌 Vue d'ensemble": onglet_vue,
        "💶 Revenu & âge": onglet_revenu_age,
        "🎓 Éducation & famille": onglet_education_famille,
        "🏦 Produits bancaires": onglet_produits,
        "🎯 Profils à cibler": onglet_profils,
        "📝 Synthèse": onglet_synthese,
    }
    with st.container(key="nav_analyse"):  # même style que la navigation principale, en plus compact
        choix = st.radio("Sections de l'analyse", list(sections), key="onglet_analyse",
                         horizontal=True, label_visibility="collapsed")
    sections[choix](d)