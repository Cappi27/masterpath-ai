# 3_streamlit_app.py — MULTI-PAGE STREAMLIT DASHBOARD — MasterPath AI

import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
from pathlib import Path
from pyvis.network import Network
import pydeck as pdk
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# CONFIG
# =========================
DATA_DIR = Path("data")

st.set_page_config(
    page_title="MasterPath AI – University Research & Master's Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global title for all pages
st.title("🎓 MasterPath AI")
st.caption(
    "An AI-powered dashboard for exploring research institutions, collaboration networks, "
    "and recommending the best universities for master's students based on research topics."
)

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_all():
    inst_ai = pd.read_csv(DATA_DIR / "institution_ai_summary.csv")
    inst_year = pd.read_csv(DATA_DIR / "institution_year_output.csv")
    edges = pd.read_csv(DATA_DIR / "institution_collaboration_edges.csv")
    metrics = pd.read_csv(DATA_DIR / "ai_model_metrics.csv")
    bert_emb = pd.read_csv(DATA_DIR / "bert_embeddings.csv")
    return inst_ai, inst_year, edges, metrics, bert_emb

inst_ai, inst_year, edges, metrics, bert_emb = load_all()

# =========================
# SIDEBAR — PAGE NAVIGATION
# =========================
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Top Institutions",
        "Collaboration Network",
        "AI Recommender",
        "Global Map",
        "Model Analysis",      # NEW PAGE
    ]
)

# =========================
# SIDEBAR — GLOBAL FILTERS
# =========================
st.sidebar.title("🎛 Dashboard Controls")
clusters = sorted(inst_ai["cluster"].unique().tolist())
selected_cluster = st.sidebar.multiselect(
    "Select Institution Cluster(s)",
    clusters,
    default=clusters
)

filtered_ai = inst_ai[inst_ai["cluster"].isin(selected_cluster)]

# =========================
# PAGE 1 — OVERVIEW (ENHANCED VISUALS + DE-SKEWED)
# =========================
if page == "Overview":
    st.subheader("📊 Overview")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Institutions", inst_ai.shape[0])

    with cols[1]:
        st.metric("Clusters", inst_ai["cluster"].nunique())

    with cols[2]:
        if metrics is not None and len(metrics) > 0 and "clf_accuracy" in metrics.columns:
            st.metric("Model Accuracy", f"{metrics.loc[0, 'clf_accuracy']:.3f}")
        else:
            st.metric("Model Accuracy", "N/A")

    # ---- AI Model Performance Summary ----
    st.markdown("### 🤖 AI Model Performance")

    if metrics is not None and len(metrics) > 0:
        m = metrics.iloc[0]
        cols_m = st.columns(5)
        with cols_m[0]:
            st.metric("Accuracy", f"{m.get('clf_accuracy', np.nan):.3f}")
        with cols_m[1]:
            st.metric("Precision", f"{m.get('clf_precision', np.nan):.3f}")
        with cols_m[2]:
            st.metric("Recall", f"{m.get('clf_recall', np.nan):.3f}")
        with cols_m[3]:
            st.metric("F1-score", f"{m.get('clf_f1', np.nan):.3f}")
        with cols_m[4]:
            st.metric("ROC-AUC", f"{m.get('clf_roc_auc', np.nan):.3f}")

        st.caption("Metrics reported from the BERT + Logistic Regression classifier.")
    else:
        st.warning("AI model metrics not found.")

    st.markdown("---")

    # ---- Top institutions bar chart (STRONGLY DE-SKEWED) ----
    if "weighted_degree" in inst_ai.columns:
        st.markdown("### 🏆 Top Institutions by Collaboration Strength (De-Skewed View)")

        top_n = st.slider(
            "Number of institutions to display",
            5,
            40,
            20,
            key="overview_top_n_institutions",
        )

        scale_mode = st.radio(
            "Scale for collaboration strength",
            ["Percentile Rank"],
            horizontal=True,
            key="overview_scale_mode",
        )

        top_inst = (
            inst_ai.sort_values("weighted_degree", ascending=False)
            .head(top_n)
            .copy()
        )

        y = top_inst["weighted_degree"].astype(float)

        # Percentile Rank
        plot_y = y.rank(pct=True) * 100
        y_label = "Percentile Rank (%)"

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(top_inst["institution"], plot_y)

        ax.set_ylabel(y_label)
        ax.set_xlabel("Institution")
        ax.tick_params(axis='x', rotation=75)

        st.pyplot(fig)
        st.caption("Percentile Rank gives the clearest comparison when data is extremely skewed.")

# =========================
# PAGE 2 — TOP INSTITUTIONS (MORE VISUAL)
# =========================
elif page == "Top Institutions":
    st.subheader("🏆 Top Research Institutions")

    top_k = st.slider("Top K Institutions", 10, 50, 20, 5)

    top_inst = filtered_ai.sort_values("weighted_degree", ascending=False).head(top_k)

    st.dataframe(
        top_inst[["institution", "cluster", "weighted_degree", "degree_centrality", "top_prob"]]
        .rename(columns={"top_prob": "Top Producer Probability"})
    )


# =========================
# PAGE 3 — COLLABORATION NETWORK
# =========================
elif page == "Collaboration Network":
    st.subheader("🤝 Interactive Collaboration Network")

    import tempfile
    import streamlit.components.v1 as components

    st.caption("Click a node to explore all its collaboration relationships.")

    # -------- Institution selector --------
    focus_inst = st.selectbox(
        "Select an institution to explore",
        sorted(inst_ai["institution"].dropna().unique())
    )

    # -------- Filter collaboration edges --------
    sub_edges = edges[(edges["source"] == focus_inst) | (edges["target"] == focus_inst)]

    if sub_edges.empty:
        st.warning("No collaboration data available for this institution.")
        st.stop()

    # -------- Build interactive network --------
    net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white")

    # Add focus node
    net.add_node(
        focus_inst,
        label=focus_inst,
        color="#FF5733",
        size=35
    )

    # Add collaborator nodes & edges
    for _, row in sub_edges.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        w = float(row.get("weight", 1))

        # Safely add nodes without KeyError
        for node in [src, tgt]:
            if node not in net.node_map:
                net.add_node(
                    node,
                    label=node,
                    color="#4A90E2",
                    size=15 + np.log1p(w) * 6
                )

        # Add edge
        net.add_edge(src, tgt, value=w, title=f"Collaboration Weight: {w:.1f}")

    # Enable interaction controls
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])

    # -------- Render to HTML & display --------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
            components.html(html_content, height=620, scrolling=True)

    # -------- Centrality Table --------
    centrality_table = (
        sub_edges
        .groupby(['source', 'target'])['weight']
        .sum()
        .reset_index()
    )

    st.markdown("### 🔗 Direct Collaboration Links")
    st.dataframe(centrality_table, use_container_width=True)

# =========================
# PAGE 4 — AI RECOMMENDER (BERT)
# =========================
elif page == "AI Recommender":
    st.subheader("🎓 Master's Program / Research Match (BERT AI)")

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    @st.cache_resource
    def load_bert_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    user_interest = st.text_input(
        "Describe your research interest (e.g., AI, IoT, Computer Vision, Renewable Energy):"
    )

    if user_interest.strip():
        bert_model = load_bert_model()
        q_vec = bert_model.encode([user_interest])

        X = bert_emb.drop(columns=["institution", "label"]).values
        sims = cosine_similarity(q_vec, X)[0]

        bert_emb_local = bert_emb.copy()
        bert_emb_local["bert_similarity"] = sims

        recs = bert_emb_local.sort_values("bert_similarity", ascending=False).head(10)
        recs = recs.merge(
            inst_ai[["institution", "cluster", "top_prob", "weighted_degree", "country"]],
            on="institution",
            how="left"
        )

        st.metric("🏆 Best Match", recs.iloc[0]["institution"])

        # ---- Visual: Similarity scores bar chart ----
        st.markdown("### 🔍 Similarity Scores for Top Recommendations")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(recs["institution"], recs["bert_similarity"])
        ax.invert_yaxis()
        ax.set_xlabel("BERT Similarity")
        ax.set_ylabel("Institution")
        st.pyplot(fig)

        st.caption("Higher similarity means closer match between your interest and institutional research profile.")

        st.markdown("### 📋 Recommendation Details")
        st.dataframe(
            recs[["institution", "country", "cluster", "weighted_degree", "top_prob", "bert_similarity"]]
            .rename(columns={
                "top_prob": "Top Producer Probability",
                "bert_similarity": "BERT Similarity"
            }),
            use_container_width=True
        )

# =========================
# PAGE 5 — GLOBAL MAP
# =========================
elif page == "Global Map":
    st.subheader("🗺 Global Research Footprint")

    map_df = inst_ai.dropna(subset=["lat", "lon"]).copy()

    if map_df.empty:
        st.error("❌ No valid geographic points available.")
    else:
        map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
        map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")

        np.random.seed(42)
        map_df["lat_j"] = map_df["lat"] + np.random.uniform(-0.2, 0.2, len(map_df))
        map_df["lon_j"] = map_df["lon"] + np.random.uniform(-0.2, 0.2, len(map_df))

        # --------- Graceful bubble overlap handling ---------
        # 1) Use square-root scaling (compresses large hubs)
        # 2) Apply a tight cap on radius
        # 3) Keep small jitter to separate co-located institutions

        wd = map_df["weighted_degree"].fillna(0).astype(float)
        map_df["radius"] = 1200 + np.sqrt(wd) * 120
        map_df["radius"] = map_df["radius"].clip(upper=4200)

        layer = pdk.Layer(
            "ScatterplotLayer",
            map_df,
            get_position='[lon_j, lat_j]',
            get_radius="radius",
            get_fill_color='[255, 80, 80, 180]',
            get_line_color='[255, 255, 255]',
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
            highlight_color='[255, 255, 255, 180]',
        )

        view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.1, pitch=0)

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": """
                <b>{institution}</b><br/>
                Country: {country}<br/>
                Cluster: {cluster}<br/>
                Weighted Degree: {weighted_degree}<br/>
                Top Producer Probability: {top_prob}
                """
            },
            map_style=None,
        )

        st.pydeck_chart(deck, use_container_width=True)
        st.markdown("""
        **Map Interpretation**
        - Each bubble represents a research institution.
        - Bubble size reflects **collaboration strength** (weighted degree in the network).
        - Nearby institutions are separated using slight geographic jitter to avoid exact overlap.
        - Square-root scaling and radius capping prevent large universities from hiding smaller ones.
        - Hover over a bubble to see institutional details.
        """)

# =========================
# PAGE 6 — MODEL ANALYSIS (FULL AI MODEL ANALYSIS)
# =========================
elif page == "Model Analysis":
    st.subheader("📊 Full AI Model Analysis")

    if metrics is None or metrics.empty:
        st.warning("AI model metrics not found.")
    else:
        m = metrics.iloc[0]

        st.markdown("### 🔢 Summary Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Accuracy", f"{m.get('clf_accuracy', np.nan):.3f}")
        with c2:
            st.metric("Precision", f"{m.get('clf_precision', np.nan):.3f}")
        with c3:
            st.metric("Recall", f"{m.get('clf_recall', np.nan):.3f}")
        with c4:
            st.metric("F1-score", f"{m.get('clf_f1', np.nan):.3f}")

        st.markdown("### 📉 Metric Breakdown (Bar Chart)")
        metric_names = []
        metric_values = []
        for col in ["clf_accuracy", "clf_precision", "clf_recall", "clf_f1", "clf_roc_auc"]:
            if col in m and not pd.isna(m[col]):
                metric_names.append(col.replace("clf_", "").upper())
                metric_values.append(float(m[col]))

        if metric_names:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(metric_names, metric_values)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            for i, v in enumerate(metric_values):
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
            st.pyplot(fig)
        else:
            st.info("No valid classification metrics available for plotting.")

        st.markdown("### 🧮 Confusion Matrix (BERT Classifier)")
        cm_path = DATA_DIR / "confusion_matrix_bert.png"
        if cm_path.exists():
            cm_img = Image.open(cm_path)
            st.image(cm_img, caption="Confusion Matrix — BERT + Logistic Regression", use_column_width=True)
        else:
            st.info("`confusion_matrix_bert.png` not found in `data/` folder.")

        st.markdown("### 📑 Raw Metrics Table")
        st.dataframe(metrics, use_container_width=True)