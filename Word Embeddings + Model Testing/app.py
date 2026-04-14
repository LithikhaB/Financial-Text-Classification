"""
Reuters-21578 Financial Text Classifier — NLP Viva Demo
Best model: TF-IDF Ngram + LinearSVC (Micro-F1 = 0.9534)

Run: streamlit run app.py
"""

import re
import math
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import fetch_20newsgroups          # fallback demo data
from scipy.sparse import issparse

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Text Classifier",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
COLOURS = {
    "acq":       "#4C72B0",
    "corn":      "#DD8452",
    "crude":     "#55A868",
    "earn":      "#C44E52",
    "grain":     "#8172B3",
    "interest":  "#937860",
    "money-fx":  "#DA8BC3",
    "ship":      "#8C8C8C",
    "trade":     "#CCB974",
    "wheat":     "#64B5CD",
}
LABELS = list(COLOURS.keys())

DEMO_CORPUS = [
    # earn
    ("compani report net earn quarter profit revenue increas strong result quarterly earn annound",
     ["earn"]),
    ("earn per share dividend paid common stockholder quarter net incom rose",
     ["earn"]),
    ("net loss quarter report lower earn per share annual report financi statement",
     ["earn"]),
    ("quarterly earn surpris analyt forecast beat earn per share grew year",
     ["earn"]),
    ("compani announc record earn profit margin expans oper incom increas significantli",
     ["earn"]),

    # acq
    ("mergerz acquis announc takeover bid compani acquir stake sharehold approv",
     ["acq"]),
    ("compani acquir rival firm deal billion shareholder vote merger complet",
     ["acq"]),
    ("takeover offer tender offer premium paid acquir board approv director",
     ["acq"]),
    ("acquis announc strategic partner acquir minoriti interest shareholdr approv",
     ["acq"]),
    ("leverag buyout compani acquir privat equiti firm deal merger complet",
     ["acq"]),

    # crude / oil
    ("crude oil price barrel opec product market suppli demand energi",
     ["crude"]),
    ("oil barrel product quota opec energi market price suppli",
     ["crude"]),
    ("petroleum refin crude barrel suppli energi market price product",
     ["crude"]),
    ("oil field product barrel reserv energi company drill crude market",
     ["crude"]),

    # grain / wheat / corn
    ("grain wheat corn harvest crop yield farm suppli demand export",
     ["grain", "wheat", "corn"]),
    ("wheat price bushel grain export crop season harvest suppli market",
     ["wheat", "grain"]),
    ("corn crop yield farmer plant harvest grain market suppli demand export",
     ["corn", "grain"]),
    ("soybean grain harvest crop field yield export price market suppli",
     ["grain"]),
    ("barley oat grain sorghum harvest crop export suppli market price",
     ["grain"]),

    # trade
    ("trade deficit surplus export import tariff barrier trade balanc nation",
     ["trade"]),
    ("export import trade balanc deficit nation tariff quotas commerci",
     ["trade"]),
    ("trade negoti agreement import export deficit surplus tariff trade",
     ["trade"]),

    # money-fx
    ("dollar yen mark sterling foreign exchang rate currenc market monetari",
     ["money-fx"]),
    ("currenc exchang rate foreign market dollar euro yen monetari polic",
     ["money-fx"]),
    ("forex market dollar yen currenc rate central bank monetari interventionist",
     ["money-fx"]),

    # interest
    ("interest rate feder reserv monetari polic credit bank loan borrow",
     ["interest"]),
    ("interest rate central bank monetari polic credit market treasuri bond",
     ["interest"]),
    ("bond yield interest rate monetari polic feder reserv bank credit",
     ["interest"]),

    # ship
    ("ship tanker vessel port cargo freight ocean transport bulk carrier",
     ["ship"]),
    ("vessel port cargo freight ship ocean tanker transport bulk carrier",
     ["ship"]),

    # compound / multi-label
    ("grain export trade deficit suppli market price",
     ["grain", "trade"]),
    ("earn per share acquis complet quarter profit increas",
     ["earn", "acq"]),
    ("crude oil price barrel trade deficit import market suppli",
     ["crude", "trade"]),
    ("wheat corn grain harvest export market price suppli demand",
     ["wheat", "corn", "grain"]),
    ("interest rate monetari currenci exchang dollar rate market",
     ["interest", "money-fx"]),
]

@st.cache_resource(show_spinner="Training demo model…")
def train_model():
    texts  = [d[0] for d in DEMO_CORPUS]
    labels = [d[1] for d in DEMO_CORPUS]

    mlb = MultiLabelBinarizer(classes=LABELS)
    Y   = mlb.fit_transform(labels)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.99,
        max_features=3000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    clf = OneVsRestClassifier(
        LinearSVC(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
    )
    clf.fit(X, Y)
    return vectorizer, mlb, clf


vectorizer, mlb, model = train_model()

def get_prediction_trace(text: str):
    """Return rich trace dict for visualising the whole pipeline."""

    # Step 1 — tokenise
    raw_tokens  = text.lower().split()
    vocab       = vectorizer.vocabulary_
    stop_words  = getattr(vectorizer, "stop_words_", set())

    # Step 2 — TF-IDF
    X_sparse    = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    # Top-15 features by TF-IDF weight
    row = X_sparse.toarray()[0]
    top_idx = np.argsort(row)[::-1][:15]
    tfidf_top = [(feature_names[i], round(float(row[i]), 4)) for i in top_idx if row[i] > 0]

    # Step 3 — OvR decisions
    decisions = {}
    for est, label in zip(model.estimators_, mlb.classes_):
        raw_dec = float(est.decision_function(X_sparse)[0])
        decisions[label] = raw_dec

    # Step 4 — binary predictions
    y_pred = model.predict(X_sparse)[0]
    predicted_labels = [mlb.classes_[i] for i, v in enumerate(y_pred) if v == 1]

    # TF-IDF math breakdown for top-3 features
    idf_map = dict(zip(feature_names, vectorizer.idf_))
    tfidf_math = []
    for feat, weight in tfidf_top[:5]:
        # Re-derive TF
        count = text.lower().count(feat)  # approximate
        tf_raw  = count if count else 0
        tf_sub  = (1 + math.log(tf_raw)) if tf_raw > 0 else 0
        idf_val = idf_map.get(feat, 0.0)
        tfidf_math.append({
            "feature": feat,
            "raw_count": tf_raw,
            "tf_sublinear": round(tf_sub, 4),
            "idf": round(idf_val, 4),
            "tfidf": round(weight, 4),
        })

    return {
        "raw_tokens":      raw_tokens,
        "vocab_size":      len(vocab),
        "feature_dim":     X_sparse.shape[1],
        "nnz":             X_sparse.nnz,
        "tfidf_top":       tfidf_top,
        "tfidf_math":      tfidf_math,
        "decisions":       decisions,
        "predicted_labels":predicted_labels,
    }


st.title("📰 Financial Text Classifier — NLP Viva Demo")
st.markdown(
    """
    **Pipeline:** Raw text → TF-IDF Ngrams (1,2) → LinearSVC (OneVsRest) → Multi-label prediction  
    **Best model:** TF-IDF-Ngram + SVM · Micro-F1 = **0.9534** · Macro-F1 = **0.9212**  
    *Dataset: Reuters-21578 (R10 — 10-class subset)*
    """
)

st.divider()

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Labels covered:**
        - earn · acq
        - grain · wheat · corn
        - crude
        - money-fx · interest
        - ship
        - trade
        
        """
    )
    st.divider()
    st.header("Quick Inputs")
    quick = st.radio(
        "Select an example:",
        ["Custom", "Earnings Report", "M&A Deal", "Oil Market", "Grain Export", "Currency Rate"],
    )

QUICK_TEXTS = {
    "Earnings Report":
        "The company reported strong quarterly earnings, with net income rising 18% year over year. "
        "Earnings per share beat analyst forecasts as operating margins expanded significantly.",
    "M&A Deal":
        "Acme Corp announced the acquisition of rival firm XYZ Ltd in a $2.4 billion deal. "
        "Shareholders of both companies will vote on the merger at an extraordinary general meeting.",
    "Oil Market":
        "Crude oil prices fell sharply as OPEC production quotas were raised. "
        "The price per barrel dropped below $70 amid concerns about global energy demand.",
    "Grain Export":
        "Wheat and corn exports surged following a bumper harvest season. "
        "Grain traders expect strong demand from Asian markets as crop yields exceed forecasts.",
    "Currency Rate":
        "The US dollar weakened against the Japanese yen and German mark on forex markets. "
        "Currency traders expect further monetary policy easing by the Federal Reserve.",
}


col_input, col_note = st.columns([3, 1])
with col_input:
    if quick != "Custom":
        default_text = QUICK_TEXTS[quick]
    else:
        default_text = (
            "The company reported strong quarterly earnings while announcing the acquisition "
            "of a rival grain trader. Crude oil exports also fell due to trade barriers."
        )

    user_text = st.text_area(
        "Enter financial news text to classify:",
        value=default_text,
        height=130,
        key="user_text",
    )

with col_note:
    st.markdown("##### Tips")
    st.markdown(
        "- Use financial vocabulary\n"
        "- Multiple labels can fire simultaneously\n"
        "- The model was trained on a **demo corpus**; "
        "results may differ from the Kaggle notebook"
    )

predict_btn = st.button("🔍 Classify & Explain", type="primary", use_container_width=True)

# ── Run prediction ─────────────────────────────────────────────────────────────
if predict_btn and user_text.strip():
    trace = get_prediction_trace(user_text)

    st.divider()

    # ── STEP 0 — Input stats ──────────────────────────────────────────────────
    st.subheader("Step 0 — Raw Input")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tokens",    len(trace["raw_tokens"]))
    c2.metric("Unique tokens",   len(set(trace["raw_tokens"])))
    c3.metric("TF-IDF features", trace["feature_dim"])
    c4.metric("Non-zero feats",  trace["nnz"])

    with st.expander("📖 Token view"):
        st.write(" · ".join(trace["raw_tokens"]))

    st.divider()

    # ── STEP 1 — TF-IDF vectorisation ────────────────────────────────────────
    st.subheader("Step 1 — TF-IDF Vectorisation")
    st.markdown(
        r"""
        Each token (and bigram) is weighted using:

        $$\text{TF-IDF}(t, d) = \underbrace{(1 + \log \text{TF}(t,d))}_{\text{sublinear TF}} \times \underbrace{\log\!\left(\frac{1+N}{1+\text{df}(t)}\right)+1}_{\text{IDF}}$$

        The resulting sparse vector is ℓ₂-normalised.
        """
    )

    col_tbl, col_bar = st.columns([1, 1])

    with col_tbl:
        st.markdown("**TF-IDF Breakdown (top 5 features)**")
        df_math = pd.DataFrame(trace["tfidf_math"])
        if not df_math.empty:
            st.dataframe(df_math, use_container_width=True, hide_index=True)

    with col_bar:
        if trace["tfidf_top"]:
            feats, weights = zip(*trace["tfidf_top"][:10])
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.barh(list(feats)[::-1], list(weights)[::-1], color="#2E75B6", alpha=0.85)
            ax.set_xlabel("TF-IDF weight")
            ax.set_title("Top features in this document", fontsize=11)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # ── STEP 2 — OvR Decision functions ──────────────────────────────────────
    st.subheader("Step 2 — OneVsRest Decision Functions")
    st.markdown(
        r"""
        For each label $k$, a LinearSVC computes a signed distance to the separating hyperplane:

        $$d_k = \mathbf{w}_k^\top \mathbf{x} + b_k$$

        - $d_k > 0$ → label $k$ is **predicted positive**  
        - The decision boundary is at $d_k = 0$
        """
    )

    decisions = trace["decisions"]
    labels_sorted = sorted(decisions, key=lambda l: decisions[l], reverse=True)
    vals = [decisions[l] for l in labels_sorted]
    colors_bar = [
        COLOURS.get(l, "#AAAAAA") if decisions[l] > 0 else "#DDDDDD"
        for l in labels_sorted
    ]

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    bars2 = ax2.barh(labels_sorted[::-1], vals[::-1], color=colors_bar[::-1], edgecolor="grey", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Decision boundary")
    ax2.set_xlabel("Decision function value (dₖ)")
    ax2.set_title("OvR Decision functions per label", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    for bar, val in zip(bars2, vals[::-1]):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # ── STEP 3 — Final prediction ─────────────────────────────────────────────
    st.subheader("Step 3 — Predicted Labels")

    if trace["predicted_labels"]:
        col_pred, col_gauge = st.columns([2, 1])

        with col_pred:
            for lbl in trace["predicted_labels"]:
                dec_val = decisions[lbl]
                col_hex = COLOURS.get(lbl, "#AAAAAA")
                st.markdown(
                    f'<span style="background:{col_hex};color:white;padding:6px 14px;'
                    f'border-radius:20px;font-weight:bold;font-size:1.1em;margin:4px;">'
                    f'{lbl.upper()}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("")

        with col_gauge:
            # Confidence gauge (max decision value)
            max_d = max(abs(v) for v in decisions.values()) or 1
            pos_vals = {l: v for l, v in decisions.items() if v > 0}
            if pos_vals:
                best_label = max(pos_vals, key=pos_vals.get)
                confidence = min(pos_vals[best_label] / max_d, 1.0)
                st.metric("Top label", best_label.upper())
                st.progress(float(confidence), text=f"Confidence proxy: {confidence:.0%}")

    else:
        st.warning("No label fires positively. The text may be outside the training distribution.")

    st.divider()

    # ── STEP 4 — Label probability radar ─────────────────────────────────────
    st.subheader("Step 4 — Decision Landscape (Radar Chart)")

    dec_vals = np.array([decisions[l] for l in LABELS])
    # Normalise to [0,1] for display
    dec_norm = (dec_vals - dec_vals.min()) / (dec_vals.max() - dec_vals.min() + 1e-9)

    N = len(LABELS)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    dec_norm_plot = list(dec_norm) + [dec_norm[0]]

    fig3, ax3 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax3.plot(angles, dec_norm_plot, "o-", linewidth=2, color="#2E75B6")
    ax3.fill(angles, dec_norm_plot, alpha=0.25, color="#2E75B6")
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(LABELS, size=9)
    ax3.set_yticklabels([])
    ax3.axhline(
        y=(0 - dec_vals.min()) / (dec_vals.max() - dec_vals.min() + 1e-9),
        color="red", linewidth=1, linestyle="--", alpha=0.6
    )
    ax3.set_title("Normalised decision values\n(dashed = boundary)", size=11, pad=15)
    plt.tight_layout()

    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        st.pyplot(fig3)
        plt.close(fig3)
    with col_r2:
        st.markdown("**Decision values table**")
        df_dec = pd.DataFrame(
            [{"Label": l, "Decision (dₖ)": round(decisions[l], 4),
              "Predicted": "✅" if decisions[l] > 0 else "❌"}
             for l in LABELS]
        ).sort_values("Decision (dₖ)", ascending=False)
        st.dataframe(df_dec, use_container_width=True, hide_index=True)

    st.divider()

    # ── MODEL PERFORMANCE SUMMARY ─────────────────────────────────────────────
    st.subheader("Model Performance (Full Reuters-21578 Test Set)")

    perf_data = {
        "Embedding":  ["TF-IDF-Ngram", "TF-IDF", "BoW", "Word2Vec-SG", "FastText"],
        "Model":      ["SVM", "SVM", "Logistic Reg", "XGBoost", "XGBoost"],
        "Micro-F1":   [0.9534, 0.9528, 0.9465, 0.9340, 0.9302],
        "Macro-F1":   [0.9212, 0.9244, 0.9139, 0.8673, 0.8570],
    }
    df_perf = pd.DataFrame(perf_data)

    fig4, ax4 = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(df_perf))
    w = 0.35
    b1 = ax4.bar(x - w / 2, df_perf["Micro-F1"], w, label="Micro-F1", color="#2E75B6", alpha=0.85)
    b2 = ax4.bar(x + w / 2, df_perf["Macro-F1"], w, label="Macro-F1", color="#C44E52", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{e}\n({m})" for e, m in zip(df_perf["Embedding"], df_perf["Model"])],
                        fontsize=8)
    ax4.set_ylabel("F1 Score")
    ax4.set_ylim(0.85, 0.97)
    ax4.legend()
    ax4.set_title("Top-5 model combinations on Reuters-21578 test set", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # Highlight best bar
    b1[0].set_edgecolor("gold")
    b1[0].set_linewidth(2.5)

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # ── Mathematical summary ──────────────────────────────────────────────────
    with st.expander("Mathematical Summary of This Prediction"):
        st.markdown(
            f"""
            **Input document** → {len(trace['raw_tokens'])} tokens

            **TF-IDF transformation:**
            - Vocabulary: {trace['vocab_size']:,} terms
            - Feature dimension: {trace['feature_dim']:,}
            - Non-zero features: {trace['nnz']}
            - Sparsity: {1 - trace['nnz']/trace['feature_dim']:.1%}

            **OneVsRest decision:**  
            For each label $k$: $d_k = \\mathbf{{w}}_k^\\top \\mathbf{{x}} + b_k$  
            Label fires if $d_k > 0$

            | Label | dₖ | Predict |
            |---|---|---|
            {chr(10).join(f"| **{l}** | {decisions[l]:.4f} | {'✅' if decisions[l]>0 else '❌'} |" for l in LABELS)}

            **Result:** `{trace['predicted_labels'] if trace['predicted_labels'] else ['(none)']}`
            """
        )

elif predict_btn:
    st.warning("Please enter some text above.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Reuters-21578 NLP Pipeline · Best model: TF-IDF-Ngram + LinearSVC · "
    "Micro-F1 0.9534 on full test set · "
    )
