import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
)

# ── Load Artifacts ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_model_svm.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_eval():
    try:
        with open("eval_artifacts.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

artifacts    = load_model()
model        = artifacts["model"]
scaler       = artifacts["scaler"]
class_names  = artifacts["class_names"]
feature_cols = artifacts["feature_cols"]
eval_data    = load_eval()

# ── Bean descriptions ──────────────────────────────────────────
bean_info = {
    "BARBUNYA": {"emoji": "🟤", "desc": "Medium-large, kidney-shaped with a speckled pattern. Common in Turkish cuisine."},
    "BOMBAY":   {"emoji": "⚫", "desc": "The largest bean variety. Dark-coloured, smooth and oval-shaped."},
    "CALI":     {"emoji": "🟠", "desc": "Large bean with an oval shape and light colour. Also known as Calı."},
    "DERMASON": {"emoji": "🟡", "desc": "Smallest bean variety. Round shape with a pale yellow colour."},
    "HOROZ":    {"emoji": "🔴", "desc": "Medium-large with a distinct elongated and slightly curved shape."},
    "SEKER":    {"emoji": "⚪", "desc": "Round, smooth and shiny white bean. 'Seker' means sugar in Turkish."},
    "SIRA":     {"emoji": "🟢", "desc": "Medium-sized with an oval shape. Very similar to Dermason but slightly larger."},
}

# ── Header ─────────────────────────────────────────────────────
st.title("🫘 Dry Bean Classifier")
st.markdown(
    "Enter the **physical measurements** of a bean below and the trained "
    "**Support Vector Machine (SVM)** model will predict which variety it belongs to."
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
        **Model:** Support Vector Machine (SVM)  
        **Kernel:** RBF | **C:** 10 | **Gamma:** auto  
        **Test Accuracy:** 92.6%  
        **Macro F1:** 93.7%  

        Trained on the **Dry Beans Dataset**  
        13,611 samples · 7 classes · 16 features

        ---
        **Bean Varieties:**
    """)
    for name, info in bean_info.items():
        st.markdown(f"{info['emoji']} **{name}**")

# ── Input Form ─────────────────────────────────────────────────
st.subheader("📐 Bean Measurements")
st.caption("Adjust the values to match the physical measurements of your bean sample.")

col1, col2, col3 = st.columns(3)

with col1:
    area            = st.number_input("Area (px²)",          min_value=20000,  max_value=260000, value=53000,   step=100)
    perimeter       = st.number_input("Perimeter (px)",       min_value=524.0,  max_value=2000.0, value=855.0,   step=0.1,    format="%.1f")
    majoraxislength = st.number_input("Major Axis Length",    min_value=183.0,  max_value=740.0,  value=320.0,   step=0.1,    format="%.2f")
    minoraxislength = st.number_input("Minor Axis Length",    min_value=122.0,  max_value=461.0,  value=202.0,   step=0.1,    format="%.2f")
    aspectration    = st.number_input("Aspect Ratio",         min_value=1.0,    max_value=2.5,    value=1.58,    step=0.01,   format="%.4f")

with col2:
    eccentricity    = st.number_input("Eccentricity",         min_value=0.21,   max_value=0.92,   value=0.75,    step=0.001,  format="%.6f")
    convexarea      = st.number_input("Convex Area (px²)",    min_value=20000,  max_value=265000, value=53768,   step=100)
    equivdiameter   = st.number_input("Equiv. Diameter",      min_value=161.0,  max_value=570.0,  value=253.0,   step=0.1,    format="%.2f")
    extent          = st.number_input("Extent",               min_value=0.55,   max_value=0.87,   value=0.75,    step=0.001,  format="%.6f")
    solidity        = st.number_input("Solidity",             min_value=0.91,   max_value=1.0,    value=0.987,   step=0.0001, format="%.6f")

with col3:
    roundness       = st.number_input("Roundness",            min_value=0.48,   max_value=1.0,    value=0.873,   step=0.001,  format="%.6f")
    compactness     = st.number_input("Compactness",          min_value=0.64,   max_value=1.0,    value=0.80,    step=0.001,  format="%.6f")
    shapefactor1    = st.number_input("Shape Factor 1",       min_value=0.002,  max_value=0.011,  value=0.00656, step=0.00001,format="%.5f")
    shapefactor2    = st.number_input("Shape Factor 2",       min_value=0.0005, max_value=0.0036, value=0.00172, step=0.00001,format="%.5f")
    shapefactor3    = st.number_input("Shape Factor 3",       min_value=0.4,    max_value=0.98,   value=0.64,    step=0.001,  format="%.6f")
    shapefactor4    = st.number_input("Shape Factor 4",       min_value=0.92,   max_value=1.0,    value=0.996,   step=0.0001, format="%.6f")

st.divider()

# ── Predict ────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Classify Bean", type="primary", use_container_width=True)

if predict_clicked:
    input_data = pd.DataFrame([[
        area, perimeter, majoraxislength, minoraxislength,
        aspectration, eccentricity, convexarea, equivdiameter,
        extent, solidity, roundness, compactness,
        shapefactor1, shapefactor2, shapefactor3, shapefactor4
    ]], columns=feature_cols)

    input_scaled   = scaler.transform(input_data)
    pred_idx       = model.predict(input_scaled)[0]
    pred_class     = class_names[pred_idx]
    info           = bean_info[pred_class]

    decision_scores = model.decision_function(input_scaled)[0]
    exp_scores      = np.exp(decision_scores - decision_scores.max())
    confidences     = exp_scores / exp_scores.sum()

    st.subheader("🎯 Prediction Result")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="Predicted Bean Variety", value=f"{info['emoji']} {pred_class}")
        st.info(info["desc"])

    with res_col2:
        st.markdown("**Confidence Scores across all classes:**")
        conf_df = pd.DataFrame({
            "Bean Class": class_names,
            "Confidence": confidences,
        }).sort_values("Confidence", ascending=False)

        for _, row in conf_df.iterrows():
            bar_color = "🟩" if row["Bean Class"] == pred_class else "⬜"
            st.progress(
                float(row["Confidence"]),
                text=f"{bar_color} {row['Bean Class']}: {row['Confidence']:.2%}"
            )

    st.divider()

# ── Analytics Tabs ─────────────────────────────────────────────
st.subheader("📊 Model Analytics")

if eval_data is None:
    st.warning(
        "⚠️ `eval_artifacts.pkl` not found. "
        "Run the *'Save evaluation artifacts'* cell in your notebook first to enable these charts."
    )
else:
    tab1, tab2, tab3 = st.tabs([
        "🔢 Confusion Matrix",
        "📈 Model Comparison",
        "🥧 Class Distribution",
    ])

    # ── Tab 1: Confusion Matrix ────────────────────────────────
    with tab1:
        st.markdown("#### Confusion Matrix — Best Model (SVM, RBF kernel)")
        st.caption(
            "Rows = True class · Columns = Predicted class · "
            "Diagonal = Correct predictions · Off-diagonal = Mistakes"
        )

        y_test_arr  = np.array(eval_data["y_test"])
        y_pred_arr  = np.array(eval_data["y_pred_best"])
        cm          = confusion_matrix(y_test_arr, y_pred_arr)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix — SVM (RBF)", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Key observations
        st.markdown("**Key Observations:**")
        c1, c2, c3 = st.columns(3)
        c1.success("✅ BOMBAY perfectly classified (100% recall)")
        c2.warning("⚠️ DERMASON ↔ SIRA most confused pair")
        c3.info("ℹ️ SEKER separates well due to high roundness")

    # ── Tab 2: Model Comparison ────────────────────────────────
    with tab2:
        st.markdown("#### All Models — Baseline Performance Comparison")

        comp_df = pd.DataFrame(eval_data["comparison"])
        comp_df = comp_df.sort_values("Test Acc", ascending=False).reset_index(drop=True)
        comp_df["Overfitting"] = comp_df["Overfit Gap"].apply(
            lambda g: "⚠️ Yes" if g > 0.05 else "✅ No"
        )

        # Styled data table
        st.dataframe(
            comp_df.style
                .highlight_max(subset=["Test Acc", "F1 Score"], color="#d4edda")
                .highlight_min(subset=["Overfit Gap"], color="#d4edda")
                .format({"Train Acc": "{:.4f}", "Test Acc": "{:.4f}",
                         "F1 Score": "{:.4f}", "Overfit Gap": "{:.4f}"}),
            use_container_width=True,
        )

        # Bar chart: Train vs Test Acc
        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

        x     = np.arange(len(comp_df))
        width = 0.35
        axes[0].bar(x - width/2, comp_df["Train Acc"], width, label="Train Accuracy", color="steelblue")
        axes[0].bar(x + width/2, comp_df["Test Acc"],  width, label="Test Accuracy",  color="coral")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(comp_df["Model"], rotation=35, ha="right", fontsize=9)
        axes[0].set_ylim(0.70, 1.01)
        axes[0].set_title("Train vs Test Accuracy", fontsize=12)
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # Horizontal F1 ranking
        f1_sorted = comp_df.sort_values("F1 Score", ascending=True)
        best_name = comp_df.loc[comp_df["Test Acc"].idxmax(), "Model"]
        colors    = ["gold" if m == best_name else "steelblue" for m in f1_sorted["Model"]]
        axes[1].barh(f1_sorted["Model"], f1_sorted["F1 Score"], color=colors)
        axes[1].set_xlim(0.70, 1.0)
        axes[1].set_title("F1 Score Ranking (🥇 Best in Gold)", fontsize=12)
        axes[1].set_xlabel("F1 Score (Weighted)")
        axes[1].grid(axis="x", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Tab 3: Class Distribution ──────────────────────────────
    with tab3:
        st.markdown("#### Dataset Class Distribution")
        st.caption("Shows how many samples exist per bean class in the original training dataset.")

        dist = eval_data["class_distribution"]
        dist_df = pd.DataFrame({
            "Bean Class": list(dist.keys()),
            "Count":      list(dist.values()),
        }).sort_values("Count", ascending=False)
        dist_df["Percentage"] = (dist_df["Count"] / dist_df["Count"].sum() * 100).round(2)

        d1, d2 = st.columns([1, 2])

        with d1:
            st.dataframe(dist_df.reset_index(drop=True), use_container_width=True)

        with d2:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            colors_bar = sns.color_palette("Set2", len(dist_df))
            ax3.bar(dist_df["Bean Class"], dist_df["Count"], color=colors_bar)
            ax3.set_title("Class Distribution — Dry Beans Dataset", fontsize=12)
            ax3.set_ylabel("Sample Count")
            ax3.set_xlabel("Bean Class")
            for i, (count, pct) in enumerate(zip(dist_df["Count"], dist_df["Percentage"])):
                ax3.text(i, count + 30, f"{pct}%", ha="center", fontsize=9)
            ax3.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        if dist_df["Count"].max() / dist_df["Count"].min() > 5:
            st.warning(
                f"⚠️ Class imbalance detected: "
                f"**{dist_df.iloc[0]['Bean Class']}** ({dist_df.iloc[0]['Count']:,} samples) vs "
                f"**{dist_df.iloc[-1]['Bean Class']}** ({dist_df.iloc[-1]['Count']:,} samples) "
                f"— a **{dist_df['Count'].max() / dist_df['Count'].min():.1f}x** ratio."
            )
