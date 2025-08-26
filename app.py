import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from io import BytesIO

# -------------------------------
# Streamlit App: KARL Lab Correlation Tool (Pro Edition)
# -------------------------------
st.set_page_config(page_title="KARL Correlation Tool", layout="wide")

# Custom CSS for stylish UI
st.markdown("""
    <style>
        .main { background-color: #f9fafc; }
        .stSidebar { background-color: #e6f0ff; }
        h1 { color: #003366; }
        .stTabs [data-baseweb="tab-list"] button { font-size: 16px; font-weight: 600; }
        .credit-box {
            padding: 12px;
            border-radius: 10px;
            background-color: #dce9f9;
            margin-top: 20px;
            font-size: 14px;
        }
        .credit-box h4 { margin-bottom: 8px; color: #002147; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 KARL Lab Correlation & Visualization Tool (Pro Edition)")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    # Read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔎 Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Sidebar Options
    st.sidebar.header("⚙️ Settings")

    # Correlation Method
    method = st.sidebar.radio("Correlation Method", ["pearson", "spearman", "kendall"])

    # Developer & Credits
    st.sidebar.markdown("""
        <div class="credit-box">
            <h4>👨‍💻 Developer Team</h4>
            <b>Developer:</b> Rayhan Miah<br>
            <b>UX Designer:</b> Al Amin<br>
            <b>Testing & QA Engineer:</b><br>
            • Md Nurnabe Sagor<br>
            • Pranto Das<br>
            • Md. Sabbir Ahmed<br>
            • Abu Sadat<br><br>
            <b>Domain Expert:</b> Shahariar Emon<br>
            <b>Co-Supervisor:</b> Md. Asaduzzaman<br>
            <b>Supervisor:</b> Md. Khorshed Alam
        </div>
    """, unsafe_allow_html=True)

    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📌 Correlation Heatmap", "📌 Scatter Plot", "📌 Pair Plot", "📌 Smart Insights"])

    # ---------------------------
    # Tab 1: Correlation Heatmap
    # ---------------------------
    with tab1:
        corr = df.corr(method=method, numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title=f"Correlation Heatmap ({method.title()})"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download correlation matrix
        csv_corr = corr.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download Correlation Matrix (CSV)",
            csv_corr,
            "correlation_matrix.csv",
            "text/csv"
        )

        # TIFF Downloader for Heatmap
        buf = BytesIO()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
        plt.title(f"Correlation Heatmap ({method.title()})")
        plt.tight_layout()
        plt.savefig(buf, format="tiff", dpi=300)
        st.download_button(
            "⬇️ Download Heatmap (TIFF)",
            buf.getvalue(),
            "correlation_heatmap.tiff",
            "image/tiff"
        )

    # ---------------------------
    # Tab 2: Scatter Plot
    # ---------------------------
    with tab2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        x_var = st.selectbox("X-axis", numeric_cols)
        y_var = st.selectbox("Y-axis", numeric_cols)

        reg_option = st.checkbox("Add Regression Line")

        if reg_option:
            fig = px.scatter(df, x=x_var, y=y_var, trendline="ols", title=f"Scatter: {x_var} vs {y_var}")
        else:
            fig = px.scatter(df, x=x_var, y=y_var, title=f"Scatter: {x_var} vs {y_var}")

        st.plotly_chart(fig, use_container_width=True)

        # TIFF Downloader for Scatter
        buf = BytesIO()
        plt.figure(figsize=(6, 5))
        if reg_option:
            sns.regplot(x=df[x_var], y=df[y_var])
        else:
            sns.scatterplot(x=df[x_var], y=df[y_var])
        plt.title(f"Scatter: {x_var} vs {y_var}")
        plt.tight_layout()
        plt.savefig(buf, format="tiff", dpi=300)
        st.download_button(
            "⬇️ Download Scatter (TIFF)",
            buf.getvalue(),
            "scatter_plot.tiff",
            "image/tiff"
        )

    # ---------------------------
    # Tab 3: Pair Plot
    # ---------------------------
    with tab3:
        selected_cols = st.multiselect("Select variables for pair plot", df.select_dtypes(include=np.number).columns.tolist())

        if len(selected_cols) > 1:
            fig = px.scatter_matrix(df[selected_cols], title="Pair Plot (Scatterplot Matrix)")
            st.plotly_chart(fig, use_container_width=True)

            # TIFF Downloader for Pairplot
            buf = BytesIO()
            g = sns.pairplot(df[selected_cols])
            g.savefig(buf, format="tiff", dpi=300)
            st.download_button(
                "⬇️ Download Pair Plot (TIFF)",
                buf.getvalue(),
                "pair_plot.tiff",
                "image/tiff"
            )
        else:
            st.warning("⚠️ Please select at least 2 variables for pair plot.")

    # ---------------------------
    # Tab 4: Smart Insights
    # ---------------------------
    with tab4:
        st.subheader("🤖 Automated Correlation Insights")
        corr_unstacked = corr.unstack().sort_values(ascending=False)

        # Remove self-correlations
        corr_unstacked = corr_unstacked[corr_unstacked < 0.9999]

        # Top 5 strongest positive & negative correlations
        top_pos = corr_unstacked.head(5)
        top_neg = corr_unstacked.tail(5)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔺 Strongest Positive Correlations")
            st.table(top_pos.to_frame("Correlation"))

        with col2:
            st.markdown("### 🔻 Strongest Negative Correlations")
            st.table(top_neg.to_frame("Correlation"))
