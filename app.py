import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

# -------------------------------
# Streamlit App: KARL Lab Correlation Tool (Pro Edition)
# -------------------------------
st.set_page_config(page_title="KARL Correlation Tool", layout="wide")

# ---------- Styling helpers ----------
def apply_style(paper_mode: bool):
    """Apply plotting style safely."""
    if paper_mode:
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            "font.family": "DejaVu Serif",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 1.0
        })
    else:
        sns.set_theme(style="ticks")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "figure.dpi": 120,
            "savefig.dpi": 120
        })

def fig_to_bytes(fmt: str, dpi: int) -> bytes:
    """Save Matplotlib figure to bytes."""
    buf = BytesIO()
    plt.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    data = buf.getvalue()
    plt.close()
    return data

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        .main { background-color: #f9fafc; }
        .stSidebar { background-color: #1c1c1c; color: #ffffff; }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6,
        .stSidebar p, .stSidebar span, .stSidebar label { color: #ffffff !important; }
        h1 { color: #002147; }
        .stTabs [data-baseweb="tab-list"] button { font-size: 16px; font-weight: 600; }
        .credit-box { padding:12px; border-radius:10px; background-color:#333333; margin-top:20px; font-size:14px; color:#ffffff; }
        .credit-box h4 { margin-bottom:8px; color:#e6e6e6; }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("Correlation & Visualization Tool (Pro Edition)")
st.markdown("""
<p style="font-size:18px; color:green; margin-top:-10px;">
    A platform for interactive data correlation, visualization, and scientific insights — powered by <b>KARL</b>.
</p>
""", unsafe_allow_html=True)

# ---------- Sidebar Logo ----------
st.sidebar.markdown("""
<div style="text-align:center; margin-bottom:15px;">
    <img src="https://raw.githubusercontent.com/dev-rayhan-byte/Correlations-APPbyKARL/2a8e89dfe0f61c2eaa204ff02656814506148c98/ddAsset%2011v3.png" width="150">
</div>
""", unsafe_allow_html=True)

# ---------- File uploader ----------
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- Sidebar Controls ----------
    st.sidebar.header("Settings")
    method = st.sidebar.radio("Correlation Method", ["pearson", "spearman", "kendall"])
    paper_mode = st.sidebar.checkbox("Enable Paper-grade Style", value=False)
    apply_style(paper_mode)

    st.sidebar.header("Export")
    export_fmt = st.sidebar.selectbox("Format", ["png", "jpg", "tiff"], index=0)
    export_dpi = st.sidebar.selectbox("DPI", [100, 150, 200, 300, 600], index=3)
    heatmap_decimals = st.sidebar.slider("Heatmap value decimals", 0, 4, 2)

    st.sidebar.markdown("""
    <div class="credit-box">
        <h4>Developer Team</h4>
        <b>Developer:</b> Rayhan Miah<br>
        <b>UX Designer & Benchmarking Lead:</b> Al Amin<br>
        <b>Testing & QA Engineer:</b><br>
        • Md Nurnabe Sagor<br>• Pranto Das<br>• Md. Sabbir Ahmed<br>• Abu Sadat<br><br>
        <b>Domain Expert:</b> Shahariar Emon<br>
        <b>Co-Supervisor:</b> Md. Asaduzzaman<br>
        <b>Supervisor:</b> Md. Khorshed Alam
    </div>
    """, unsafe_allow_html=True)

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs(["Correlation Heatmap", "Scatter Plot", "Pair Plot", "Smart Insights"])

    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr(method=method)

    # --- Tab 1: Heatmap ---
    with tab1:
        st.markdown("### Correlation Heatmaps")

        numeric_df_clean = numeric_df.loc[:, ~numeric_df.columns.str.contains("Unnamed")]
        corr = numeric_df_clean.corr(method=method)

        heatmap_configs = [
            ("Diverging Heatmap", "RdBu_r", -1, 1, "diverging"),
            ("Monotonic Heatmap", "viridis", 0, 1, "monotonic"),
            ("Gradient Heatmap", "plasma", corr.values.min(), corr.values.max(), "gradient"),
        ]

        for title, cmap, vmin, vmax, tag in heatmap_configs:
            st.markdown(f"#### {title} ({method.title()})")

            # Interactive plotly version
            fig_px = px.imshow(
                corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=vmin,
                zmax=vmax,
                color_continuous_scale=cmap,
                text_auto=f".{heatmap_decimals}f",
                aspect="equal"
            )
            fig_px.update_traces(xgap=2, ygap=2)
            fig_px.update_layout(
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(family="DejaVu Serif" if paper_mode else None,
                          size=10 if len(corr.columns) > 8 else 12),
                coloraxis_colorbar=dict(title="Correlation"),
                title=title
            )
            fig_px.update_xaxes(tickangle=45)
            st.plotly_chart(fig_px, use_container_width=True)

            # Matplotlib version (for download)
            fig_dl, ax = plt.subplots(figsize=(len(corr.columns) * 1.2, len(corr.columns) * 1.2))
            im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=vmax)

            # Colorbar close and well-scaled
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("Correlation", fontsize=12, fontweight="bold", family="DejaVu Serif")

            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9, family="DejaVu Serif")
            ax.set_yticklabels(corr.index, fontsize=9, family="DejaVu Serif")

            # Title = Method Correlation
            ax.set_title(f"{method.title()} Correlation", fontsize=16, weight="bold", family="DejaVu Serif")

            # Annotate values
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.{heatmap_decimals}f}",
                            ha="center", va="center", color="black", fontsize=8, family="DejaVu Serif")

            # Add cell borders
            for edge, spine in ax.spines.items():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, len(corr.columns), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(corr.index), 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)

            plt.tight_layout()
            buf = BytesIO()
            fig_dl.savefig(buf, format=export_fmt, dpi=export_dpi,
                           bbox_inches="tight", facecolor="white")
            buf.seek(0)

            st.download_button(
                f"Download ({method.title()} Correlation, {export_fmt.upper()}, {export_dpi} DPI)",
                buf,
                file_name=f"{method.lower()}_correlation_heatmap.{export_fmt}",
                mime={"png": "image/png", "jpg": "image/jpeg", "tiff": "image/tiff"}[export_fmt]
            )
            plt.close(fig_dl)

    # --- Tab 2: Scatter ---
    with tab2:
        numeric_cols = numeric_df.columns.tolist()
        x_var = st.selectbox("X-axis", numeric_cols)
        y_var = st.selectbox("Y-axis", numeric_cols)
        reg_option = st.checkbox("Add Regression Line")

        fig_sc = px.scatter(df, x=x_var, y=y_var,
                            trendline="ols" if reg_option else None,
                            title=f"Scatter: {x_var} vs {y_var}")
        st.plotly_chart(fig_sc, use_container_width=True)

        plt.figure(figsize=(6.5, 5.0))
        if reg_option:
            sns.regplot(x=df[x_var], y=df[y_var], scatter_kws={"s": 25})
        else:
            sns.scatterplot(x=df[x_var], y=df[y_var])
        plt.title(f"Scatter: {x_var} vs {y_var}")
        st.download_button(
            f"Download Scatter ({export_fmt.upper()}, {export_dpi} DPI)",
            fig_to_bytes(export_fmt, export_dpi),
            f"scatter.{export_fmt}",
            mime={"png": "image/png", "jpg": "image/jpeg", "tiff": "image/tiff"}[export_fmt]
        )
        plt.close()

    # --- Tab 3: Pair Plot ---
    with tab3:
        selected_cols = st.multiselect("Select variables for pair plot", numeric_df.columns.tolist())
        if len(selected_cols) > 1:
            fig_px_pair = px.scatter_matrix(df[selected_cols], title="Pair Plot (Scatterplot Matrix)")
            st.plotly_chart(fig_px_pair, use_container_width=True)

            g = sns.pairplot(df[selected_cols], corner=False, plot_kws={"s": 15})
            g.fig.suptitle("Pair Plot (Scatterplot Matrix)", y=1.02)
            st.download_button(
                f"Download Pair Plot ({export_fmt.upper()}, {export_dpi} DPI)",
                fig_to_bytes(export_fmt, export_dpi),
                f"pairplot.{export_fmt}",
                mime={"png": "image/png", "jpg": "image/jpeg", "tiff": "image/tiff"}[export_fmt]
            )
            plt.close(g.fig)
        else:
            st.warning("Please select at least 2 variables for pair plot.")

    # --- Tab 4: Smart Insights ---
    with tab4:
        st.subheader("Automated Correlation Insights")
        corr_unstacked = corr.unstack().sort_values(ascending=False)
        corr_unstacked = corr_unstacked[corr_unstacked < 0.9999]

        top_pos = corr_unstacked.head(5)
        top_neg = corr_unstacked.tail(5)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Strongest Positive Correlations")
            st.table(top_pos.to_frame("Correlation"))
        with col2:
            st.markdown("### Strongest Negative Correlations")
            st.table(top_neg.to_frame("Correlation"))
