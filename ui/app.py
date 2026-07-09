import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from collections import Counter
from utils.loader import load_and_initialize_pipeline
from utils.text_utils import get_highlighted_text

st.set_page_config(page_title="Shakespeare Authorship Workbench", layout="wide")
disputed_df, per_play_summary, shap_values_disputed, feature_names, text_registry = load_and_initialize_pipeline()
available_plays = sorted(disputed_df["author"].unique().tolist())

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("Configuration")
selected_play_key = st.sidebar.selectbox("Choose disputed play for analysis", available_plays)
play_rows = disputed_df[disputed_df["author"] == selected_play_key].reset_index(drop=True)

if play_rows.empty:
    st.error("Selected item reference configuration discrepancy.")
    st.stop()

probability = play_rows["shakespeare_probability"].mean()
prediction_label = "Shakespeare" if probability >= 0.50 else "Contemporary"

# ==========================================
# APP TABS LAYOUT
# ==========================================
st.title("🎭 Shakespeare Attribution Dashboard")
st.write("---")

tab1, tab2 = st.tabs(["Attribution Dashboard", "Discovery Engine"])

with tab1:
    st.subheader(f"Analysis Profile: {selected_play_key.upper().replace('_', ' ')}")

    col_meta, col_gauge = st.columns([1, 1])

    with col_meta:
        st.metric("Model Verdict", prediction_label)
        st.write(f"**Mean Aggregate Score Output:** `{probability:.4f}`")
        if prediction_label == "Shakespeare":
            st.success("The structural features assign this document inside the Shakespearean baseline cluster.")
        else:
            st.warning("The token markers indicate structural proximity to baseline contemporary authors.")

    with col_gauge:
        st.markdown(
            f'<div style="text-align: center; margin-bottom: -15px;">'
            f'<span style="font-size: 14px; font-weight: bold; color: #706255; text-transform: uppercase; letter-spacing: 1px;">Current Verdict Score</span><br>'
            f'<span style="font-size: 38px; font-weight: 800; color: #4a2c11; font-family: monospace;">{probability*100:.1f}%</span>'
            f'</div>', 
            unsafe_allow_html=True
        )

        fig_metric = go.Figure()
        
        # Base Axis Line Track
        fig_metric.add_trace(go.Scatter(
            x=[0, 100], y=[0, 0],
            mode="lines",
            line=dict(color="#706255", width=4),
            hoverinfo="skip"
        ))
        
        # Left Terminal Endpoint
        fig_metric.add_trace(go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(size=12, color="#e74c3c"),
            hoverinfo="skip"
        ))
        
        # Right Terminal Endpoint
        fig_metric.add_trace(go.Scatter(
            x=[100], y=[0],
            mode="markers",
            marker=dict(size=12, color="#2ecc71"),
            hoverinfo="skip"
        ))
        
        #Value Tracker Diamond
        fig_metric.add_trace(go.Scatter(
            x=[probability * 100], y=[0],
            mode="markers",
            marker=dict(symbol="diamond", size=20, color="#4a2c11", line=dict(color="#fbf9f4", width=2)),
            hoverinfo="skip"
        ))
        
        fig_metric.update_layout(
            xaxis=dict(
                range=[-5, 105], showgrid=False, zeroline=False,
                tickvals=[0, 50, 100], ticktext=["Contemporary Choice", "Neutral Boundary", "Shakespeare Baseline"],
                tickfont=dict(color="#3b220c", size=11, family="sans-serif")
            ),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=30, r=30, t=15, b=15),
            height=90,
            showlegend=False
        )
        st.plotly_chart(fig_metric, use_container_width=True)

    st.write("---")
    st.subheader("SHAP Feature Importance (Top Stylistic Drivers)")
    play_indices = disputed_df[disputed_df["author"] == selected_play_key].index.tolist()
    slice_shap_values = shap_values_disputed[play_indices]
    mean_shap_array = np.mean(slice_shap_values, axis=0)
    shap_summary_df = pd.DataFrame({"feature": feature_names, "mean_shap": mean_shap_array})
    shap_summary_df = shap_summary_df.groupby("feature", as_index=False)["mean_shap"].mean()
    shap_summary_df["abs_shap"] = shap_summary_df["mean_shap"].abs()
    top_features_df = shap_summary_df.sort_values("abs_shap", ascending=False).head(20)
    top_features_df = top_features_df.sort_values("mean_shap")

    bar_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_features_df["mean_shap"]]

    fig_shap = go.Figure(go.Bar(
        x=top_features_df["mean_shap"],
        y=top_features_df["feature"],
        orientation="h",
        marker=dict(color=bar_colors),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))
    fig_shap.update_layout(
        xaxis_title="Mean SHAP value  (→ Shakespeare  |  ← Contemporary)",
        yaxis_title=None,
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        shapes=[dict(
            type="line", x0=0, x1=0, y0=-0.5, y1=len(top_features_df) - 0.5,
            line=dict(color="gray", width=1, dash="dot")
        )]
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption("🟢 pushes toward Shakespeare  ·  🔴 pushes toward a contemporary author")

    st.write("---")
    st.subheader("Interactive Stylistic Token Viewer")
    st.write("The view below highlights stylistic character patterns or function words that shift predictions toward **Shakespeare** (green) vs **Contemporaries** (red):")

    TEXT_COLUMN_CANDIDATES = ["text", "chunk_text", "raw_text", "content", "passage", "segment", "clean_text"]
    text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in play_rows.columns), None)
    
    sample_chunk_text = ""
    if text_col:
        sample_chunk_text = play_rows.loc[0, text_col]
        
    if not sample_chunk_text:
        # Fallback to text_registry mapping lookup if the direct layout vector is missing
        candidates = [
            v[0] for k, v in text_registry.items()
            if selected_play_key.lower() in v[1].lower() or selected_play_key.lower() in k.lower()
        ]
        if candidates:
            sample_chunk_text = candidates[0]

    highlighted_html = get_highlighted_text(sample_chunk_text, feature_names, mean_shap_array)
    st.markdown(
        f'<div style="background-color:#f9f9f9; padding:20px; border-radius:6px; max-height:400px; '
        f'overflow-y:auto; font-family:monospace; font-size:14px; line-height:1.8; border:1px solid #ddd;">'
        f'{highlighted_html}</div>',
        unsafe_allow_html=True
    )

with tab2:
    st.header("Discovery Engine")
    st.write("### Frequent Stylistic Pattern Profile")
    st.write("Instead of hunting for exact full sentences, we extract the underlying repeated structural and lexical choices characterizing this document's text block:")

    # Tokenize the available clean sequence
    raw_text_block = play_rows.loc[0, "text"] if "text" in play_rows.columns else sample_chunk_text
    words = [w.strip(".,;:?!\"'()[]") for w in raw_text_block.lower().split() if len(w.strip(".,;:?!\"'()[]")) > 1]

    if len(words) > 5:
        # Calculate Frequent Single Words
        top_unigrams = Counter(words).most_common(5)
        
        # Calculate Frequent Pairs (Bigrams)
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        top_bigrams = Counter(bigrams).most_common(5)
        
        # Calculate Frequent Triplets (Trigrams)
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        top_trigrams = Counter(trigrams).most_common(5)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Top Recurrent Words")
            df_uni = pd.DataFrame(top_unigrams, columns=["Token Word", "Frequency Counts"])
            st.dataframe(df_uni, use_container_width=True)
        with c2:
            st.markdown("#### Top Recurrent Pairs")
            df_bi = pd.DataFrame(top_bigrams, columns=["Word Pair", "Frequency Counts"])
            st.dataframe(df_bi, use_container_width=True)
        with c3:
            st.markdown("#### Top Recurrent Triplets")
            df_tri = pd.DataFrame(top_trigrams, columns=["Word Triplet", "Frequency Counts"])
            st.dataframe(df_tri, use_container_width=True)
            
        st.write("---")
        st.markdown("**Raw Segment Reference Snippet:**")
        st.caption(raw_text_block[:1500] + "...")
    else:
        st.info("Insufficient textual context found to parse lexical frequency distributions.")
