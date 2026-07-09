import os
import streamlit as st
import pickle
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ui/utils
DATA_DIR = os.path.dirname(BASE_DIR)                      # .../ui

@st.cache_resource
def load_and_initialize_pipeline():
    shap_path = os.path.join(DATA_DIR, "shap_results.pkl")
    corpus_path = os.path.join(DATA_DIR, "preprocessed_corpus_final.pkl")

    with open(shap_path, "rb") as f:
        shap_data = pickle.load(f)

    disputed_df = shap_data["disputed_df"]
    per_play_summary = shap_data["per_play_summary"]
    shap_values_disputed = shap_data["shap_values_disputed"]
    feature_names = shap_data["feature_names"]

    with open(corpus_path, "rb") as f:
        corpus_df = pickle.load(f)

    corpus_df['text'] = corpus_df['text'].fillna('').astype(str)
    text_registry = {}

    for idx, row in corpus_df.iterrows():
        text = row["text"]
        author = row["author"]
        filename = row["filename"]
        key = f"{idx}__{author}__{filename}"
        text_registry[key] = (text, author, filename)

    return disputed_df, per_play_summary, shap_values_disputed, feature_names, text_registry
