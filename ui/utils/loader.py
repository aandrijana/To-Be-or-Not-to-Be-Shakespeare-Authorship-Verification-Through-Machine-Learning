import streamlit as st
import pickle
import pandas as pd
from datasketch import MinHash, MinHashLSH


@st.cache_resource
def load_and_initialize_pipeline():
    with open("shap_results.pkl", "rb") as f:
        shap_data = pickle.load(f)

    disputed_df = shap_data["disputed_df"]
    per_play_summary = shap_data["per_play_summary"]
    shap_values_disputed = shap_data["shap_values_disputed"]
    feature_names = shap_data["feature_names"]

    # Load all texts obtained after first cleaning
    with open("preprocessed_corpus_final.pkl", "rb") as f:
        corpus_df = pickle.load(f)

    corpus_df['text'] = corpus_df['text'].fillna('').astype(str)
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    text_registry = {}

    for idx, row in corpus_df.iterrows():
        text = row["text"]
        author = row["author"]
        filename = row["filename"]

        words = text.lower().split()
        if len(words) < 3:
            continue
        shingles = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]

        m = MinHash(num_perm=128)
        for s in shingles:
            m.update(s.encode('utf8'))

        key = f"{idx}__{author}__{filename}"
        lsh.insert(key, m)
        text_registry[key] = (text, author, filename)

    return disputed_df, per_play_summary, shap_values_disputed, feature_names, lsh, text_registry