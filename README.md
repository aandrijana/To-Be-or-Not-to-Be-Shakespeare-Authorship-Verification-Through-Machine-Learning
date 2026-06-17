# To Be or Not to Be Shakespeare
### Authorship Verification Through Machine Learning

> A computational framework for Shakespearean authorship attribution using stylometric features, ensemble classifiers, and SHAP-based explainability — applied to several genuinely disputed Early Modern plays.

---

## What This Project Does

This project builds an authorship verification system for Early Modern English literature. Given a text, the system estimates the probability that Shakespeare wrote it. For disputed plays, rather than producing a single yes/no verdict, it slides a classification window across the text and produces act-by-act and scene-by-scene probability scores — making it possible to detect where authorship shifts within a single play.

The system is then applied to three genuinely disputed works:
- **Henry VI Parts 1, 2, and 3** — widely believed to involve multiple authors including Marlowe, Greene, and Peele
- **The Spanish Tragedy** — a Kyd play with disputed later additions sometimes attributed to Shakespeare
- **Double Falsehood** — Theobald's 1728 adaptation of a claimed lost Shakespeare/Fletcher manuscript

---

## Corpus

### Shakespeare (positive class)
| File | Source |
|------|--------|
| `shakespeare_complete.txt` | Project Gutenberg (Complete Moby Shakespeare) |

### Contemporaries (negative class)
| Author | Works | Why Included |
|--------|-------|--------------|
| Marlowe | Faustus, Tamburlaine I & II, Jew of Malta, Edward II | Closest stylistic rival; candidate author for Henry VI |
| Jonson | Volpone, The Alchemist, Epicoene | Stylistically distinct anchor for the non-Shakespeare class |
| Fletcher | The Faithful Shepherdess, The Humorous Lieutenant | Confirmed co-author of late Shakespeare plays; key reference for Double Falsehood |
| Kyd | Cornelia | Baseline for Spanish Tragedy attribution |
| Greene | James IV | Candidate author for Henry VI Part 1 |
| Peele | The Old Wives' Tale | Candidate author for Henry VI Part 1 |
| Middleton | The Changeling | Broadens negative class beyond Shakespeare's immediate rivals |

### Disputed Texts
| File | Notes |
|------|-------|
| `henry_vi_part1/2/3.txt` | Collaborative authorship widely accepted; specific contributions debated |
| `spanish_tragedy.txt` | Kyd attribution settled; later additions disputed |
| `double_falsehood.txt` | Sourced from Greenblatt's Cardenio Project (greenblattrenaissance.com) |

---

## Methodology

### Stage 1 — Feature Extraction
Each text is converted into a numeric feature vector capturing:
- **Lexical features** — word frequency distributions, vocabulary richness, n-gram profiles
- **Structural features** — sentence length, punctuation patterns, function word usage
- **Psycholinguistic features** — Referential Activity Power (Bucci & Freedman, 1981), sensory adjective profiles, personal pronoun gender score (RPAS framework, Kernot et al. 2018)
- **Rhythmic/metrical features** — stress patterns extracted line by line using the `Prosodic` library
- **Similarity features** — MinHash/LSH-based phrase overlap scores between disputed and known-author texts

PCA is applied for dimensionality reduction and visualization before classification.

### Stage 2 — Classification
Multiple classifiers are trained on labeled Shakespeare vs. non-Shakespeare texts and evaluated with cross-validation:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- (Potentially) a simple Neural Network

### Stage 3 — Explainability
SHAP values are computed on the best-performing model(s) to identify which stylometric features most strongly indicate Shakespearean authorship — providing a data-driven answer to "what makes Shakespeare sound like Shakespeare."

### Stage 4 — Disputed Text Attribution
A sliding window (e.g. 100 lines) is passed across each disputed text. Each window is classified independently, producing a probability time-series across the play. This makes it possible to identify the specific scenes or acts where authorship is most and least likely to be Shakespeare.

---

## Key Outputs

- A cleaned and preprocessed literary corpus of 20 Early Modern texts
- Comparative classifier evaluation with cross-validation results
- SHAP feature importance analysis revealing Shakespeare's stylometric signature
- Act- and scene-level authorship probability scores for Henry VI, The Spanish Tragedy, and Double Falsehood

---

## Project Structure

```
shakespeare_project/
│
├── texts/                        # Raw downloaded texts
│   ├── shakespeare_complete.txt
│   ├── marlowe_faustus.txt
│   └── ...
│
├── 01_data_collection.ipynb      # Corpus acquisition (Gutenberg, Greenblatt)
├── 02_preprocessing.ipynb        # Cleaning, splitting, normalization
├── 03_feature_extraction.ipynb   # Stylometric feature engineering
├── 04_classification.ipynb       # Model training and cross-validation
├── 05_explainability.ipynb       # SHAP analysis
├── 06_disputed_texts.ipynb       # Sliding window attribution
│
└── README.md
```

---

## Team

| Member | Responsibilities |
|--------|-----------------|
| **Anđela Maksimović** | Corpus acquisition, preprocessing, lexical feature extraction, MinHash/LSH similarity modeling, distance-based classifiers |
| **Andrijana Kešelj** | Linguistic annotation pipeline, structural and rhythmic features, PCA, ensemble modeling, SHAP explainability |

---

## References

Key prior work this project builds on:

- Marsden et al. (2013) — marker word approach, 90%+ accuracy on 168 plays
- Arefin et al. (2014) — Jensen-Shannon divergence clustering of 256 Elizabethan texts
- Kernot, Bossomaier & Bradbury (2018) — RPAS psycholinguistic framework
- Boyd & Pennebaker (2015) — psychological scoring applied to Double Falsehood
- Segarra et al. (2016) & Eisen et al. (2018) — Word Adjacency Networks
- Plecháč (2019) — metrical features and sliding-window attribution for Henry VIII
- Skurla et al. (2026) — SHAP applied to stylometric classification
- Bucci & Freedman (1981) — Referential Activity and the language of depression
