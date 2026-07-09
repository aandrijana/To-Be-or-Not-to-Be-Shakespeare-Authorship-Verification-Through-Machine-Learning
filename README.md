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

1. Shapiro, J. Contested Will: Who Wrote Shakespeare? (Simon and Schuster, New York, 2010).
2. Taylor, G. & Egan, G. The New Oxford Shakespeare: Authorship Companion (Oxford University Press, Oxford,
2017).
3. Stamatatos, E. A survey of modern authorship attribution methods. J. Am. Soc. for Inf. Sci. Technol. 60,
538–556 (2009).
7/8
4. Mendenhall, T. C. The characteristic curves of composition. Science (1887).
5. Burrows, J. Word-patterns and story-shapes: The statistical analysis of narrative style. Lit. Linguist. Comput.
(1987).
6. Savoy, J. Machine Learning Methods for Stylometry: Authorship Attribution and Author Profiling (Springer,
Cham, Switzerland, 2020).
7. Marsden, J., Budden, D., Craig, H. & Moscato, P. Language individuation and marker words: Shakespeare and
his maxwell’s demon. PLOS ONE (2013).
8. Arefin, A. S., Vimieiro, R., Riveros, C., Craig, H. & Moscato, P. An information theoretic clustering approach
for unveiling authorship affinities in shakespearean era plays and poems. PLOS ONE (2014).
9. Boyd, R. L. & Pennebaker, J. W. Did shakespeare write double falsehood? identifying individuals by creating
psychological signatures with text analysis. Psychol. Sci. (2015).
10. Kernot, D., Bossomaier, T. & Bradbury, R. Novel psychological text feature rpas and authorship analysis of
shakespeare, marlowe, and cary. (2018).
11. Segarra, S., Eisen, M., Egan, G. & Ribeiro, A. Attributing the authorship of the henry vi plays by word
adjacency. Shakespear. Q. (2016).
12. Segarra, S., Eisen, M., Egan, G. & Ribeiro, A. Authorship attribution through function word adjacency networks.
(2018).
13. Plecháč, P. Relative contributions of shakespeare and fletcher in henry viii: An analysis based on most frequent
words and most frequent rhythmic patterns. Digit. Scholarsh. Humanit. (2019).
14. Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. In Advances in Neural
Information Processing Systems (NeurIPS) (2017).
15. Burrows, J. All the way through: Testing for authorship in different frequency strata. Lit. Linguist. Comput.
(2007)
