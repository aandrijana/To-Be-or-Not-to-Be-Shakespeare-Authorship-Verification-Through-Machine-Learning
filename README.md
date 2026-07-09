# To Be or Not to Be Shakespeare
### Authorship Verification Through Machine Learning

> A computational stylometry pipeline for Shakespearean authorship attribution, using function-word frequencies, character n-grams, Burrows' Zeta marker words, and SHAP-based explainability, applied to five genuinely disputed Early Modern texts.

---
Collaborators: Andrijana Kešelj, Anđela Maksimović

This project is done by the students of DSAI at ETF Sarajevo.
---

## What This Project Does

This project builds an authorship verification pipeline for Early Modern English drama.
Texts are split into fixed 1,000-word chunks, each converted into a stylometric feature
vector, and classified as Shakespearean or not. For each disputed play, every chunk gets its
own probability; these are aggregated to give an overall verdict for the play along with a
sense of how much that signal varies across the text.

The trained model is applied to five disputed texts:
- **Henry VI Parts 1, 2, and 3** — a collaborative history widely believed to involve other
  hands; a dedicated Shakespeare-vs-Marlowe classifier is used here, since Marlowe's
  involvement (particularly in Part I) is the specific, most-discussed hypothesis in the
  literature
- **The Spanish Tragedy** — established as Thomas Kyd's, included as a check on the pipeline
- **Double Falsehood** — Theobald's 1728 adaptation, traditionally linked to the lost
  *Cardenio* and a claimed Shakespeare/Fletcher collaboration; a dedicated
  Shakespeare-vs-Fletcher classifier is used here

*(Titus Andronicus was also collected as a candidate disputed text — it's traditionally
attributed to a Shakespeare-Peele collaboration — but was excluded from the final pipeline:
there wasn't enough surviving Peele text to support a reliable comparison.)*

---

## Corpus

### Shakespeare (positive class) — 33 works
Sourced from Project Gutenberg's Complete Works. Poems (*the Sonnets*, *Venus and Adonis*,
*The Rape of Lucrece*, *The Passionate Pilgrim*, *A Lover's Complaint*) and known
collaborations (*The Two Noble Kinsmen*, *Titus Andronicus*) were excluded from the positive
class, since the goal is a clean, single-author signal.

### Contemporaries (negative class) — 23 works across 7 authors
| Author | Works | Why Included |
|---|---|---|
| Marlowe | Doctor Faustus, Tamburlaine I & II, The Jew of Malta, Edward II, The Massacre at Paris | Closest stylistic rival; candidate co-author for Henry VI |
| Jonson | The Alchemist, Bartholomew Fair, Every Man in His Humour, Volpone | Stylistically distinct anchor for the non-Shakespeare class |
| Fletcher | Beggar's Bush, The Chances, The Tragedy of Valentinian, The Wild Goose Chase | Confirmed co-author of late Shakespeare plays; key reference for Double Falsehood |
| Massinger | The Great Duke of Florence, The Bondman, The Maid of Honour | Broadens the negative class beyond Shakespeare's closest rivals |
| Greene | Orlando Furioso, Friar Bacon and Friar Bungay, James IV | Broadens the negative class; occasionally raised in the wider literature as a Henry VI candidate |
| Kyd | Cornelia, The Tragedy of Soliman and Perseda | Author of *The Spanish Tragedy*; baseline for that comparison |
| Nashe | Summer's Last Will and Testament | Candidate co-author for Henry VI; prose stylist, broadens the class further |

### Disputed Texts — 5 texts (94,001 words), held out entirely from training
| File | Notes |
|---|---|
| `henry_vi_part1/2/3.txt` | Collaborative authorship widely accepted; Marlowe's specific involvement debated, especially in Part I |
| `spanish_tragedy.txt` | Kyd attribution settled; included as a pipeline sanity check |
| `double_falsehood.txt` | Sourced from the Greenblatt Cardenio Project (greenblattrenaissance.com) |

---

## Methodology

### Stage 1 — Data Collection & Cleaning
- Texts sourced from Project Gutenberg (and Double Falsehood from the Greenblatt Cardenio
  Project).
- Non-authorial text (editorial intros, character lists, licensing headers) was removed;
  Gutenberg headers/footers were stripped automatically, everything else by hand, since the
  corpus was small enough to check file-by-file.
- Files containing multiple plays were split into individual play files.
- Named entities (character/place names) were masked with a generic placeholder token, so
  the model can't learn to associate specific names with authorship rather than style.
- An initial near-perfect Fletcher-vs-Shakespeare split was traced to formatting differences
  (dash characters, spacing before punctuation) between source editions rather than real
  style, and was corrected before finalizing features.

### Stage 2 — Feature Extraction
Each 1,000-word chunk is represented as a **2,182-dimensional** feature vector:
- **Function-word frequencies** — 122 function words (articles, prepositions, pronouns,
  auxiliaries, conjunctions), length-normalized
- **Character n-grams** — the 2,000 most frequent 3–4 character n-grams (`char_wb`
  tokenization), L1-normalized per chunk
- **Burrows' Zeta marker words** — 60 words (30 per class), selected by how consistently
  they appear across chunks rather than by raw frequency

The train/test split is done **by whole play**, not by chunk, so no play's vocabulary leaks
between training and evaluation; the same grouping is used for cross-validation.

### Stage 3 — Classification
Four classifiers were trained and compared via 5-fold grouped cross-validation (macro F1,
chosen over accuracy due to class imbalance):

| Model | CV Macro F1 |
|---|---|
| **Logistic Regression** (selected) | **0.906 ± 0.032** |
| SVM (linear kernel) | 0.901 ± 0.034 |
| Gradient Boosting | 0.852 ± 0.036 |
| Random Forest | 0.679 ± 0.146 |

A multiclass (8-author) model was also tested to get a full author-probability distribution
directly, but performed poorly and wasn't used further.

Two **dedicated pairwise classifiers** (Logistic Regression) were also trained —
Shakespeare vs. Marlowe and Shakespeare vs. Fletcher — to compare specific disputed texts
against one candidate collaborator at a time, rather than "not Shakespeare" in general.

### Stage 4 — Explainability
SHAP values (main model) and raw logistic-regression coefficients (pairwise models) are used
to identify which features drive predictions on the disputed texts. Any suspiciously
high-ranking feature is checked for per-play concentration and inspected in context before
being trusted as real stylistic signal rather than a formatting or edition artifact.

### Stage 5 — Disputed Text Attribution
Each disputed play is chunked the same way as the training data and every chunk gets an
independent `P(Shakespeare)`. These are aggregated per play (mean/std/min/max) to summarize
the overall verdict and how much it varies within the text.

---

## Key Outputs

- A cleaned corpus of 62 Early Modern texts: 33 Shakespeare works, 23 works across seven
  contemporary authors, and 5 held-out disputed texts 
- Comparative classifier evaluation with grouped cross-validation
- SHAP feature-importance analysis for the main model, and coefficient inspection for the
  Marlowe/Fletcher pairwise models
- Per-chunk and per-play authorship probability scores for Henry VI I–III, The Spanish
  Tragedy, and Double Falsehood
---
## Interactive Dashboard

A Streamlit app (`app.py`) provides a browser-based view into the precomputed pipeline
outputs (per-play probabilities and SHAP values).

**Attribution Dashboard tab**
- Pick a disputed play and see the model's verdict (Shakespeare / Contemporary) and its mean
  aggregate probability across that play's chunks
- A SHAP feature-importance bar chart showing the top 20 features driving predictions for
  that play, colored by direction (toward Shakespeare vs. toward a contemporary author)
- A highlighted-text view of one representative chunk, with stylistic tokens colored by
  their SHAP contribution

**Discovery Engine tab**
- Top recurring unigrams, bigrams, and trigrams for a single chunk of the selected play, as
  a lightweight, exploratory look at repeated phrasing (this is exploratory
  only, not a model output)

To run it locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

It expects the precomputed artifacts (`disputed_df`, `per_play_summary`,
`shap_values_disputed`, `feature_names`, `text_registry`)
---

## Project Structure

```
shakespeare_project/
│
├── texts/                                    # Raw + cleaned corpus
│
├── 01_data_collection_eda.ipynb              # Corpus acquisition, cleaning, EDA
├── 02_feature_engineering.ipynb              # Chunking, feature extraction, Zeta analysis
├── 03_modeling.ipynb                         # Classifier training, CV, test evaluation
├── 04_shap_disputed_inference.ipynb          # Disputed-text inference + SHAP
├── 06_pairwise_author_attribution.ipynb      # Shakespeare-vs-Marlowe / Shakespeare-vs-Fletcher
│
├── report/
│   └── report.pdf
│
└── README.md
```

---

## References

Key prior work this project builds on:

1. Shapiro, J. *Contested Will: Who Wrote Shakespeare?* (Simon and Schuster, New York, 2010).
2. Taylor, G. & Egan, G. *The New Oxford Shakespeare: Authorship Companion* (Oxford University Press, 2017).
3. Stamatatos, E. A survey of modern authorship attribution methods. *J. Am. Soc. Inf. Sci. Technol.* 60, 538–556 (2009).
4. Mendenhall, T. C. The characteristic curves of composition. *Science* (1887).
5. Burrows, J. Word-patterns and story-shapes: The statistical analysis of narrative style. *Lit. Linguist. Comput.* (1987).
6. Savoy, J. *Machine Learning Methods for Stylometry: Authorship Attribution and Author Profiling* (Springer, 2020).
7. Marsden, J., Budden, D., Craig, H. & Moscato, P. Language individuation and marker words: Shakespeare and his maxwell's demon. *PLOS ONE* (2013).
8. Arefin, A. S., Vimieiro, R., Riveros, C., Craig, H. & Moscato, P. An information theoretic clustering approach for unveiling authorship affinities in Shakespearean era plays and poems. *PLOS ONE* (2014).
9. Boyd, R. L. & Pennebaker, J. W. Did Shakespeare write Double Falsehood? Identifying individuals by creating psychological signatures with text analysis. *Psychol. Sci.* (2015).
10. Kernot, D., Bossomaier, T. & Bradbury, R. Novel psychological text feature RPAS and authorship analysis of Shakespeare, Marlowe, and Cary (2018).
11. Segarra, S., Eisen, M., Egan, G. & Ribeiro, A. Attributing the authorship of the Henry VI plays by word adjacency. *Shakespeare Q.* (2016).
12. Segarra, S., Eisen, M., Egan, G. & Ribeiro, A. Authorship attribution through function word adjacency networks (2018).
13. Plecháč, P. Relative contributions of Shakespeare and Fletcher in Henry VIII: An analysis based on most frequent words and most frequent rhythmic patterns. *Digit. Scholarsh. Humanit.* (2019).
14. Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. NeurIPS (2017).
15. Burrows, J. All the way through: Testing for authorship in different frequency strata. *Lit. Linguist. Comput.* (2007).
