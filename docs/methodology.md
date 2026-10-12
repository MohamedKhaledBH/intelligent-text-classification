# Methodology

This document describes what the **original notebook implemented**. It was written from the notebook source and recorded outputs
(`archive/original_master_notebook.ipynb`, 121 cells). Cell numbers below refer to that file. Three labels are used consistently:

* **Implemented** - code exists in the notebook (and in `src/` / `notebooks/`).
* **Recorded** - a result was printed in the notebook's outputs. It was *not* re-executed by the refactor.
* **Proposed** - not done in the notebook; never presented as a result.

The refactor changed no metric definition, split, seed or hyper-parameter; the few deviations are listed in `REFACTORING_REPORT.md` §5.

## 1. Objectives
Intern project "Intelligent Text Classification": (a) six-class emotion classification of English and French text; (b) prediction of Big Five (OCEAN) personality traits
from text; (c) an exploratory LLM section. The notebook contains all three, plus an LLM section, as one linear "master notebook" in which the parts share variables.

## 2. Datasets and targets
| Dataset | Source | Size in the original run | Target |
|---|---|---|---|
| English emotions | Hugging Face `dair-ai/emotion`, config `unsplit` | 416,809 rows (393,822 after exact-text de-duplication in Part 1; 416,123 after S9 `prepare_emotion`) | 6 classes: sadness, joy, love, anger, fear, surprise |
| French emotions | **local file** `emotions_fr_final.csv` (provenance not documented in the notebook) | 29,286 rows (28,996 after `prepare_emotion`) | `emotion` column, 6 classes (distribution recorded: joy 28.6 %, sadness 25.0 %, love 18.7 %, surprise 9.9 %, anger 9.6 %, fear 8.1 %) |
| PANDORA Big Five | Hugging Face `jingjietan/pandora-big5` (Reddit comments) | train 1,924,201 / val 481,051 / test 601,314 (1,919,729 / 479,864 / 599,901 after cleaning) | O, C, E, A, N scores 0-100; regression on `score/100`, evaluation as binary `score >= train median` |
| French personality/emotion JSONL | **local file** `personality_emotions_french_no_labels.jsonl` | 3,400 rows | trait name (`personality` column) for LLM fine-tuning |

## 3. Data preparation
* **English Part 1:** drop exact duplicate `text` → `clean_text_dair` → stratified 70/15/15 split (`random_state=42`).
* **English S9 / French classical:** `prepare_emotion` (drop NaN, strip, drop empty, drop duplicate `(text, emotion)`, drop classes with ≤ 50 rows), stratified 80/20.
* **French Part A / B:** stratified 15 % test, then validation = 0.15/0.85 of the rest (≈ 70/15/15); training set subsampled (stratified) to 20,000.
* **PANDORA:** `clean_text_pandora` (lower-case, remove @mentions / HTML entities / whitespace chars / non `[a-z0-9']`), drop empty cleaned texts; train medians for binarisation;
  inside the training split: first 300k rows (classical binary models), first 100k rows (recurrent models), a random 500k / disjoint 50k partition (BERT / ensemble).

## 4. Features
* TF-IDF (fit on train only): English Part 1 (20k, 1-2 grams, sublinear); PANDORA (150k, 1-2 grams, `min_df=3`, `max_df=0.95`, sublinear, `lowercase=False`); S9 (50k, 1-2 grams, `min_df=2`, `max_df=0.95`, sublinear).
* Truncated SVD to 300 dimensions on the PANDORA TF-IDF matrix (explained variance recorded in the notebook: 0.149).
* Word-index sequences for the recurrent models (whitespace tokens, most-frequent-word vocabulary, right padding).
* Sentence embeddings `BAAI/bge-m3` (max length 256) for PANDORA (no result recorded).

## 5. Models and training
| Block | Models |
|---|---|
| Classical emotion | LogisticRegression (balanced), LinearSVC (balanced), MultinomialNB, SGDClassifier (balanced) |
| Recurrent | Embedding(128) → LSTM / GRU / BiLSTM (hidden 64) → last time-step → Dropout 0.3 → Linear. Adam; English lr 1e-3 × 5 epochs; PANDORA lr 2e-3 × 3 epochs, BCE; French Part A: class-weighted CE, ReduceLROnPlateau + early stopping |
| Transformers | English `distilbert-base-uncased` (8k rows, 1 epoch, CPU config); French `camembert-base` (5k rows, 2 epochs); PANDORA multi-task DistilBERT (CLS → 5 MLP regression heads, MSE, 3 frozen layers, 500k rows, 2 epochs, fp16) |
| PANDORA classical | Ridge on TF-IDF (regression, full train); LogReg / NB / LinearSVC binary (300k rows) |
| PANDORA ensemble | Ridge on `[SVD(TF-IDF) \| BERT predictions]` fitted on a disjoint 50k held-out block; BGE-M3 + multi-output Ridge |
| LLM (exploratory) | Qwen2.5-0.5B-Instruct zero-shot (one example); Qwen2.5-1.5B-Instruct 4-bit + LoRA (r=16) via Unsloth, 1 epoch on 3,400 rows |

## 6. Evaluation protocol and metrics
* Emotions: accuracy, macro precision/recall/F1, weighted F1, train/inference time; classification reports; confusion matrix; LIME on one instance.
* PANDORA: accuracy and macro P/R/F1 per trait on the binary labels; mean over the five traits. Decision threshold = train median / 100, except the ensemble and BGE-M3 whose
  per-trait threshold is selected on **validation** (grid `np.arange(0.20, 0.81, 0.01)`, maximise macro-F1).
* Test set: English models are scored on the test split (no validation use except DistilBERT checkpoint selection); French Part A selects the checkpoint by validation loss;
  PANDORA BERT selects its checkpoint on a 20k validation subset; **the test split is scored only for the final ensemble**.
* Single run, single seed (42) for everything; no cross-validation, no confidence intervals, no significance tests.

## 7. Known methodological caveats (preserved, NOT corrected)
Correcting any of these would change the experiment and requires the author's decision.
1. **Right padding + last-position read-out.** Sequences are padded on the right with 0 and every recurrent model reads `rnn_out[:, -1, :]`, i.e. the state after the padding for texts shorter than `max_len`
   (BiLSTM backward half sees only the last pad token). Recorded symptoms: English LSTM / BiLSTM accuracy 0.349 / 0.415 vs GRU 0.963; PANDORA recurrent macro-F1 ≈ 0.37 (< 0.5). Causal link **not tested**.
2. **Non-comparable rows.** The English DistilBERT row is trained on 8k and scored on the first 8k test rows; French classical rows (80/20 split) and DL/CamemBERT rows (70/15/15 split) are scored on different test sets.
3. **Different protocols for "the same" task.** English Part 1 vs S9 differ in cleaning, de-duplication, split and TF-IDF; the best classical model flips between them (SVM in Part 1, LogReg in S9).
4. **Mislabelled report.** The S9 cell prints "Rapport classification EN/FR (LogReg)" using the predictions of the *last fitted* model (LinearSVM). Metrics tables are unaffected.
5. **Optimistic validation scores.** For the ensemble and BGE-M3 the threshold is tuned and scored on the same validation split. (Test is untouched.)
6. **Possible leakage not measured.** 8.27 % of PANDORA training texts and 5.51 % of the English raw data are duplicates; overlap across PANDORA splits and author-level overlap (no author column loaded) were never checked.
7. **Subsets by row order.** "First N rows" subsets (300k, 100k, 50k, 8k) assume the stored order is random; this was not verified.
8. **Threshold grid quirk.** `np.arange(0.20, 0.81, 0.01)` has 62 points ending at ≈ 0.81 because of floating point; a strict `>` keeps the first maximum. Reproduced as is.
9. **Unsupported statements in the original text.** (a) "max_length 128 … without truncating meaningful data": the recorded EDA shows 116,790 PANDORA training texts (6.08 %) longer than 128 words. (b) "OCEAN scores follow roughly normal distributions": the recorded quartiles
   (e.g. Extraversion 13 / 31 / 61) suggest broad, asymmetric distributions; not tested. (c) The summary goal "F1 > 0.70" and the hard-coded "v1 LogReg 0.614" baseline have no supporting output.
10. **Stale / missing outputs.** Cells 66, 68, 69, 72, 74, 75, 80, 82, 84, 86, 88 have no output; the Part 2 sections therefore have no recorded BERT, BGE-M3, French DL or CamemBERT result. Whether the recorded outputs that do exist come from the current version of each cell cannot be established from the file.
11. **Environment drift.** Library versions printed in different cells contradict each other (see `REFACTORING_REPORT.md` §8).

## 8. Experiments: implemented vs recorded vs proposed
| Experiment | Implemented | Recorded result |
|---|---|---|
| English: 4 classical, 3 RNN, DistilBERT-CPU, error analysis, LIME | yes | yes (cells 9-16) |
| English S9 80/20 pipeline | yes | yes (cell 78) |
| French classical (LogReg, LinearSVM) | yes | yes (cell 78) |
| French GRU / LSTM / BiLSTM, CamemBERT, benchmark, legacy S4 RNN | yes | **no** |
| PANDORA: Ridge TF-IDF, LogReg / NB / SVM, LSTM / BiLSTM / GRU | yes | yes (cells 52, 54, 59) |
| PANDORA: multi-task BERT | yes | only training time (7.66 min) |
| PANDORA: ensemble | yes | test score only (cell 89: mean macro-F1 0.5745) |
| PANDORA: BGE-M3 + Ridge, final val comparison table | yes | **no** |
| LIME for English emotion and PANDORA Extraversion | yes | yes (single examples, figures) |
| LLM zero-shot + Unsloth LoRA | yes (exploratory) | qualitative only (5 hand-written sentences) |
| Proposed / future work (not in the notebook) | - | Fixing the padding read-out; same-test-set French comparison; author-level splits; multiple seeds; threshold selection on validation for every model |

## 9. Reproducibility considerations
* Seeds: Python / NumPy / PyTorch seeded with 42 once at the start; scikit-learn models and splits take `random_state=42` (S9 models have none). No deterministic-cuDNN flags, so GPU runs are not bit-reproducible.
* The recurrent models draw from the global torch RNG sequentially (English: LSTM → GRU → BiLSTM), so their results depend on the order of operations; the refactored notebooks keep the original order and seed once.
* Hardware: Colab A100-80GB for the GPU parts; the English DistilBERT "CPU configuration" was designed for a laptop. TF-IDF + SVD on 1.9M rows took 248 s + 576 s and substantial RAM.
* Data: two of four datasets are private local files; the Hugging Face datasets may change on the Hub (no revision is pinned).
* Library versions are only partly recoverable (README, `requirements.txt`).

## 10. Limitations
Single-split, single-seed evidence; modest PANDORA scores (≈ 0.55-0.61 macro-F1); unknown provenance of the French data and label quality; no author-aware evaluation;
no calibration or error analysis for personality prediction; LLM section anecdotal.
