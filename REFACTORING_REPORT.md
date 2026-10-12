# Refactoring report

Source: `deepseek_json_20260916_9fe93d_2_2_2-3.ipynb` (1.68 MB, nbformat 4.5, 121 cells = 82 code + 39 markdown, Colab metadata, A100 GPU). Backup: `archive/original_master_notebook.ipynb`
(SHA-256 `3889b1e2…dec2fe2`, identical to the upload). The original was not modified.

**Headline.** The notebook is one "master notebook" with **four** workflows (not three): English emotions, PANDORA Big Five (which also contains the French emotion experiments), and an LLM section.
All three requested tasks are *implemented*, but **only part of them have recorded results**. The refactored notebooks and the torch / Hugging Face code **were not executed** (no datasets, no GPU, PyTorch not installable in the preparation sandbox).

## 1. Original notebook inventory
| Cells | Content | Recorded outputs? |
|---|---|---|
| 0 | version printout (transformers 5.18.0, peft 0.21.1, accelerate 1.15.0, torch 2.11.0+cu130) | yes |
| 1-2 | table of contents; "Executive Summary PANDORA v3" (architecture, goal F1 > 0.70) | - |
| 3-18 | **Part 1 English dair-ai**: setup, load, de-dup + clean, 70/15/15 split, EDA, 4 TF-IDF models, LSTM/GRU/BiLSTM, DistilBERT-CPU, comparison, confusion matrix, LIME, export | yes (all) |
| 19-28 | **Part 2 PANDORA** setup (hardware, seeds, paths, installs), data loading (PANDORA, EN emotions, local French CSV via `input()`), cleaning | yes |
| 29-46 | EDA (distributions, quality, correlations, leakage notes, key findings) | yes |
| 47-59 | labels (train medians), Ridge TF-IDF baseline, LogReg/NB/SVM (300k), LSTM/BiLSTM/GRU (100k) | yes |
| 60-66 | multi-task DistilBERT (500k + 50k held-out) | only cell 65 (7.66 min) |
| 67-75 | BERT predictions, SVD, stacked ensemble, BGE-M3 | **none** |
| 76-78 | S9 EN/FR emotions: `prepare_emotion`, 80/20 LogReg + LinearSVM | yes |
| 79-84 | French Part A (RNNs), Part B (CamemBERT), Part C/D benchmark | **none** |
| 85-86 | "S4" French RNN (duplicate protocol) | **none** |
| 87-91 | final comparison (val) + ensemble test score + config JSON | only cell 89 (test) |
| 92-95 | LIME for PANDORA Extraversion | yes |
| 96-98 | stray "test" cell, embedding demo (no output), `df_train.tail` | none |
| 99-117 | **Part 3 LLM**: Qwen zero-shot, Unsloth LoRA (cells 110-111 error out; 114/116/117 work) | yes |
| 118-119 | "checklist" claiming all tasks are done | yes |

**Datasets / targets:** dair-ai/emotion (6 classes); `emotions_fr_final.csv` (local, 6 classes); `jingjietan/pandora-big5` (OCEAN 0-100); `personality_emotions_french_no_labels.jsonl` (local, trait name).
**Reproducibility settings:** `SEED=42` (random / numpy / torch), `random_state=42` in splits and sklearn models, Unsloth seeds 3407 / 42.
**Generated artefacts in the source:** CSVs written to the working directory (`english_dair_ai_cleaned.csv`, `english_dair_ai_model_comparison.csv`, `emotions_*.csv`, `french_*`, `comparison_val.csv`, `config_v3.json`),
`best_<rnn>.pt`, BERT checkpoints (`bert_mt`, `bert_mt_final`), `.npy` prediction files, 12 figures embedded in outputs. None of these files was supplied; only the embedded outputs could be recovered.

**Hidden state / execution-order dependencies found** (all resolved by making each notebook self-contained):
`models`, `vectorizer`, `X_*_tfidf`, `vocab`, `train_loader`, `y_test`, `X_train`, `SEED`, `DEVICE`, `MAX_LEN` are re-assigned by later parts (English → PANDORA → French);
cell 14/16 depend on cell 9's variables; cell 41 raises if earlier cells were skipped; cells 80/82 depend on cell 78; cell 110 raised `NameError: DATA_DIR` in the source; cell 18 refers to `RESULTS_DIR` "defined in Part 2/3".

## 2. Notebook-to-module mapping
| Original cells | Destination |
|---|---|
| 4-5 loading, label checks | `src/data/emotion.py` (`load_dair_emotion`, `validate_dair_label_names`) |
| 6, 28 cleaners; 78 `prepare_emotion` | `src/preprocessing/text_cleaning.py` |
| 25, 26 | `src/data/pandora.py`, `src/data/emotion.py` (`load_french_emotions`; `input()` prompt replaced by an explicit path) |
| 7, 61, 80 split code | `src/data/splits.py` |
| 48 labels | `src/preprocessing/labels.py` |
| 10, 58, 80, 86 vocab / `encode_*` | `src/preprocessing/sequences.py` |
| 9, 51, 78 TF-IDF | `src/features/tfidf.py` ; 71 SVD → `features/dimensionality.py` ; 74-75, 97 → `features/embeddings.py` |
| 9, 54, 78, 52, 72, 75 estimators | `src/models/classical.py` |
| 10, 59, 80, 86 RNN classes + loops | `src/models/rnn.py` (4 classes → 1, 4 loops → 2 functions + presets) |
| 11, 63-65, 68, 82 | `src/models/hf_finetuning.py` |
| 72 ensemble | `src/models/ensemble.py` |
| 104, 110, 114, 117 prompt formats | `src/models/llm.py` (rest stays in the LLM notebook) |
| 9, 49 metrics, thresholds | `src/evaluation/metrics.py` |
| 14, 56, 59, 84 plots / tables | `src/evaluation/reporting.py` ; 16, 95 LIME → `evaluation/explain.py` |
| 0, 20-22 seeds / device / paths | `src/utils/` |
| cells 3-18 | `notebooks/english_emotion.ipynb` (+ S9 English part of 76-78) |
| 76-86 French parts of 26, 29-45, 78-86 | `notebooks/french_emotion.ipynb` |
| 19-75, 87-98 | `notebooks/personality_prediction.ipynb` |
| 99-117 | `notebooks/llm_inference_finetuning.ipynb` (extra) |
| 1-2, 39, 46 explanations | kept in notebook markdown / `docs/methodology.md` |
| 18, 118-119 (Drive export tips, "checklist") | dropped (no scientific content; the checklist claims completion that the outputs do not support) |

## 3. Files created (74 files, excluding `archive/` and caches)
`README.md`, `REFACTORING_REPORT.md`, `.gitignore`, `requirements.txt`, `requirements-llm.txt`, `docs/methodology.md`, `results/README.md`, 4 notebooks, 25 source files under `src/`
(no empty modules; `src/models/llm.py` is torch-free by design), 6 test files + `conftest.py`, 15 CSV + 12 PNG recorded results, `archive/` (backup + `extract_recorded_results.py`).
Nothing was deleted, moved or overwritten. **No `LICENSE` was created** (see §9).

## 4. Differences to the requested structure
Added: a 4th notebook (LLM, because it is implemented work), `tests/`, `requirements-llm.txt`, `archive/`, `results/README.md`. Not created: `LICENSE`. No empty `src/` subpackage exists.

## 5. Methodological changes
**No metric, split, seed, hyper-parameter or model architecture was changed.** Deviations that do not alter results *when run on the same state*, all documented in the notebooks:
1. French CamemBERT: the 5,000-row draw is now seeded (`np.random.seed(42)` before `np.random.choice`); the original draw depended on unrecorded RNG state, so the exact original subset is unrecoverable. `try/except … FAILED` removed (errors propagate).
2. Early stopping: best weights kept in memory (and optionally saved under `checkpoints/`) instead of `best_<rnn>.pt` in the working directory; same selection rule.
3. S9 report: each model's report now carries its own name (original printed LinearSVM predictions under "LogReg"). Metric tables unchanged.
4. PANDORA classical models: fresh estimator per trait instead of re-fitting the same object (equivalent).
5. `report_to="none"` in the LLM SFT (original logged to a personal W&B account). Logging only.
6. Removed the hard-coded `0.614 "v1 LogReg"` line and the derived "gain" printout (no source, compared a test score to a validation baseline).
7. Seaborn calls use `hue=`/`legend=False` instead of the deprecated `palette`-only form (same colours).
8. Original `input()` prompts and `/content`, `~/PANDORA_Project`, Google-Drive paths replaced by configurable paths.
9. Not carried over: the laptop-CPU smoke branches of the PANDORA BERT cells (no recorded result; BERT section now requires CUDA).
Known issues of the original that were **left unfixed on purpose** are in `docs/methodology.md` §7 (decisions in §9 below).

## 6. Experiments and checks actually executed
Environment: Python 3.12.3, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8, seaborn 0.13.2, lime 0.2.0.1, pytest 9.1.1, nbformat 5.11.1; 1 CPU, 3 GB RAM, no GPU, **no PyTorch / transformers / datasets / sentence-transformers**.
| Check | Result |
|---|---|
| `pytest tests` | **55 passed, 1 skipped** (the skipped item is `tests/test_torch_models.py`, 15 test cases, auto-skipped: no PyTorch) |
| Equivalence of refactored torch-free code with the original notebook's *own source* (cleaners, `prepare_emotion`, vocab / `encode_dl`, `evaluate_binary`, `find_best_threshold`, `evaluate_model`, English / French splits, bootstrap of ensemble indices, TF-IDF configs, all sklearn model hyper-parameters via `get_params()`, the ensemble loop, LLM prompt formats) | passed |
| Mutation test of those checks (6 deliberate bugs injected, e.g. dropped apostrophe, changed `dual`, wrong val fraction, threshold grid) | 5 caught immediately; the threshold-grid mutation was **missed at first** → a test was added, re-mutated, now caught; source restored and diffed identical |
| Imports of `src.utils/data/preprocessing/features/models/evaluation` in a torch-free interpreter | ok |
| `py_compile` + `pyflakes` on `src/` and `tests/` (incl. torch / HF modules) | clean |
| Notebooks: nbformat schema, no stored outputs, every cell parses, every `from src… import name` resolves (checked statically), no machine-specific strings, setup cell executed from repo root and from `notebooks/` | passed (4 notebooks) |
| Smoke pipeline on synthetic data (clean → TF-IDF → 4 models → metrics → plots → LIME helper) | passed |
**Not run (could not):** any dataset download; any real experiment; `src/models/rnn.py`, `hf_finetuning.py` and their 15 tests (no PyTorch wheel installable: only the CUDA build is on PyPI and the CPU index is outside the sandbox allow-list);
`transformers` argument shims; sentence-transformers / BGE-M3; Unsloth; execution of any full notebook. **These modules are unverified beyond syntax/lint and static import resolution.**

## 7. Results-preservation comparison
Recomputing the notebook's metrics was impossible (no data / GPU). What was compared: (a) code and hyper-parameters against the original cells (§6, passed); (b) recorded outputs were re-exported with provenance
(`results/`, 15 tables + 12 figures, each tagged `__src-cell<N>`); table values were parsed programmatically (row/column counts checked: e.g. 15 classical rows = 5 traits × 3 models, 6 rows ensemble test).
Recorded ≠ re-verified: expect DL numbers to differ slightly on re-run (global RNG consumption, GPU non-determinism, different library versions).

## 8. Missing dependencies, data and checkpoints; environment contradictions
* **Data not supplied:** `emotions_fr_final.csv`, `personality_emotions_french_no_labels.jsonl`. The HF datasets are downloaded (revisions unpinned). The third column of the French CSV is unknown.
* **Checkpoints / predictions not supplied:** `bert_mt*`, `best_<rnn>.pt`, `bert_*_preds.npy`, Unsloth outputs, original CSV/JSON result files.
* **Recorded versions contradict each other:** transformers 5.18.0 (cell 0) vs 4.57.6 (cell 109) vs `<5` pinned (cell 4); peft 0.21.1 vs `<0.20`; Python 3.13.15 (cell 21, Colab path `python3.13`) vs 3.11.7 (metadata) vs `/usr/bin/python3` (cell 0); cell 41/78 code expects `fr_text_cleaned`.
  The English "CPU laptop" DistilBERT cell and the macOS-CPU messages in Part 3 suggest parts were run on a different machine than the Colab A100. Never recorded: pandas, scikit-learn, datasets, sentence-transformers, lime, trl versions.
* `requirements.txt` therefore mixes recorded pins and ranges and is **not verified by installation** (checked only: pinned versions exist on PyPI; sentence-transformers 6.x requires transformers ≥ 5, hence `<6`).

## 9. Confidentiality and licensing - decisions required from you
1. **NDA / publication rights:** not provided, so nothing was assumed. Who owns the code and may it be published? Which licence (none was chosen; no `LICENSE` file)?
2. **Private data:** may the French CSV and JSONL be used or referenced publicly? Their provenance (possible machine translation, "translation artifacts" noted in the notebook) is undocumented.
3. **PANDORA / Reddit terms:** the figures `…lime_explanation…src-cell16/95.png` display words taken from dataset texts; check dataset-card terms before publishing them.
4. **`archive/` content:** the original notebook contains a personal Weights & Biases username and organisation name (cell 116 output, `wandb: Currently logged in as …`), Colab paths, and a 338 KB `widgets` metadata blob. It is git-ignored; sanitise or exclude before publication.
5. **Whether to fix the methodological caveats** (`docs/methodology.md` §7) - especially the right-padding / last-position read-out, which likely explains the near-chance LSTM/BiLSTM scores, and the non-comparable French benchmark. Each is a scientific change that I did not make.
6. Whether to keep the 4th (LLM) notebook in the public version.

## 10. Remaining tasks
1. Run `pytest tests/test_torch_models.py` on a machine with PyTorch (+ transformers) and fix anything it reveals.
2. Execute the notebooks (English first; then French with the CSV; PANDORA on a GPU) and compare to `results/*__src-cell*.csv`; save new outputs under the same naming convention (without the `src-cell` suffix).
3. Pin a verified environment (`pip freeze` from a run) and update `requirements*.txt`; pin dataset revisions.
4. Obtain the missing results (French RNN/CamemBERT, PANDORA BERT / BGE-M3 / ensemble-validation) or state them as absent in the internship report.
5. Resolve §9, then (and only then) choose a licence, add `LICENSE`, sanitise `archive/`, and publish.
