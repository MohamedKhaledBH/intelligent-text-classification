# Intelligent Text Classification

Internship project: **emotion classification** (English, French) and **Big Five personality prediction** from text, refactored from a single
Colab "master notebook" into a modular Python repository.

> **Status: private working copy.** Publication rights, licence and NDA constraints are *unverified* (see [Data access, licensing and confidentiality](#data-access-licensing-and-confidentiality)).
> The repository has **no `LICENSE` file on purpose**. The refactored notebooks have **not been re-executed**: results in `results/` are the ones recorded in the original notebook.

## Research objectives
1. Compare classical (TF-IDF), recurrent and transformer models for six-class emotion classification in English and French.
2. Predict the five Big Five traits (binary, macro-F1) on PANDORA with TF-IDF, recurrent, multi-task BERT, BGE-M3 and a stacked ensemble.
3. (Exploratory) probe a small LLM for personality-trait generation.

## Tasks (only what the source notebook implements)
| Task | Notebook | Recorded results in the source |
|---|---|---|
| English emotion (dair-ai) | `notebooks/english_emotion.ipynb` | complete (8 models, error analysis, LIME) |
| French emotion | `notebooks/french_emotion.ipynb` | classical models only; RNN / CamemBERT outputs absent |
| Big Five on PANDORA | `notebooks/personality_prediction.ipynb` | baselines and ensemble test score; BERT / BGE-M3 metrics absent |
| LLM inference + LoRA (extra, not requested) | `notebooks/llm_inference_finetuning.ipynb` | qualitative only |

## Repository structure
```
├── README.md  REFACTORING_REPORT.md  requirements.txt  requirements-llm.txt  .gitignore
├── notebooks/        4 experiment entry points (no stored outputs)
├── src/
│   ├── data/           loaders + validation (dair-ai, French CSV, PANDORA) and the original split protocols
│   ├── preprocessing/  text cleaners, label binarisation, vocabulary / sequence encoding
│   ├── features/       TF-IDF configs, SVD, sentence embeddings
│   ├── models/         classical, recurrent (rnn), Hugging Face fine-tuning, stacked ensemble, LLM prompt formats
│   ├── evaluation/     metrics, reports, plots, LIME helper
│   └── utils/          seeding, device, paths (ITC_* env vars), result naming
├── tests/            pytest suite (equivalence with the original notebook's own code, notebook checks)
├── results/          tables/ and figures/ re-exported from the original run (see results/README.md)
├── docs/methodology.md
└── archive/          original notebook backup - git-ignored, contains private traces; not for publication as is
```
There is deliberately no `LICENSE` (see below).

## Environment setup
Python ≥ 3.10 (original: Python 3.13 on Colab; the torch-free tests ran on 3.12).
```bash
python -m venv .venv && source .venv/bin/activate
# CPU-only machine: install the CPU build of torch 2.11.0 first (https://pytorch.org), otherwise pip pulls the CUDA build
pip install -r requirements.txt
pip install -r requirements-llm.txt      # only for the LLM notebook (Linux + CUDA GPU)
```
`requirements.txt` pins only what the notebook printed (torch 2.11.0, transformers 4.57.6, accelerate 1.15.0, numpy 2.1.3) and uses ranges elsewhere.
**The original environment cannot be fully reproduced**: the notebook prints contradictory versions (e.g. transformers 5.18.0 in cell 0, 4.57.6 in cell 109, `<5` pinned in cell 4;
peft 0.21.1 vs `<0.20`; Python 3.13.15 vs 3.11.7) and never prints pandas / scikit-learn / datasets versions. Details: `REFACTORING_REPORT.md` §8.

GPU: the PANDORA multi-task BERT, BGE-M3 and the LLM notebook need a CUDA GPU (original: A100-80GB). The PANDORA TF-IDF + SVD on 1.9M rows needs tens of GB of RAM.

## Dataset acquisition and configuration
| Data | How to get it | Where |
|---|---|---|
| `dair-ai/emotion` | automatic download from the Hugging Face Hub (config `unsplit`); check the dataset card for its terms | HF cache (`cache/`) |
| `jingjietan/pandora-big5` | automatic download from the Hub; Reddit-derived - check the dataset card and Reddit/PANDORA terms before sharing any derived file | HF cache |
| `emotions_fr_final.csv` | **private, not distributed.** Ask the project owner. Needs columns `emotion` and `fr_text_cleaned` | `data/emotions_fr_final.csv` |
| `personality_emotions_french_no_labels.jsonl` | **private, not distributed**; only for the LLM notebook; columns `text`, `personality` | `data/` |

Locations are configurable: `ITC_DATA_DIR`, `ITC_RESULTS_DIR`, `ITC_CACHE_DIR`, `ITC_CHECKPOINT_DIR` (defaults: `./data`, `./results`, `./cache`, `./checkpoints`).
Nothing is tied to the original machine (`/content`, `~/PANDORA_Project`, Google Drive).

## Running
```bash
jupyter lab                      # launch from the repository root (or from notebooks/); each notebook finds the repo root itself
# open notebooks/english_emotion.ipynb, french_emotion.ipynb, personality_prediction.ipynb, (llm_inference_finetuning.ipynb)
pytest tests -v                  # tests; torch tests skip automatically if PyTorch is missing
```
There are no command-line scripts: notebooks are the entry points and call `src/`.

## Models and metrics
Models: LogReg, Linear SVM, Naive Bayes, SGD, Ridge; LSTM / GRU / BiLSTM; DistilBERT; CamemBERT; multi-task DistilBERT (5 regression heads); BGE-M3 + Ridge; stacked Ridge ensemble;
Qwen2.5 + LoRA. Metrics: accuracy, macro precision / recall / F1, weighted F1 (emotions); per-trait accuracy and macro-F1 on median-binarised scores (PANDORA). Full detail: [`docs/methodology.md`](docs/methodology.md).

## Verified results
Recorded in the original run, **not re-executed here** - see [`results/README.md`](results/README.md) for the index and provenance (`__src-cell<N>` suffix).
Headline recorded values: English test macro-F1 from 0.136 (LSTM) to 0.938 (GRU) across 8 models (classical 0.787-0.917; Linear SVM 0.9174; DistilBERT on an 8k-row test subset 0.9218);
French classical macro-F1 0.656 (LogReg) / 0.689 (LinearSVM); PANDORA ensemble test mean macro-F1 0.5745.
Results absent from the source: French RNN / CamemBERT, PANDORA BERT / BGE-M3 / ensemble-validation, LLM metrics.

## Known limitations and reproducibility notes
* Several caveats of the original experiments are **preserved, not fixed** (right-padded RNNs reading the last position, non-comparable benchmark rows, different splits for the same task,
  optimistic validation thresholds, unmeasured duplicate leakage): [`docs/methodology.md` §7](docs/methodology.md).
* Single run, single seed. GPU runs are not bit-reproducible.
* **Do not describe this project as fully reproducible** until the datasets are available and the notebooks have been re-run with a verified environment.

## Data access, licensing and confidentiality
* This is an internship project that **may be subject to an NDA**. No NDA or publication agreement was available during the refactor; nothing was assumed.
* Open decisions (owner / employer): who owns the code; whether the repository may be published; which licence; whether the French datasets, the French JSONL and any derived tables or figures may be shared.
* No `LICENSE` has been added on purpose. Without one, default copyright applies (all rights reserved).
* `archive/` (git-ignored) holds the original notebook, which contains Colab paths, a personal Weights & Biases account name in one output and ~340 KB of widget state. Sanitise or exclude it before any publication.
* Keep the repository **private** until the above is settled.
