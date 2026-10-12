"""Hugging Face fine-tuning: DistilBERT (English), CamemBERT (French) and the
PANDORA multi-task DistilBERT with five regression heads.

Requires ``torch``, ``transformers`` (and ``datasets`` for the classification
helper).  Everything heavy is imported lazily inside functions except PyTorch
(needed to define ``MultiTraitBERT``).

``TrainingArguments`` / ``Trainer`` changed several argument names between
``transformers`` 4.x and 5.x (``evaluation_strategy`` → ``eval_strategy``,
``no_cuda`` → ``use_cpu``, ``tokenizer`` → ``processing_class``).  The notebook
handled that by introspecting the signatures; the same shims are kept here.
"""

from __future__ import annotations

import inspect
import logging
import time
import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


logger = logging.getLogger(__name__)

# --- Presets reproducing the original runs ------------------------------------
DAIR_DISTILBERT_CPU = dict(
    model_name="distilbert-base-uncased",
    max_samples=8_000,          # first 8,000 rows of each split (NOT the full test split)
    max_length=64,
    training_kwargs=dict(
        learning_rate=5e-5, per_device_train_batch_size=4, per_device_eval_batch_size=4,
        num_train_epochs=1, weight_decay=0.01, save_strategy="epoch", load_best_model_at_end=True,
        report_to="none", dataloader_num_workers=0,
    ),
    cpu=True,
)
FRENCH_CAMEMBERT = dict(
    model_name="camembert-base",
    train_size=5_000, val_size=1_000, max_length=128,
    training_kwargs=dict(
        save_strategy="epoch", per_device_train_batch_size=4, per_device_eval_batch_size=4,
        num_train_epochs=2, learning_rate=2e-5, weight_decay=0.01, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", report_to="none",
    ),
    cpu=False,
)
PANDORA_BERT = dict(
    model_name="distilbert-base-uncased", max_len=128, n_train=500_000, n_ensemble=50_000,
    n_val_subset=20_000, n_traits=5, dropout=0.3, freeze_layers=3, head_hidden=256,
)


# --- transformers-version compatibility shims ---------------------------------
def build_training_arguments(cpu: bool = False, **kwargs):
    """``TrainingArguments`` with version-dependent names resolved.

    ``eval_strategy="epoch"`` is always requested; keyword arguments not accepted
    by the installed ``transformers`` are dropped with a warning (this mirrors the
    notebook's ``{k: v for k, v in wanted.items() if k in accepted}``).
    """
    from transformers import TrainingArguments

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_key = "eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"
    cpu_key = "use_cpu" if "use_cpu" in accepted else "no_cuda"
    wanted = dict(kwargs)
    wanted.setdefault(eval_key, "epoch")
    wanted[cpu_key] = cpu
    dropped = sorted(k for k in wanted if k not in accepted)
    if dropped:
        warnings.warn(f"TrainingArguments ignored unsupported arguments: {dropped}")
    return TrainingArguments(**{k: v for k, v in wanted.items() if k in accepted})


def trainer_tokenizer_kwargs(tokenizer) -> Dict:
    from transformers import Trainer

    params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    return {}


# --- Sequence classification (English DistilBERT / French CamemBERT) ----------
def accuracy_macro_f1_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def macro_f1_metric(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, predictions, average="macro", zero_division=0)}


def fine_tune_sequence_classifier(
    model_name: str,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    test_texts: Sequence[str],
    test_labels: Sequence[int],
    num_labels: int,
    training_kwargs: Dict,
    output_dir: str,
    max_length: int,
    compute_metrics: Callable,
    label_column: str = "labels",
    use_padding_collator: bool = True,
    cpu: bool = False,
) -> Dict:
    """Fine-tune ``AutoModelForSequenceClassification`` and predict on the test set.

    Texts are tokenised with ``padding='max_length'`` and ``truncation=True``.
    Returns ``y_pred``, ``y_true``, ``train_time``, ``inference_time`` and the trainer.
    The best checkpoint is selected on the **validation** set
    (``load_best_model_at_end``); the test set is only used for the final prediction.

    Equivalence note: the original French cell did not pass the tokenizer / a
    collator to the ``Trainer``.  Since every example is already padded to
    ``max_length``, passing them (as the English cell did) yields identical batches.
    """
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    def to_ds(texts, labels):
        ds = Dataset.from_dict({"text": list(texts), label_column: list(labels)})
        return ds.map(tokenize, batched=True)

    train_ds, val_ds, test_ds = (
        to_ds(train_texts, train_labels),
        to_ds(val_texts, val_labels),
        to_ds(test_texts, test_labels),
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    args = build_training_arguments(cpu=cpu, output_dir=output_dir, **training_kwargs)

    extra = {}
    if use_padding_collator:
        extra["data_collator"] = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        **extra,
        **trainer_tokenizer_kwargs(tokenizer),
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    t1 = time.time()
    predictions = trainer.predict(test_ds)
    inference_time = time.time() - t1

    return dict(
        y_pred=np.argmax(predictions.predictions, axis=1),
        y_true=np.asarray(test_labels),
        train_time=train_time,
        inference_time=inference_time,
        trainer=trainer,
    )


# --- PANDORA multi-task DistilBERT --------------------------------------------
class MultiTraitBERT(nn.Module):
    """Shared encoder (CLS token) + one MLP regression head per Big Five trait.

    Heads: ``Linear(hidden, 256) → GELU → Dropout → Linear(256, 1)``.
    Loss: MSE between the 5 concatenated outputs and the [0, 1] targets.
    The ``freeze_layers`` lowest DistilBERT transformer layers are frozen.
    ``encoder`` may be injected (used by the unit tests with a tiny random model).
    """

    def __init__(
        self,
        model_name: str = PANDORA_BERT["model_name"],
        n_traits: int = 5,
        dropout: float = 0.3,
        freeze_layers: int = 3,
        cache_dir=None,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        if encoder is None:
            from transformers import AutoModel

            kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
            encoder = AutoModel.from_pretrained(model_name, **kwargs)
        self.encoder = encoder
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1)
                )
                for _ in range(n_traits)
            ]
        )
        if hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
            for i, layer in enumerate(self.encoder.transformer.layer):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
            logger.info("Froze %d/%d encoder layers", freeze_layers, len(self.encoder.transformer.layer))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        out = self.dropout(out)
        logits = torch.cat([h(out) for h in self.heads], dim=-1)
        loss = nn.functional.mse_loss(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def multitrait_training_kwargs(output_dir: str, cpu_mode: bool, seed: int = 42) -> Dict:
    """Training arguments of the PANDORA BERT run (GPU values unless ``cpu_mode``)."""
    return dict(
        output_dir=output_dir,
        num_train_epochs=2 if not cpu_mode else 1,
        per_device_train_batch_size=64 if not cpu_mode else 4,
        per_device_eval_batch_size=128 if not cpu_mode else 4,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.1,
        warmup_ratio=0.1,
        eval_steps=1000,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1_avg",
        greater_is_better=True,
        logging_steps=200,
        report_to="none",
        fp16=not cpu_mode,
        bf16=False,
        dataloader_num_workers=0 if cpu_mode else 2,
        seed=seed,
        max_grad_norm=1.0,
    )


def make_multitrait_compute_metrics(
    thresholds_median: Dict[str, float], traits: Sequence[str]
) -> Callable:
    """Metric used for checkpoint selection: mean macro-F1 over traits at the
    **training-median** thresholds (applied to the validation subset)."""
    thr_arr = np.array([thresholds_median[c] / 100.0 for c in traits])
    n = len(traits)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds).reshape(-1, n)
        labels = np.asarray(labels).reshape(-1, n)
        preds_bin = (preds >= thr_arr).astype(int)
        labels_bin = (labels >= thr_arr).astype(int)
        f1s = [
            f1_score(labels_bin[:, i], preds_bin[:, i], average="macro", zero_division=0)
            for i in range(n)
        ]
        metrics = {"macro_f1_avg": float(np.mean(f1s))}
        for i, c in enumerate(traits):
            metrics[f"f1_{c}"] = float(f1s[i])
        return metrics

    return compute_metrics


def predict_multitrait(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: str,
    max_length: int = 128,
    batch_size: Optional[int] = None,
    n_traits: int = 5,
) -> np.ndarray:
    """Batched inference returning an ``(n, n_traits)`` float32 array (fp16 autocast on CUDA)."""
    if batch_size is None:
        batch_size = 8 if device == "cpu" else 128
    model.eval()
    all_preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = tokenizer(
                batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output = model(**enc)
            else:
                output = model(**enc)
            all_preds.append(output["logits"].float().cpu().numpy())
    if not all_preds:
        return np.empty((0, n_traits), dtype=np.float32)
    return np.concatenate(all_preds, axis=0).astype(np.float32)
