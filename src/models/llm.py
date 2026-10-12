"""Prompt / chat-format helpers for the LLM experiments (Qwen2.5, Unsloth LoRA).

This module is deliberately small and **torch-free**: the original notebook's
LLM section is an exploratory, GPU-only demonstration (zero-shot Qwen2.5 on one
PANDORA text; LoRA fine-tuning of ``unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`` to
output the dominant Big Five *trait name* of a French sentence).  No
quantitative evaluation was recorded for it, so only the formatting logic —
which is reusable and testable — lives here.  The Unsloth model/trainer code
stays in ``notebooks/llm_inference_finetuning.ipynb``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

OCEAN_SYSTEM_PROMPT = (
    "You are an expert in psychology, emotional, and personality analysis. "
    "Provide concise answers."
)


def build_ocean_analysis_messages(text: str) -> List[Dict[str, str]]:
    """Chat messages of the zero-shot analysis prompt (notebook Part 3)."""
    prompt = (
        "Analyze the following text.\n"
        "1. Provide exactly 3 keywords describing ONLY the emotions expressed in it.\n"
        "2. Estimate the author's Big Five (OCEAN) personality scores on a scale of 0 to 100 "
        "(Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).\n\n"
        f"Text: '{text}'\n\n"
        "Analysis:"
    )
    return [
        {"role": "system", "content": OCEAN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def format_chatml_examples(inputs: Sequence[str], outputs: Sequence[str]) -> List[str]:
    """ChatML training strings used for the Unsloth SFT data."""
    return [
        f"<|im_start|>user\n{i}<|im_end|>\n<|im_start|>assistant\n{o}<|im_end|>\n"
        for i, o in zip(inputs, outputs)
    ]


def chatml_inference_prompt(text: str) -> str:
    """Prompt prefix used at inference time for the fine-tuned trait classifier."""
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
