"""Prompt-format helpers vs. the original notebook cells (torch-free)."""

import pandas as pd

from conftest import cell_source, exec_definitions

C_FORMAT_INPUT_OUTPUT, C_FORMAT_TRAITS, C_ZERO_SHOT = 110, 114, 104


def test_chatml_examples_match_original(original_nb):
    from src.models.llm import format_chatml_examples

    f = exec_definitions(original_nb, C_FORMAT_TRAITS, ["formatting_prompts_func_traits"])[
        "formatting_prompts_func_traits"
    ]
    texts = ["Je suis calme.", "J'adore les fêtes !"]
    labels = ["Neuroticism", "Extraversion"]
    assert format_chatml_examples(texts, labels) == f({"text": texts, "personality": labels})["formatted_text"]

    g = exec_definitions(original_nb, C_FORMAT_INPUT_OUTPUT, ["formatting_prompts_func"])["formatting_prompts_func"]
    assert format_chatml_examples(texts, labels) == g({"input": texts, "output": labels})["formatted_text"]


def test_zero_shot_messages_match_original(original_nb):
    from src.models.llm import build_ocean_analysis_messages

    src = cell_source(original_nb, C_ZERO_SHOT)
    prefix = src.split("# Format for Qwen")[0]
    df_test = pd.DataFrame(
        {"text_clean": ["x"] * 50, "O": [1.0] * 50, "C": [2.0] * 50, "E": [3.0] * 50, "A": [4.0] * 50, "N": [5.0] * 50}
    )
    df_test.loc[42, "text_clean"] = "some text, with 'quotes'"
    ns = {"df_test": df_test}
    exec(compile(prefix, "<orig cell 104>", "exec"), ns)
    assert build_ocean_analysis_messages(ns["sample_text"]) == ns["messages"]


def test_inference_prompt_prefix_matches_training_format():
    from src.models.llm import chatml_inference_prompt, format_chatml_examples

    full = format_chatml_examples(["hello"], ["Openness"])[0]
    assert full.startswith(chatml_inference_prompt("hello"))
