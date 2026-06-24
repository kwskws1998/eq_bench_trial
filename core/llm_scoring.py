from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeneratorBundle:
    tokenizer: object
    model: object
    device: str


def load_generator(model_name: str, device: str = "auto", dtype: str = "auto", cache_dir: str | None = None) -> GeneratorBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .encoder import resolve_torch_device

    resolved_device = resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = None
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32
    model_kwargs = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs)
    model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return GeneratorBundle(tokenizer=tokenizer, model=model, device=str(resolved_device))


def score_answer_options(
    bundle: GeneratorBundle,
    prompt_prefix: str,
    choices: list[str],
    max_length: int = 4096,
    normalization: str = "mean",
) -> np.ndarray:
    import torch

    if not choices:
        raise ValueError("choices must not be empty.")
    tokenizer = bundle.tokenizer
    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=True)["input_ids"]
    choice_ids = [tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in choices]
    max_choice_len = max(len(ids) for ids in choice_ids)
    if max_choice_len >= max_length:
        raise ValueError("A choice is longer than max_length.")
    max_prefix_len = max_length - max_choice_len
    if len(prefix_ids) > max_prefix_len:
        prefix_ids = prefix_ids[-max_prefix_len:]
    rows = [prefix_ids + ids for ids in choice_ids]
    row_lengths = [len(row) for row in rows]
    padded = [row + [tokenizer.pad_token_id] * (max(row_lengths) - len(row)) for row in rows]
    attention = [[1] * len(row) + [0] * (max(row_lengths) - len(row)) for row in rows]
    input_ids = torch.tensor(padded, dtype=torch.long, device=bundle.device)
    attention_mask = torch.tensor(attention, dtype=torch.long, device=bundle.device)
    with torch.inference_mode():
        logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    valid = torch.zeros_like(token_log_probs)
    start = max(len(prefix_ids) - 1, 0)
    for row_index, ids in enumerate(choice_ids):
        valid[row_index, start : start + len(ids)] = 1.0
    scores = (token_log_probs * valid).sum(dim=1)
    if normalization == "mean":
        scores = scores / valid.sum(dim=1).clamp(min=1.0)
    elif normalization != "sum":
        raise ValueError(f"Unsupported normalization: {normalization}")
    return scores.float().detach().cpu().numpy().astype(np.float64)


def generate_text(
    bundle: GeneratorBundle,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    max_length: int = 4096,
) -> str:
    import torch

    encoded = bundle.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    do_sample = temperature > 0.0
    with torch.inference_mode():
        output = bundle.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=bundle.tokenizer.pad_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
