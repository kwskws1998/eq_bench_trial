from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np


SKBOY_EMOTION_ET_2ND_REPO_ID = "skboy/emotion_et_2nd_model"
SKBOY_EMOTION_ET_2ND_SUBFOLDER = "hf_emotion_et_aug_lr2e-5_len256_seed123"
SKBOY_EMOTION_ET_2ND_WEIGHTS = "et_predictor2_iitb_sa1_sa2_lr2e5_len256_seed123.safetensors"
FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]


@dataclass(frozen=True)
class PredictedTRTWord:
    word: str
    trt: float
    features: dict[str, float]


class TRTPredictor(Protocol):
    def predict_words(self, text: str) -> list[PredictedTRTWord]:
        ...


@dataclass(frozen=True)
class HFArtifactLocation:
    weights_filename: str
    subfolder: str | None


def resolve_hf_artifact_location(
    weights_filename: str,
    subfolder: str | None = None,
) -> HFArtifactLocation:
    weights_path = PurePosixPath(weights_filename)
    if weights_path.parent != PurePosixPath("."):
        inferred = str(weights_path.parent)
        if subfolder is not None and subfolder != inferred:
            raise ValueError(
                "predicted TRT weights include a subfolder that conflicts with "
                "--predicted-trt-subfolder."
            )
        subfolder = inferred
        weights_filename = weights_path.name
    return HFArtifactLocation(weights_filename=weights_filename, subfolder=subfolder)


class SkboyTRTPredictor:
    def __init__(
        self,
        repo_id: str = SKBOY_EMOTION_ET_2ND_REPO_ID,
        weights_filename: str = SKBOY_EMOTION_ET_2ND_WEIGHTS,
        subfolder: str | None = SKBOY_EMOTION_ET_2ND_SUBFOLDER,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        from huggingface_hub import hf_hub_download, snapshot_download

        location = resolve_hf_artifact_location(weights_filename, subfolder)
        download_kwargs = {
            "repo_id": repo_id,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)
        if location.subfolder is not None:
            download_kwargs["subfolder"] = location.subfolder
        model_py = hf_hub_download(filename="model.py", **download_kwargs)
        module = load_module(Path(model_py))
        if not hasattr(module, "load_et_predictor") or not hasattr(module, "predict_word_features"):
            raise ImportError(
                "Unsupported emotion ET export. Expected load_et_predictor and "
                "predict_word_features in model.py."
            )
        patterns = ["*"] if location.subfolder is None else [f"{location.subfolder}/*"]
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
            allow_patterns=patterns,
        )
        model_dir = Path(snapshot_dir)
        if location.subfolder is not None:
            model_dir = model_dir / location.subfolder
        self.model, self.tokenizer = module.load_et_predictor(
            model_dir,
            weight_name=location.weights_filename,
        )
        self.predict_word_features_fn = module.predict_word_features

    def predict_words(self, text: str) -> list[PredictedTRTWord]:
        words, features = self.predict_word_features_fn(text, self.model, self.tokenizer)
        feature_array = np.asarray(features, dtype=np.float64)
        if feature_array.ndim != 2 or feature_array.shape[1] < len(FEATURE_NAMES):
            raise ValueError("ET predictor returned an unexpected feature matrix shape.")
        if len(words) != feature_array.shape[0]:
            raise ValueError("ET predictor returned different word and feature counts.")
        trt_index = FEATURE_NAMES.index("TRT")
        rows: list[PredictedTRTWord] = []
        for word, feature_row in zip(words, feature_array):
            feature_dict = {
                name: float(feature_row[index])
                for index, name in enumerate(FEATURE_NAMES)
            }
            rows.append(
                PredictedTRTWord(
                    word=str(word),
                    trt=float(feature_row[trt_index]),
                    features=feature_dict,
                )
            )
        if not rows:
            raise ValueError("ET predictor returned no words.")
        return rows


class HeuristicTRTPredictor:
    def predict_words(self, text: str) -> list[PredictedTRTWord]:
        rows: list[PredictedTRTWord] = []
        for raw_word in re.findall(r"\S+", text):
            cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", raw_word)
            score = max(0.0, len(cleaned) / 10.0)
            rows.append(
                PredictedTRTWord(
                    word=raw_word,
                    trt=score,
                    features={
                        "nFix": 1.0 + score,
                        "FFD": score,
                        "GPT": score,
                        "TRT": score,
                        "fixProp": min(1.0, score),
                    },
                )
            )
        if not rows:
            raise ValueError("Cannot predict TRT for empty text.")
        return rows


def load_trt_predictor(
    backend: str,
    repo_id: str = SKBOY_EMOTION_ET_2ND_REPO_ID,
    weights_filename: str = SKBOY_EMOTION_ET_2ND_WEIGHTS,
    subfolder: str | None = SKBOY_EMOTION_ET_2ND_SUBFOLDER,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> TRTPredictor:
    if backend == "skboy":
        return SkboyTRTPredictor(
            repo_id=repo_id,
            weights_filename=weights_filename,
            subfolder=subfolder,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if backend == "heuristic":
        return HeuristicTRTPredictor()
    raise ValueError(f"Unsupported predicted TRT backend: {backend}")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("emotion_et_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ET predictor module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
