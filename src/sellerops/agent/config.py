"""Named model sources. Paths are relative to the project, not the working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SFT_RUN = PROJECT_ROOT / "results" / "sft" / "20260917-234821-full-sft"

# Values are keyword arguments for ModelAdapter. Add new model aliases here.
MODEL_CONFIGS = {
    "qwen-base": {
        "local": False,
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "sellerops-sft": {
        "local": True,
        "model_path": str(SFT_RUN / "final-model"),
        "tokenizer_path": str(SFT_RUN / "final-model"),
    },
}
