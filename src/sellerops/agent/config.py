"""Named model sources. Paths resolve under SELLEROPS_RESULTS_DIR (default: <project>/results)."""

from ..settings import get_settings

SFT_RUN = get_settings().results_dir / "sft" / "20260917-234821-full-sft"

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
