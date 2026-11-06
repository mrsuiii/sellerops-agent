from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelAdapter:
    def __init__(
        self,
        local: bool,
        model_name: str | None = None,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
    ):
        self.local = local

        if local:
            if not model_path:
                raise ValueError(
                    "model_path must be provided when local=True"
                )
            if not tokenizer_path:
                raise ValueError(
                    "tokenizer_path must be provided when local=True"
                )

            model_dir = Path(model_path)
            tokenizer_dir = Path(tokenizer_path)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
                local_files_only=True,
            )
            self.tokenizer.chat_template = (
                tokenizer_dir / "chat_template.jinja"
            ).read_text(encoding="utf-8")

            model_source = model_dir

        else:
            if not model_name:
                raise ValueError(
                    "model_name must be provided when local=False"
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_source = model_name


        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=local,
        )
        self.model.eval()
