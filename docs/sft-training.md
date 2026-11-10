# Full SFT on a cloud GPU

> Note: `notebooks/sft_drive_paths_fixed.ipynb` mentioned below is not in this repository yet; `notebooks/sft.ipynb` is.

For the user's completed Drive version, use
[`notebooks/sft_drive_paths_fixed.ipynb`](notebooks/sft_drive_paths_fixed.ipynb).
This copy centralizes run paths, reloads the saved model and tokenizer together,
and verifies the active-run ZIP and its Drive copy. Section 3 retains the user's
checkpoint-165 source and `EPOCHS=2`; its guard requires a total epoch target above
the already completed 3 epochs to continue. Adjust the checkpoint path if it was
restored elsewhere. This notebook has passed local path/export checks but has
not been trained in Colab. The original guided notebook below remains unchanged.

`notebooks/sft.ipynb` now provides a full-parameter SFT learning exercise for
Qwen2.5-0.5B-Instruct, with four exercises and notes before every code cell.
It includes target-only label masking, strict development evaluation, a two-step
smoke test, fresh-weight training, and saved-checkpoint evaluation.

Colab uses its managed Python runtime, as authorized on September 6. Run notebook
sections 1 and 1b to audit the GPU and install the pinned training libraries while
preserving the installed CUDA PyTorch build. Restart the Python session after an
installation, then continue. Requirements are in `configs/requirements.colab.txt`.
Local work still uses Conda `llm-post-training`; `configs/environment.cuda.yml`
remains optional for other CUDA hosts. The root environment.yml is unchanged.
Edit upload/output paths before execution. The Colab editor's notebook and the
notebook extracted from the ZIP are separate copies; save your editor copy to Drive.
The notebook intentionally stops at unfinished exercises.

The September 6 training choice is **full SFT**, superseding the earlier LoRA-only
training plan for this experiment. Tool-history replay is now available;
scenario-based evaluation and frozen end-to-end evaluation remain unfinished.
Local SFT result artifacts have not yet been recorded.
