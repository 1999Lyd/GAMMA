# Checkpoints

The trained weights are distributed separately from the code as two archives in the
Google Drive folder https://drive.google.com/drive/folders/15UWryg_Q7E0GqcG85X8RqZprk1iSMMhq:

| archive | contents | size | SHA-256 |
|---|---|---|---|
| `GAMMA_checkpoints.zip` | RoboMME writer and reasoner adapters (9B and 0.8B) and the RoboMME executor | 11.7 GB | `c27cd5f2b13019c6582b0b8f249d27c987169fca24a4e235da5ad4450dc4555d` |
| `GAMMA_checkpoints_rma.zip` | RoboMemArena writer adapter and executor | 11.7 GB | `01364694a64dcac1ee2ce5c754e782c8fadc9ff0931bb28d722b1d0c97222f15` |

Both archives unpack into the same `GAMMA_checkpoints/` tree; the RoboMME results need only
the first. The folder can be fetched from the command line with
`pip install gdown && gdown --folder https://drive.google.com/drive/folders/15UWryg_Q7E0GqcG85X8RqZprk1iSMMhq`.

Unpack them anywhere and point the environment variables at it (see `env.sh.example`):

```
GAMMA_checkpoints/
  adapters/
    agent1_v17_final/      writer  (Agent-1) LoRA on Qwen3.5-9B   -- the paper's main-table checkpoint
    agent2_v17_final/      reasoner (Agent-2) LoRA on Qwen3.5-9B  -- trained on writer rollouts (bank-only)
    agent1_0p8b_final/     writer  LoRA on Qwen3.5-0.8B           -- the 0.8B arm of the main and ablation tables
    agent2_0p8b_final/     reasoner LoRA on Qwen3.5-0.8B
    agent1_rma_v1/         writer  LoRA on Qwen3.5-9B for RoboMemArena (1 epoch on the RMA corpus)
  policy/
    pi05_subgoal_conditioned_79999/   the fixed subgoal-conditioned pi0.5 executor (openpi/JAX checkpoint,
                                      RoboMME "GroundSG" recipe, step 79999). Shared by every configuration
                                      in the paper and never modified.
    rma_pi05_sgprompt_79999/          the RoboMemArena executor: the benchmark authors' pi0.5 recipe
                                      (subtask text as prompt, 40k updates, batch 128), openpi/JAX checkpoint.
```

Each adapter directory holds `adapter_config.json`, `adapter_model.safetensors`,
`additional_config.json`, `args.json` (the exact ms-swift arguments), `trainer_state.json`
and the ms-swift `README.md`; optimizer and RNG states were dropped. The adapters load with
`peft.PeftModel.from_pretrained(base, path)` on top of the Hugging Face base model named in
`args.json` (`Qwen/Qwen3.5-9B` or `Qwen/Qwen3.5-0.8B`), which is what `serve/wam_agent_server.py`
does when `A1_CKPT` / `A2_CKPT` point at them.

Base models and the detector are public and are downloaded on first use:
`Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-0.8B`, `facebook/sam3` (SAM-3 via `transformers`).

Sizes: adapters 412 MB in total; each executor 11 GB.
