# Checkpoints

The trained weights are distributed separately from the code as one archive,
`GAMMA_checkpoints.zip`. **The archive will be released publicly upon acceptance**;
the anonymised submission contains the code only, and the numbers in the paper can be
reproduced from the training scripts in this repository (`train/`, `rma/`) in the meantime.
SHA-256 of the archive (11 GB): `c27cd5f2b13019c6582b0b8f249d27c987169fca24a4e235da5ad4450dc4555d`.

Unpack it anywhere and point the environment variables at it (see `env.sh.example`):

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

Sizes: adapters 228 MB total; policy 11 GB.
