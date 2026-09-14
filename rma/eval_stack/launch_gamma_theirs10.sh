#!/bin/bash
# GAMMA harness-only (plan-pinned, proprioceptive timing) driving THEIR published pi0.5, their labels as prompts:
# 10 trials x 26 tasks, two lanes; agents gpu2; policy servers gpu2/gpu7; clients gpu7/gpu5.
S=${GAMMA_WORK}
export CFG_OVERRIDE=rma_pi05_theirs EXP_OVERRIDE=predimem_vla_alltask CKPT_OVERRIDE=${GAMMA_DATA}/ckpts/predimem_hf/vla_alltask
export RMA_PLAN_LABELS=${GAMMA_ROOT}/rma/eval_stack/rma_their_labels.json
AGPU=2 SRV_OVERRIDE=${GPU2} CLI_OVERRIDE=${GPU7} APORT_OVERRIDE=8162 PPORT_OVERRIDE=8174 bash $S/chain_rma_gamma_probe_v2.sh theirs L1 10 theirs10 > $S/gamma_theirs10_L1.log 2>&1 &
sleep 150
AGPU=2 SRV_OVERRIDE=${GPU7} CLI_OVERRIDE=${GPU5} APORT_OVERRIDE=8163 PPORT_OVERRIDE=8175 bash $S/chain_rma_gamma_probe_v2.sh theirs L2 10 theirs10 > $S/gamma_theirs10_L2.log 2>&1 &
wait; echo "$(date +%F_%T) gamma theirs10 lanes done"
