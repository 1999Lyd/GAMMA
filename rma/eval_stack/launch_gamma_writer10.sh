#!/bin/bash
# GAMMA WITH THE WRITER (agent-1 LoRA, 1 epoch, loss 0.028) on the FINAL ckpt 79999:
# 10 trials x 26 tasks in three lanes (tasks 1-9, 10-18, 19-26); agent servers on gpu2,
# policy servers gpu2/gpu7/gpu3, clients gpu5/gpu6/gpu5; startups 150 s apart.
S=${GAMMA_WORK}
export A1_CKPT=${GAMMA_DATA}/runs/agent1_rma_v1_lora/v0-20260914-010211/checkpoint-318 WAM_BASE=Qwen/Qwen3.5-9B
AGPU=2 TS_OVERRIDE=1  TE_OVERRIDE=9  SRV_OVERRIDE=${GPU2} CLI_OVERRIDE=${GPU5} APORT_OVERRIDE=8147 PPORT_OVERRIDE=8157 bash $S/chain_rma_gamma_probe.sh 79999 L1  10 writer10 > $S/gamma_writer10_L1.log 2>&1 &
sleep 150
AGPU=2 TS_OVERRIDE=10 TE_OVERRIDE=18 SRV_OVERRIDE=${GPU7} CLI_OVERRIDE=${GPU6} APORT_OVERRIDE=8148 PPORT_OVERRIDE=8158 bash $S/chain_rma_gamma_probe.sh 79999 L2  10 writer10 > $S/gamma_writer10_L2.log 2>&1 &
sleep 150
AGPU=2 TS_OVERRIDE=19 TE_OVERRIDE=26 SRV_OVERRIDE=${GPU3} CLI_OVERRIDE=${GPU5} APORT_OVERRIDE=8149 PPORT_OVERRIDE=8159 bash $S/chain_rma_gamma_probe.sh 79999 L2c 10 writer10 > $S/gamma_writer10_L2c.log 2>&1 &
wait; echo "$(date +%F_%T) gamma writer10 lanes done"
