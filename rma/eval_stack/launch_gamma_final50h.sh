#!/bin/bash
# GAMMA harness-only, FINAL ckpt 79999, 50 trials x 26 tasks in six lanes (paper number).
# Each lane waits until its policy-server GPU is below 95 GB, then starts; 150 s between starts.
S=${GAMMA_WORK}
waitgpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 95000 ]; do sleep 60; done; }
lane() { # TAG TS TE AGPU SRV CLI APORT PPORT
  waitgpu $4
  AGPU=$4 TS_OVERRIDE=$2 TE_OVERRIDE=$3 SRV_OVERRIDE=$5 CLI_OVERRIDE=$6 APORT_OVERRIDE=$7 PPORT_OVERRIDE=$8 \
    bash $S/chain_rma_gamma_probe_v2.sh 79999 L1 50 $1 > $S/gamma_$1.log 2>&1
}
lane final50h_A 1 5   4 ${GPU4} ${GPU5} 8201 8211 & sleep 150
lane final50h_B 6 9   4 ${GPU4} ${GPU6} 8202 8212 & sleep 150
lane final50h_C 10 13 3 ${GPU3} ${GPU5} 8203 8213 & sleep 150
lane final50h_D 14 18 5 ${GPU5} ${GPU7} 8204 8214 & sleep 150
lane final50h_E 19 22 6 ${GPU6} ${GPU4} 8205 8215 & sleep 150
lane final50h_F 23 26 7 ${GPU7} ${GPU4} 8206 8216 & wait
echo "$(date +%F_%T) gamma final50h lanes done"
