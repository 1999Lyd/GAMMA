#!/bin/bash
# When the r6 detection cache is complete: build the full RMA corpus (into
# rma_sft_v1/ so frames resolve), convert to ms-swift, launch the 9B writer SFT.
set -u
S=${GAMMA_WORK}; PR=${GAMMA_ROOT}/rma
F=${GAMMA_DATA}/data/rma_sft_v1; PY=${ROBOMME_ROOT}/.venv/bin/python
echo "$(date +%F_%T) waiting for the r6 cache (workers exit + >= 1000 files)"
until ! pgrep -f "sam3_precompute_rma.py" > /dev/null && [ $(ls $F/dets_sam3/*.json 2>/dev/null | wc -l) -ge 1000 ]; do sleep 300; done
n=$(ls $F/dets_sam3/*.json | wc -l); echo "$(date +%F_%T) cache complete: $n files"
if [ $n -lt 1040 ]; then echo "$(date +%F_%T) fill pass for the $((1040-n)) missing episodes"; CUDA_VISIBLE_DEVICES=3 ${MSSWIFT_PY} -u $PR/sam3_precompute_rma.py > $S/sam3_rma_r6_fill.log 2>&1; fi
cd $PR && OUT=$F SEQUENTIAL=1 $PY -u build_rma_corpus.py > $S/build_rma_corpus.log 2>&1; echo "$(date +%F_%T) corpus built: $(tail -n 1 $S/build_rma_corpus.log | cut -c1-120)"
grep -q '"episodes"' $S/build_rma_corpus.log || { echo "ABORT: corpus build failed"; exit 1; }
SRC=$F $PY $PR/make_agent1_swift_rma.py | tee -a $S/build_rma_corpus.log
setsid nohup bash $S/train_agent1_rma9b.sh 3 > $S/train_agent1_rma9b.log 2>&1 < /dev/null &
echo "$(date +%F_%T) writer SFT launched (gpu3), pid $!"
