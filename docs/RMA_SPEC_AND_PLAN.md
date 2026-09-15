# RoboMemArena (RMA) — strict config spec + GAMMA corpus implementation plan
Sources are labeled: [paper]=arXiv:2605.10921, [repo]=rma_eval_repo,
[stack]=rma_eval_stack (installed+smoke-tested, NO eval run yet),
[measured]=verified on local data 2026-09-08. Written 2026-09-08.

## A. Benchmark identity
- RoboMemArena, OpenHelix-Team. Paper arXiv:2605.10921 [verified]. Code
  github.com/OpenHelix-Team/RoboMemArena; local clone
  ${RMA_BENCH_ROOT}/.. (evaluation_benchmark:
  LIBERO-fork, robosuite==1.4.1, mujoco==2.3.7, bddl==1.0.1) [repo].
- Their baseline PrediMem: dual-system VLA (VLM planner + memory bank with
  recent/keyframe buffers + predictive-coding head, train-time only) [paper].
- Real-suite headline (10 rollouts/task): pi0.5 21.5 TSR / 38.7 CSR; MemER
  27.3/49.1; PrediMem 38.5/55.2 [paper]. Metrics: TSR + stage-wise CSR
  (3-9 verification stages/task) [paper].

## B. Tasks and data [measured + README]
- 26 tasks, 4 families: Sequence {1,2,3,22}; Occlusion {4,5,11,12,13,14,17,
  20,21,23,24}; Counting {6,7,8,9,10,15,16}; Transferring {18,19,25,26}.
  Avg episode length 472-1835 steps (README per-task table).
- Local data: ${GAMMA_DATA}/data/RoboMemArena/
  {family}/{id}_{name}_dataset/subtask_data/*.hdf5. ~100 seeds per task
  starting at seed100; 400-900 files/task = (#subtasks x #seeds).
- HDF5 layout [measured]: data/demo_0 with actions (T,7) f64; obs:
  agentview_rgb (T,256,256,3) u8, eye_in_hand_rgb (T,256,256,3) u8,
  ee_pos(3), ee_ori(3), ee_states(6), gripper_states(2), joint_states(7);
  rewards (all 0 observed), dones, keyframe_indices (1 int per subtask
  file; = gripper-state transitions U kinematic inflections [paper]).
- Filename contract [measured]: {subtask_words}_{planIndex}_seed{S}_task{T}
  .hdf5; planIndex is the execution order (task10: pick_0 -> pour_1st_1 ->
  pour_2nd_2 -> place_3).
- EPISODE RECONSTRUCTION [measured, task10 seed100]: same-seed subtask
  files chain continuously — consecutive end/start ee_pos gaps 0.0002-
  0.0019 m. Concatenating same-seed files in planIndex order = the full
  episode WITH exact subtask boundaries (structural ground truth, no
  timing noise). Full-scale verification is plan step V1.
- Instruction: official builder derives it from filename words [repo:
  RoboMemArena_dataset_builder.py]; task-level instruction from README
  descriptions.

## C. Fixed evaluation protocol [stack — "PROTOCOL IS FIXED", do not edit]
- Eval layouts: seeds 50..99 ONLY; 50 trials/task; --replan-steps 10;
  --max-steps 2500; --num-steps-wait 10; --resize-size 256; extra-pour
  rejection ON. Runner: rma_eval_stack/run_rma_eval.sh (policy server on
  training venv + LIBERO-fork client on dedicated py3.10 venv).
- Policy: mme_vla_suite RMA checkpoints exist (runs/ckpts/rma_fs_modul,
  rma_dualgate_fs_memonly; assets rma_base). Action chunk = 20 steps,
  first 10 executed per replan [stack README].
- => Demo seeds (>=100) never appear at eval; no train/eval leakage by
  construction.
- OPEN ITEM: no subgoal-conditioned (GroundSG-style) RMA policy checkpoint
  found. Closed-loop GAMMA-on-RMA requires training one (same recipe as
  RoboMME GroundSG: finetune policy to condition on oracle subgoal text).
  Flagged for user decision; corpus below is policy-independent.

## D. GAMMA-on-RMA geometry (decisions)
- Tick H = 10 control steps (= the fixed replan cadence). One
  writer+reasoner call per tick.
- Frames per window n = 3, stride 5, boundaries included (t, t+5, t+10).
  Matches RoboMME's frame density (5/16 = 0.31/step vs 3/10 = 0.30/step)
  at equal VLM cost per step. Keyframe_indices provide extra within-
  subtask evidence anchors.
- ~105 ticks/episode avg; 26 tasks x ~100 seeds => ~2,600 episodes,
  ~270k writer records (4.9x RoboMME's 55k). v1 uses all seeds; subsample
  only if training cost demands (user decision).
- No demonstration phase in RMA episodes: the demo-record /reset replay
  and "hold and wait for the demo to complete" convention are dropped;
  phase = exec from tick 0. Initial-scene line convention kept.
- Camera: agentview_rgb only for the writer (matches RoboMME single-view
  serve); eye_in_hand reserved.

## E. Corpus semantics (oracle stream + bank lines)
- Reasoner target at tick k: humanized subtask name of the segment in
  progress at tick k+1, grounded with the target object's detected
  coordinate; after final boundary: "all tasks completed; remain static";
  coordinate not yet knowable => "hold and observe the scene" (same
  conventions as v17 minus the demo hold).
- Writer targets: "completed: <subtask>" at the tick whose window contains
  the file boundary (structural GT); initial-scene line at tick 0 from
  frame-0 detections; family augmentations, all evidence-gated:
  * Occlusion: "placed X inside <drawer/microwave> at <x,y>" +
    "closed <container> — X is now hidden inside" (from subtask names +
    open/close boundaries; identity carried by the line).
  * Counting: pour lines numbered "poured ... (1st/2nd of 2)" (from
    _1st/_2nd subtask names).
  * Transferring: "moved X from <src> to <dst>" chains (cabinet1->2,
    plate1->2).
  * Sequence: order index on placement lines.
- Evidence gates (v17 contracts kept): a line only at the window showing
  its evidence (boundary/keyframe within window; displacement for moves);
  knowability: every coordinate must appear in the window's detections or
  prior bank lines, else demote to coordinate-free.

## F. Implementation plan — mapped 1:1 onto the v17 pipeline
Pipeline provenance: the v17 scripts were developed in a working
directory; step 0 copies them into the repository.
0. cp -> ${GAMMA_ROOT}/serve/ : sam3_precompute.py,
   sam3_prompts.json(+round2), gen_wam_sft_v12.py, make_agent1_swift.py,
   make_agent2_swift.py, rollout_agent1.py, chain_rollout_v9.sh,
   wam_agent_server.py, rs_dir_algo.py. RMA forks go to
   ${GAMMA_ROOT}/rma/.
1. extract_rma_frames.py: hdf5 -> jpg on the stride-5 grid (+frame 0) per
   reconstructed episode; per-episode manifest.json (subtask boundaries in
   concatenated step coordinates, keyframes, lengths, instruction).
   Output ${GAMMA_DATA}/data/rma_sft_v1/
   {frames,manifests}/task{T}_seed{S}/... (~550k jpgs, ~10 GB). CPU-bound,
   parallel over tasks, detached.
   V1 verification (in the same pass): chaining continuity for EVERY
   (task,seed): max consecutive ee gap <= 5 mm, else quarantine episode;
   report per-task seed inventory + missing/extra planIndex.
2. SAM-3 RMA prompt tuning (fork sam3_prompts.json -> prompts_rma.json):
   concepts for butter, chocolate, cookies, cream cheese, popcorn,
   pudding, milk, tomato sauce, orange juice, wine bottle, mug, frypan,
   bowl drainer, basket, drawers (top/middle/bottom + open state),
   microwave (+door), cabinets, plates, robot gripper. Probe on a 30-frame
   stratified sample; render overlays; iterate phrases (v8 lesson:
   phrase-tuned concepts, one label per site). USER REVIEWS OVERLAYS
   before full precompute (no-oversampling/minimal-iterations rule).
3. sam3_precompute_rma.py (fork): source = step-1 frame grid; same
   lock-file multi-GPU sharding; THR from step-2 tuning. Output
   rma_sft_v1/dets_sam3/task{T}_seed{S}.json. GPUs: share 2/3 with the
   0.8B run now, add 4-7 after seed-9 closes.
4. Detection verification (gate before corpus): per-class presence rate on
   frames where the class is scene-guaranteed (from task object lists);
   accept >=0.95 for movable food items and receptacles the subgoals
   reference; document noisy classes (v8 precedent: noise is fine if
   train==serve). Spot-check 20 overlays/family by eye.
5. gen_wam_sft_rma.py (fork of gen_wam_sft_v12.py): implements sections
   D+E; emits agent1_{train,val}.jsonl + agent2_{train,val}.jsonl.
   Split: val = seed % 10 == 0.
6. make_agent{1,2}_swift.py path-swap -> agent{1,2}_swift_{train,val}
   .jsonl (system prompts updated: 10-step window, 3 frames, no demo
   phase).
7. Corpus verification suite (gate before training): record counts per
   task/family; 20-episode boundary audit (completed-line window contains
   the boundary; keyframe within +-1 window); knowability rate; bank
   length distribution; swift token-length p95 <= 3200; leakage check
   (assert no seed < 100); 3 annotated episode strips per family rendered
   for user review.
8. (next round, after user review) Stage-1 writer LoRA (v17 recipe, batch
   64) -> sharded rollout_agent1 -> stage-2 reasoner on predicted banks;
   serve fork of wam_agent_server.py (H=10, no /reset demo replay, RMA
   verifier instantiations: pour-count gate via keyframe/occlusion
   percepts, drawer/microwave occlusion state authority); closed-loop via
   rma_eval_stack once a subgoal-conditioned RMA policy exists (see C).

## G. Current compute map (2026-09-08)
- GPUs 0/1: other user (full). 2/3: Qwen3.5-0.8B writer LoRA (running) +
  post-exit holders. 4-7: RoboMME seed-9 eval (lanes H/I). RMA frame
  extraction is CPU; SAM-3 probe fits beside the 0.8B run on GPU 3.

## G. Serve-side harness plan (v18 contract -> RMA), written 2026-09-08
Architecture: fork wam_agent_server.py -> rma_agent_server.py. /reset =
initial-scene line from frame-0 detections (NO demo replay); /tick every
10 executed control steps with 3 frames (stride 5) + SAM-3 dets + proprio
(ee_pos, ee_ori, gripper_states — licit observation: the policy consumes
it and the benchmark defines keyframes from it). Same Algorithm-1 loop:
typed claims, terminal verdicts clear pending, deferred progress pins the
plan index. Env kill-switches (WAM_NO_* pattern) from day one for the
per-verdict ablation.

Progress-claim verifiers (per completion type):
- pick:  target class detection displaced from rest > thr AND gripper
         closed; else DEFER.
- place: class detection appears within radius of the target site (or
         vanishes into container) AND gripper open; else DEFER.
- pour:  tilt cycle in ee_ori (roll > theta for >=k steps, then upright)
         while ee within radius of the pour target; the gate on the LAST
         counted pour is load-bearing (premature 2nd-pour-done ->
         DEFER -> policy re-pours; RoboMME BUS analog).
- open/close (drawer, microwave): drawer-front/handle or door detection
         displacement + interior visibility change; else DEFER.

Event-claim verifiers (bank admission):
- drawer observation lines (t4/5): admitted only while the drawer's OPEN
  state is verified in the same window; "empty" (absence claim) REJECTED
  unless open verified across >=2 frames. Summary line after final close
  must equal the deterministic fold of the three observation lines ->
  CORRECT to fold (container-fold authority analog).
- placement lines with interior coordinates: knowability — coordinate
  must match a window detection in the container region, else demote to
  coordinate-free / CORRECT to nearest class-compatible detection.
- numbered pour lines: number must equal fold(prior pour lines)+1 ->
  CORRECT to fold (count authority = bank).
- consecutive duplicates -> REJECT (writer stutter, unchanged from v17).

Grounding-claim verifiers (subgoal coordinates):
- default: stale-coordinate re-snap to nearest class-compatible current
  detection (v17 guard).
- EXCEPTION microwave recall (t20/21/23/24): "place B where A was" —
  coordinate authority is A's admitted placement line; NO re-snap (the
  site may be occluded); CORRECT drifted citations to A's line.
- t4/5 recalled drawer: WHICH drawer = fold of observation lines
  (CORRECT mismatches); its coordinate = the drawer front's current
  detection (visible, normal re-snap).

Extractor-owned claims: NONE pre-planned. RMA has no motion percept
measured at chance yet (pour tilt spans several window frames). Follow
v17 discipline: keep writer-owned + verifier-gated, delegate only if a
closed-loop round measures the writer at chance.

Client/policy: adapt the eval client onto rma_eval_stack (protocol fixed:
seeds 50-99, 50 trials, replan 10, 20-step chunks / 10 executed). OPEN
ITEM unchanged: one GroundSG-style subgoal-conditioned policy finetune
needed for closed loop (oracle stream from our corpus + hdf5 actions).
