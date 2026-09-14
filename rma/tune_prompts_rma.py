#!/usr/bin/env python
"""SAM-3 phrase tuning for RMA against oracle-plan coordinates.
For every pickable object (plan kind=pick: median = its initial position) and
every receptacle (plan kind=stage: median = ee position at the stage predicate,
so a wider radius), score candidate phrases on frame 0 of seeds 100-104 of each
task that involves it: best score within the radius, hit rate at 0.15, and
false positives (>=0.15 outside the radius). Writes tune_prompts_rma.json."""
import os, json, re, math, glob, collections, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
os.environ.setdefault("HF_HOME", os.path.expandvars("${HF_HOME}"))
HERE = os.path.dirname(os.path.abspath(__file__)); F = os.path.expandvars("${GAMMA_DATA}/data/rma_sft_v1/frames")
PLANS = json.load(open(os.path.expandvars("${GAMMA_ROOT}/rma/eval_stack/rma_oracle_plans.json")))
CAND = {
 "cookies": ["a small yellow box on the table", "a yellow box", "a small yellow package", "a yellow snack box", "a yellow box with a red lid"],
 "tomato_sauce": ["a red bottle on the table", "a small red bottle", "a red bottle", "a red can", "a red condiment bottle"],
 "butter": ["a small white and blue carton on the table", "a small white box", "a white and blue box", "a small blue and white carton", "a butter box"],
 "popcorn": ["an orange snack box with a white label", "an orange box", "a small orange box on the table", "an orange package", "a popcorn box"],
 "cream_cheese": ["a small white carton on the table", "a white box", "a small white package", "a white carton", "a cream cheese box"],
 "chocolate_pudding": ["a dark brown box on the table", "a brown box", "a small brown box", "a dark box on the table", "a chocolate box"],
 "milk": ["a white milk carton", "a milk carton", "a white carton", "a tall white box", "a milk box"],
 "wine_bottle": ["a dark green bottle", "a wine bottle", "a green bottle on the table", "a dark bottle", "a bottle of wine"],
 "orange_juice": ["an orange juice bottle", "an orange bottle", "a juice bottle", "an orange carton", "a bottle of orange juice"],
 # receptacles (radius 45)
 "basket": ["a wooden basket with slatted sides", "a basket", "a wicker basket on the table", "a wooden basket"],
 "drawer": ["a dark wooden cabinet standing on the table", "a cabinet with drawers", "a wooden cabinet", "a chest of drawers"],
 "microwave": ["a black microwave with a door", "a microwave", "a microwave oven", "a black microwave oven"],
 "frypan": ["a black frying pan", "a frying pan", "a pan on the table", "a black pan"],
 "mug": ["a mug on the table", "a red mug", "a coffee mug", "a cup"],
 "drainer": ["a white dish rack with slats", "a dish rack", "a bowl drainer", "a white dish drainer"],
 "plate": ["a plate on the table", "a white plate", "a plate", "a dish"],
 "cabinet": ["a dark wooden cabinet standing on the table", "a wooden cabinet", "a cabinet", "a cupboard"],
}
RECEP = {"basket", "drawer", "microwave", "frypan", "mug", "drainer", "plate", "cabinet"}
# (object, task, median) samples
samples = collections.defaultdict(list)
for tid, plan in PLANS.items():
    for s in plan:
        if s["kind"] == "pick" and s.get("obj"):
            samples[s["obj"].rsplit("_1", 1)[0]].append((int(tid), tuple(s["median"])))
        else:
            m = re.search(r"\b(basket|drawer|microwave|frypan|mug|drainer|plate|cabinet)\b", s["instr"])
            if m and s.get("median"): samples[m.group(1)].append((int(tid), tuple(s["median"])))
DEV = "cuda:0"; proc = Sam3Processor.from_pretrained("facebook/sam3"); model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(DEV).eval()
report = {}
with torch.no_grad():
    for obj, phrases in CAND.items():
        pairs = {}
        for tid, med in samples.get(obj, []): pairs.setdefault(tid, med)      # one median per task
        if not pairs: print(f"[tune] {obj}: no plan samples"); continue
        R = 45 if obj in RECEP else 22
        frames = [(tid, med, f"{F}/task{tid}_seed{s}/f0.jpg") for tid, med in pairs.items() for s in range(100, 105) if os.path.exists(f"{F}/task{tid}_seed{s}/f0.jpg")]
        print(f"--- {obj}: tasks {sorted(pairs)} radius {R} frames {len(frames)}")
        report[obj] = {}
        for p in phrases:
            best = []; hits = 0; fps = 0
            for tid, med, fp in frames:
                pil = Image.open(fp).convert("RGB")
                inp = proc(images=[pil], text=[p], return_tensors="pt").to(DEV); out = model(**inp)
                r = proc.post_process_instance_segmentation(out, threshold=0.05, mask_threshold=0.5, target_sizes=inp["original_sizes"].tolist())[0]
                b = 0.0
                for box, sc in zip(r["boxes"].tolist(), r["scores"].tolist()):
                    row, col = (box[1] + box[3]) / 2, (box[0] + box[2]) / 2
                    if math.hypot(row - med[0], col - med[1]) <= R: b = max(b, sc)
                    elif sc >= 0.15: fps += 1
                best.append(b); hits += b >= 0.15
            mb = sum(best) / len(best); hr = hits / len(frames); fpr = fps / len(frames)
            report[obj][p] = {"mean_best": round(mb, 3), "hit@0.15": round(hr, 2), "fp_per_frame": round(fpr, 2)}
            print(f"  {p!r:46s} mean_best {mb:.2f}  hit@0.15 {hr:.2f}  fp/frame {fpr:.2f}")
json.dump(report, open(f"{HERE}/tune_prompts_rma.json", "w"), indent=1); print("[tune] written tune_prompts_rma.json")
