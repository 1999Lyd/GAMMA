import os, sys, time
from huggingface_hub import snapshot_download
dst = sys.argv[1]
for rev in ("e645741c2f34f27e1596bfa89e856d6f3560ed90", "main"):
    try:
        p = snapshot_download("huashuolei/PrediMem", revision=rev, allow_patterns=["vla_alltask/*", ".gitattributes"], local_dir=dst, max_workers=8)
        print("DONE revision", rev, "->", p, flush=True); break
    except Exception as e:
        print("revision", rev, "failed:", repr(e)[:300], flush=True)
