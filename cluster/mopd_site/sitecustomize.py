# Put on PYTHONPATH by MOPD jobs: with B2E_TEACHERS_SHARE_GPU=1 every Python process (the Ray
# TaskRunner among them) patches verl so several teachers share one GPU (b2e.distill.share_gpu).
import os
import sys

if os.environ.get("B2E_TEACHERS_SHARE_GPU") == "1":
    try:
        from b2e.distill.share_gpu import install

        install()
    except Exception as e:  # a broken patch must be loud, not a silent fall back to one GPU per teacher
        print(f"b2e sitecustomize: could not install the teacher-sharing patch: {e!r}", file=sys.stderr, flush=True)
        raise
