# Put on PYTHONPATH by MOPD jobs, so that every Python process (the Ray TaskRunner and the actor
# worker among them) patches verl as it imports it:
# - B2E_TEACHERS_SHARE_GPU=1: several teachers share one GPU (b2e.distill.share_gpu);
# - B2E_MOPD_BALANCE=share|gap: Open-MOPD's per-domain loss weights (b2e.distill.balance).
import os
import sys

if os.environ.get("B2E_TEACHERS_SHARE_GPU") == "1":
    try:
        from b2e.distill.share_gpu import install

        install()
    except Exception as e:  # a broken patch must be loud, not a silent fall back to one GPU per teacher
        print(f"b2e sitecustomize: could not install the teacher-sharing patch: {e!r}", file=sys.stderr, flush=True)
        raise

if os.environ.get("B2E_MOPD_BALANCE"):
    try:
        from b2e.distill.balance import install as install_balance

        install_balance()
    except Exception as e:  # nor a silent fall back to unweighted MOPD
        print(f"b2e sitecustomize: could not install the loss-weight patch: {e!r}", file=sys.stderr, flush=True)
        raise
