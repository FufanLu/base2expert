import os
from pathlib import Path

# nltk data for the instruction-following verifiers, downloaded into the repo
# (gitignored) by: python -c "import nltk; nltk.download('punkt_tab', download_dir='.nltk_data')"
_data = Path(__file__).resolve().parent.parent / ".nltk_data"
if _data.is_dir():
    os.environ.setdefault("NLTK_DATA", str(_data))
