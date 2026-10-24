"""Which word tokenizer does datatrove use for Chinese? MinHash dedup shingles
words, so Chinese without a real segmenter would shingle whole sentences."""

from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer

zh = "被告人以非法占有为目的，虚构投资项目，骗取十一名被害人共计四十八万元。"
en = "The defendant was convicted of fraud under Article 266 of the Criminal Law."
for name in ("english", "chinese", "chinese_simplified", "zho_Hans", "zh"):
    lang = getattr(Languages, name, None)
    if lang is None:
        print(name, "not in Languages")
        continue
    try:
        tok = load_word_tokenizer(lang)
        text = zh if name != "english" else en
        words = tok.word_tokenize(text)
        print(name, repr(lang), type(tok).__name__, len(words), words[:12])
    except Exception as e:
        print(name, repr(lang), "FAILED:", type(e).__name__, str(e)[:300])
