from b2e.data.decontam import NgramIndex, units

Q = "What was the percentage change in net revenue from 2018 to 2019 for the consumer segment?"


def test_units_mixed_text():
    assert units("净利润为15.3亿元, Net income was $1,234") == [
        "净", "利", "润", "为", "15.3", "亿", "元", "net", "income", "was", "1", "234"]


def test_exact_copy_is_found():
    idx = NgramIndex(13)
    idx.add(Q)
    assert idx.contaminated("Context... " + Q + " Answer: 12%")


def test_formatting_does_not_hide_a_copy():
    idx = NgramIndex(13)
    idx.add(Q)
    assert idx.contaminated(Q.upper().replace(" ", "  ").replace("?", "!"))


def test_short_overlap_is_not_flagged():
    idx = NgramIndex(13)
    idx.add(Q)
    # shares 8 consecutive words only
    assert not idx.contaminated("the percentage change in net revenue from 2018 was large for everyone involved here")


def test_chinese_character_ngrams():
    idx = NgramIndex(13)
    idx.add("被告人以非法占有为目的，虚构投资项目，骗取十一名被害人共计四十八万元。")
    assert idx.contaminated("经审理查明：被告人以非法占有为目的，虚构投资项目，骗取十一名被害人钱款。")
    assert not idx.contaminated("被告人以非法占有为目的实施了盗窃。")


def test_text_shorter_than_n():
    idx = NgramIndex(13)
    idx.add("short text")
    assert len(idx) == 0
    assert idx.hits("short text") == 0
