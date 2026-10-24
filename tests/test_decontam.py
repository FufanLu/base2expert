from b2e.data.decontam import FrozenNgramIndex, NgramIndex, units

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


def test_frozen_index_matches_set_index(tmp_path):
    evals = [Q, "被告人以非法占有为目的，虚构投资项目，骗取十一名被害人共计四十八万元。"]
    path = str(tmp_path / "idx.npy")
    n = FrozenNgramIndex.build(evals, path)
    frozen = FrozenNgramIndex(path)
    ref = NgramIndex(13)
    for e in evals:
        ref.add(e)
    assert n == len(ref)
    for doc in [Q + " extra words here", "经审理查明：被告人以非法占有为目的，虚构投资项目，骗取十一名被害人钱款。", "unrelated text " * 20]:
        assert frozen.hits(doc) == ref.hits(doc)


SLOGAN = "全面建设社会主义现代化国家"


def test_common_phrases_are_not_indexed(tmp_path):
    # the slogan appears in three eval items; the unique item text does not
    evals = [f"{SLOGAN}，下列说法正确的是{i}号" for i in "甲乙丙"] + ["被告人以非法占有为目的，虚构投资项目，骗取十一名被害人共计四十八万元。"]
    path = str(tmp_path / "idx.npy")
    FrozenNgramIndex.build(evals, path, max_df=3)
    idx = FrozenNgramIndex(path)
    assert not idx.contaminated_by_runs(f"新闻：我们要{SLOGAN}，推动高质量发展。")
    assert idx.contaminated_by_runs("经审理查明：被告人以非法占有为目的，虚构投资项目，骗取十一名被害人钱款。")


def test_cjk_runs_need_twenty_characters(tmp_path):
    item = "甲公司于二零二三年向乙银行借款五百万元用于购买生产设备并约定分期偿还本息"
    path = str(tmp_path / "idx.npy")
    FrozenNgramIndex.build([item], path)
    idx = FrozenNgramIndex(path)
    assert not idx.contaminated_by_runs("报道称甲公司于二零二三年向乙银行借款之后")       # 14 shared chars
    assert idx.contaminated_by_runs("报道称甲公司于二零二三年向乙银行借款五百万元用于购买设备")  # 22 shared chars


def test_english_runs_keep_thirteen_words(tmp_path):
    path = str(tmp_path / "idx.npy")
    FrozenNgramIndex.build([Q], path)
    idx = FrozenNgramIndex(path)
    assert idx.contaminated_by_runs("Context " + Q)


def test_item_index_counts_coverage_of_the_item(tmp_path):
    from b2e.data.decontam import ItemNgramIndex

    statute = "最高人民法院关于审理民间借贷案件适用法律若干问题的规定"
    item = f"根据{statute}第二十六条，借贷双方约定的利率未超过年利率百分之二十四，出借人请求借款人按照约定的利率支付利息的，人民法院应予支持。下列说法正确的是"
    prefix = str(tmp_path / "items")
    ItemNgramIndex.build([item, "另一道完全无关的题目内容，用来检查编号是否正确对应到各自的评测题上面"], prefix)
    idx = ItemNgramIndex(prefix)
    quoting = f"新闻稿：法院在判决中引用了{statute}，驳回了原告的全部诉讼请求。"
    cov, _ = idx.coverage(quoting)
    assert cov < 0.6 and not idx.contaminated(quoting)
    copied = "练习题：" + item + "（答案见下页）"
    cov, which = idx.coverage(copied)
    assert cov >= 0.9 and which == 0 and idx.contaminated(copied)
