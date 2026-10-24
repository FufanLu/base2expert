from b2e.data.pii import luhn_ok, scrub, valid_cn_id

# 11010519491231002X is the standard example ID from GB 11643 documentation
VALID_ID = "11010519491231002X"


def test_id_checksum():
    assert valid_cn_id(VALID_ID)
    assert not valid_cn_id("110105194912310021")


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")


def test_scrub_replaces_each_kind():
    t = (f"联系人张三，身份证{VALID_ID}，手机13812345678，邮箱zhang.san@example.com，"
         "卡号 4111 1111 1111 1111，服务器 192.168.1.20")
    out, c = scrub(t)
    assert "<ID_NUMBER>" in out and "<PHONE>" in out and "<EMAIL>" in out and "<CARD_NUMBER>" in out and "<IP>" in out
    assert c == {"email": 1, "ip": 1, "mobile": 1, "id": 1, "card": 1}


def test_finance_numbers_survive():
    t = "营业收入为1234567890123456元，同比增长12.4%，报告编号 2025-0001。"
    out, c = scrub(t)
    assert out == t  # 16 digits that fail Luhn stay
    assert sum(c.values()) == 0


def test_mobile_with_country_code():
    out, _ = scrub("call +86 13912345678 now")
    assert out == "call <PHONE> now"
