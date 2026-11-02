import pandas as pd
import pytest

from b2e.train.pref_train import (
    LABEL_PAD,
    bool_found,
    check_kl_rows,
    check_rows,
    expected,
    kl_partner,
    load_pairs,
    longest,
    unpaired,
)


class CharTok:
    """One id per character; <|im_end|> is the eos."""

    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


TOK = CharTok()
P, C, R = "Q?\n<think>\n", "yes</think>\n\nA", "no</think>\n\nB"
SOURCE = [{"prompt": P, "chosen": C, "rejected": R}]


def ids(s):
    return [ord(c) for c in s]


# three prompts with two pairs each, as when S-fin got 2 or more of 8 right and wrong
PAIRS = [{"prompt": f"Q{i // 2}?\n<think>\n", "chosen": f"c{i}</think>\n\nA", "rejected": f"r{i}</think>\n\nB"}
         for i in range(6)]


def kto_built(rows, group):
    """The rows TRL 0.27's KTO builds, KL columns included: kto_trainer._get_kl_dataset moves the
    completions of each batch of `group` rows one row down, the last to the top."""
    answers = [ids(r["completion"]) for r in rows]
    kl = []
    for g in range(0, len(answers), group):
        batch = answers[g:g + group]
        kl += [batch[-1]] + batch[:-1]
    out = []
    for r, a, k in zip(rows, answers, kl, strict=True):
        p = ids(r["prompt"])
        seq, kseq = p + a + [1], p + k + [1]
        out.append({"prompt_input_ids": p, "completion_input_ids": seq,
                    "completion_labels": [LABEL_PAD] * len(p) + seq[len(p):],
                    "KL_prompt_input_ids": p, "KL_completion_input_ids": kseq,
                    "KL_completion_labels": [LABEL_PAD] * len(p) + kseq[len(p):]})
    return out


@pytest.mark.parametrize("pairs", [PAIRS, PAIRS[:4]])
def test_unpaired_gives_each_answer_once_in_rows_of_two_with_different_prompts(pairs):
    rows = unpaired(pairs)
    n = len(pairs)
    assert [r["label"] for r in rows] == [True, False] * n
    assert all(rows[k]["prompt"] != rows[k + 1]["prompt"] for k in range(0, 2 * n, 2))
    assert sorted(r["completion"] for r in rows if r["label"]) == sorted(p["chosen"] for p in pairs)
    assert sorted(r["completion"] for r in rows if not r["label"]) == sorted(p["rejected"] for p in pairs)
    own = {p[k]: p["prompt"] for p in pairs for k in ("chosen", "rejected")}
    assert all(own[r["completion"]] == r["prompt"] for r in rows)


def test_unpaired_refuses_a_prompt_with_more_than_half_the_pairs():
    with pytest.raises(ValueError, match="share one prompt"):
        unpaired(PAIRS[:3])  # 2 of the 3 pairs are Q0's


def test_kl_partner_follows_trls_rotation_with_a_short_last_batch():
    rows = [{"completion": str(i)} for i in range(7)]
    built = kto_built([{"prompt": "", **r} for r in rows], 3)
    got = [b["KL_completion_input_ids"][:-1] for b in built]
    assert [ids(rows[kl_partner(i, 7, 3)]["completion"]) for i in range(7)] == got
    assert kl_partner(6, 7, 3) == 6  # a batch of one row gets its own completion back


def test_check_kl_rows_passes_unpaired_and_catches_a_pairs_rows_side_by_side():
    rows = unpaired(PAIRS)
    built = kto_built(rows, 2)
    assert check_kl_rows(built, rows, TOK, 2) == []
    # the old order: each pair's chosen and rejected next to each other
    side = [r for p in PAIRS for r in ({"prompt": p["prompt"], "completion": p["chosen"], "label": True},
                                       {"prompt": p["prompt"], "completion": p["rejected"], "label": False})]
    bad = check_kl_rows(kto_built(side, 2), side, TOK, 2)
    assert len(bad) == 12 and all("answers the same prompt" in b for b in bad)
    built[3]["KL_completion_input_ids"] = built[3]["KL_completion_input_ids"][:-2]  # as if truncated
    assert len(check_kl_rows(built, rows, TOK, 2)) == 2  # the tokens, and so the labels too


def test_expected_is_prompt_completion_and_one_eos():
    assert expected(TOK, P, C) == ids(P) + ids(C) + [1]


def test_check_rows_dpo_catches_a_doubled_end():
    good = [{"prompt_input_ids": ids(P), "chosen_input_ids": ids(C) + [1], "rejected_input_ids": ids(R) + [1]}]
    assert check_rows("dpo", good, SOURCE, TOK) == []
    # TRL's default on a conversational row: the template's "<|im_end|>\n", then eos again
    doubled = [{**good[0], "chosen_input_ids": ids(C) + [1, ord("\n"), 1]}]
    assert check_rows("dpo", doubled, SOURCE, TOK) == ["row 0 chosen: tokens differ from the generated sequence"]


def test_check_rows_cpo_and_orpo_check_sequences_and_label_masks():
    def side(c):
        seq = ids(P) + ids(c) + [1]
        return seq, [LABEL_PAD] * len(P) + seq[len(P):]
    (cs, cl), (rs, rl) = side(C), side(R)
    good = [{"prompt_input_ids": ids(P), "chosen_input_ids": cs, "chosen_labels": cl,
             "rejected_input_ids": rs, "rejected_labels": rl}]
    for method in ("simpo", "orpo"):
        assert check_rows(method, good, SOURCE, TOK) == []
    unmasked = [{**good[0], "rejected_labels": rs}]
    assert check_rows("orpo", unmasked, SOURCE, TOK) == ["row 0 rejected: labels do not mask exactly the prompt"]


def test_check_rows_kto_checks_each_unpaired_completion():
    src = unpaired(PAIRS)
    built = kto_built(src, 2)
    assert check_rows("kto", built, src, TOK) == []
    built[1]["completion_input_ids"] = built[1]["completion_input_ids"][:-1]
    assert len(check_rows("kto", built, src, TOK)) == 2  # the sequence, and so the labels too


def test_bool_found_gives_trl_a_bool_and_keeps_the_version_tuple():
    def found(name, return_version=False):  # transformers 5.x: a tuple either way
        return (name == "here", "1.0" if name == "here" else None)

    assert bool(found("weave")) is True  # TRL 0.27's check reads this as installed
    available = bool_found(found)
    assert available("weave") is False and available("here") is True
    assert available("here", return_version=True) == (True, "1.0")


def test_load_pairs_reads_the_rendered_strings(tmp_path):
    pd.DataFrame([{"prompt": P, "chosen": C, "rejected": R, "id": str(i)} for i in range(5)]).to_parquet(
        tmp_path / "train_rendered.parquet", index=False)
    assert load_pairs(tmp_path, "train") == SOURCE * 5


def test_longest_takes_the_pairs_with_the_longest_sequence():
    pairs = [{"prompt": "p" * n, "chosen": "c" * c, "rejected": "r" * r}
             for n, c, r in [(5, 1, 1), (1, 9, 2), (3, 2, 6), (2, 3, 3)]]
    # prompt + the longer answer + eos: 7, 11, 10, 6
    assert longest(TOK, pairs, 2) == [pairs[1], pairs[2]]
    assert longest(TOK, pairs, 10) == [pairs[1], pairs[2], pairs[0], pairs[3]]


def test_po_configs_share_the_settings_and_do_not_cut_any_pair():
    from pathlib import Path

    import yaml

    from b2e.train.pref_train import METHODS

    files = sorted(Path(__file__).resolve().parents[1].glob("configs/po/*.yaml"))
    assert len(files) == 4
    for f in files:
        spec = yaml.safe_load(f.read_text())
        t = spec["trl"]
        assert spec["method"] in METHODS, f
        # the longest pair (tools/check_pref_render.py, 11556254): prompt 3,051, prompt + completion 6,903
        assert t["max_length"] >= 6903 and t.get("max_prompt_length", 3072) >= 3051, f
        per_step = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
        assert per_step == (64 if spec["method"] == "kto" else 32), f  # KTO: 2 completions a pair
        assert (t["learning_rate"], t["num_train_epochs"], t["bf16"]) == (5.0e-7, 2, True), f
