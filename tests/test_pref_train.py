import pandas as pd

from b2e.train.pref_train import LABEL_PAD, check_rows, expected, load_pairs, unpaired


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


def test_unpaired_gives_each_pair_a_desirable_and_an_undesirable_completion():
    rows = unpaired(SOURCE * 2)
    assert [r["label"] for r in rows] == [True, False, True, False]
    assert rows[0] == {"prompt": P, "completion": C, "label": True} and rows[1]["completion"] == R


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
    src = unpaired(SOURCE)
    built = []
    for r in src:
        seq = ids(P) + ids(r["completion"]) + [1]
        built.append({"prompt_input_ids": ids(P), "completion_input_ids": seq,
                      "completion_labels": [LABEL_PAD] * len(P) + seq[len(P):]})
    assert check_rows("kto", built, src, TOK) == []
    built[1]["completion_input_ids"] = built[1]["completion_input_ids"][:-1]
    assert len(check_rows("kto", built, src, TOK)) == 2  # the sequence, and so the labels too


def test_load_pairs_reads_the_rendered_strings(tmp_path):
    pd.DataFrame([{"prompt": P, "chosen": C, "rejected": R, "id": str(i)} for i in range(5)]).to_parquet(
        tmp_path / "train_rendered.parquet", index=False)
    assert load_pairs(tmp_path, "train", rows=2) == SOURCE * 2
    assert len(load_pairs(tmp_path, "train")) == 5


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
