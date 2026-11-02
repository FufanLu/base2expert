import json

import pandas as pd

from b2e.data.pref_pairs import load_samples
from b2e.data.rescore import COLUMNS, rescore, unequal

DRUGS = json.dumps({"kind": "charges", "value": ["[走私、贩卖、运输、制造]毒品"]}, ensure_ascii=False)
THEFT = json.dumps({"kind": "charges", "value": ["盗窃"]}, ensure_ascii=False)


def answer(charge):
    return f"<think>\nfacts\n</think>\n\n\\boxed{{{charge}}}"


def parts(tmp_path, rows):
    """A synth parts dir as synth writes it: stats (stored scores) and samples (texts)."""
    stats = [{"data_source": "cail2018_charges", "id": i, "pass_rate": sum(s) / len(s), "mean_score": sum(s) / len(s),
              "ground_truth": gt, "answers": [None] * len(s), "scores": s, "tokens": [10] * len(s),
              "finish": ["stop"] * len(s), "seconds": 1.0} for i, (gt, _, s) in enumerate(rows)]
    samples = [{"data_source": "cail2018_charges", "id": i, "texts": t} for i, (_, t, _) in enumerate(rows)]
    pd.DataFrame(stats).to_parquet(tmp_path / "stats_0000.parquet", index=False)
    pd.DataFrame(samples).to_parquet(tmp_path / "samples_0000.parquet", index=False)
    return tmp_path


def test_rescore_pays_selective_charges_the_old_verifier_did_not(tmp_path):
    rows = [
        # stored with the old verifier: a right 贩卖毒品 scored 0 against the bracketed label
        (DRUGS, [answer("贩卖毒品罪"), answer("盗窃")], [0.0, 0.0]),
        (THEFT, [answer("盗窃"), answer("诈骗")], [1.0, 0.0]),  # unaffected
    ]
    rates, summary = rescore(load_samples(parts(tmp_path, rows)))
    assert list(rates.columns) == COLUMNS
    assert [list(s) for s in rates["scores"]] == [[1.0, 0.0], [1.0, 0.0]]
    assert list(rates["pass_rate"]) == [0.5, 0.5] and list(rates["answers"][0]) == ["贩卖毒品罪", "盗窃"]
    s = summary["cail2018_charges"]
    assert s["prompts"] == 2 and s["samples_changed"] == 1
    assert (s["pass_rate_before"], s["pass_rate_after"]) == (0.25, 0.5)
    assert (s["unequal_before"], s["unequal_after"]) == (1, 2)  # the first prompt enters the pool


def test_unequal_is_rl_pools_rule():
    assert unequal([1.0, 0.0]) and not unequal([0.0, 0.0]) and unequal([0.4, 0.6])
