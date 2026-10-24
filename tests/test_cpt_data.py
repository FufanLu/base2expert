from b2e.data.pretokenize import chunk, tokenize_docs
from b2e.train.cpt_dataset import sample_fields


class FakeTok:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


def test_chunk_drops_short_tail():
    assert list(chunk(list(range(10)), 4, 3)) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_documents_end_with_eos_and_are_chunked():
    out = list(tokenize_docs(["abcde", "xy"], FakeTok(), eos_id=0, max_len=4, min_len=1, batch=1))
    assert out == [(0, [97, 98, 99, 100]), (0, [101, 0]), (1, [120, 121, 0])]


def test_first_token_is_never_a_target():
    f = sample_fields([5, 6, 7], mrope_rows=False)
    assert f["loss_mask"] == [0, 1, 1]
    assert f["position_ids"] == [0, 1, 2]


def test_mrope_position_rows():
    f = sample_fields([5, 6], mrope_rows=True)
    assert f["position_ids"] == [[0, 1]] * 4


def test_split_thinking():
    from b2e.data.synth import split_thinking

    assert split_thinking("step 1...\n</think>\n\nso \\boxed{3}") == ("step 1...", "so \\boxed{3}")
    assert split_thinking("no think block \\boxed{3}") == ("", "no think block \\boxed{3}")


def test_heldout_documents_are_not_in_the_training_stream():
    from b2e.data.m0_mix import split_heldout

    held, rest = split_heldout(iter(["a", "b", "c", "d"]), 2)
    assert held == ["a", "b"]
    assert list(rest) == ["c", "d"]


def test_bpb_counts_bytes_after_the_first_token():
    from b2e.eval.bpb import scored_bytes

    # "ab中" as three tokens: the first ("a") is never predicted
    assert scored_bytes("ab中", [(0, 1), (1, 2), (2, 3)]) == 1 + 3
