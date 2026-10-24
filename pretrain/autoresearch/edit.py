"""Search/replace edit blocks: how the research agent changes train.py.

The agent answers with a one-line description and one or more blocks

    DESCRIPTION: raise the matrix learning rate to 0.05
    <<<<<<< SEARCH
    MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
    =======
    MATRIX_LR = 0.05        # learning rate for matrix parameters (Muon)
    >>>>>>> REPLACE

Each SEARCH text must occur exactly once in the current file (so an edit
cannot land in the wrong place), and the blocks apply in order. Asking for
blocks instead of a whole 630-line file keeps answers short and the change
reviewable; asking for blocks instead of a unified diff avoids line numbers
and hunk headers, which models often get wrong.
"""

from __future__ import annotations

import re

_BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n?>>>>>>> REPLACE", re.S)


class EditError(ValueError):
    pass


def parse(answer: str) -> tuple[str, list[tuple[str, str]]]:
    """(description, [(search, replace), ...]) from the agent's answer."""
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]
    m = re.search(r"^DESCRIPTION:\s*(.+)$", answer, re.M)
    desc = m.group(1).strip() if m else ""
    blocks = [(s, r) for s, r in _BLOCK.findall(answer)]
    if not blocks:
        raise EditError("no SEARCH/REPLACE block")
    return desc, blocks


def apply(source: str, blocks: list[tuple[str, str]]) -> str:
    for i, (search, replace) in enumerate(blocks):
        n = source.count(search)
        if n != 1:
            raise EditError(f"block {i + 1}: SEARCH text found {n} times (must be exactly once)")
        source = source.replace(search, replace, 1)
    return source
