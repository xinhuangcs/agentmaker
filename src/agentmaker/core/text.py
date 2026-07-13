"""agentmaker.core.text: lightweight text utilities.

count_tokens: a rough token estimate for mixed CJK / Western text (CJK plus Japanese/Korean characters count as 1 token each, everything else at roughly 4 characters per token), without pulling in heavy dependencies such as tiktoken. Shared by rag.splitter (chunk budget) and context (context budget) so the rule lives in one place and cannot drift.
"""

import re
from typing import Callable

# CJK plus Japanese/Korean characters: estimated at 1 token each. Covers kana / CJK Extension A / CJK Unified /
# Hangul syllables / CJK compatibility ideographs, not just the basic Chinese block. Otherwise a long run of
# whitespace-free Japanese / Korean text would count as a single token and severely underestimate the budget
# (builder / reducer / splitter all rely on this).
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힣豈-﫿]")

# Token counter seam: `(text) -> int`. The framework defaults to the count_tokens below (rough, zero-dependency,
# undercounts English / JSON tool_calls). For more accuracy in production, inject tiktoken or similar via each
# component's token_counter= constructor argument (see builder / history_compactor / harness / splitter).
TokenCounter = Callable[[str], int]


def count_tokens(text: str) -> int:
    """Estimate the token count of text: CJK plus Japanese/Korean characters count as 1 token each, everything else at roughly 4 characters per token.

    This is a pre-send context budget estimate and does not feed billing or quota (cost and limits always use the
    real usage returned by the LLM). It does not tokenize on whitespace, so long whitespace-free runs
    (base64 / compressed data / very long URLs) are correctly covered by "other characters / 4" rather than being
    counted as a single token.

    Args:
        text: The text to measure.

    Returns:
        int: The estimated token count.
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))     # count of CJK plus Japanese/Korean characters
    other = len(text) - cjk              # remaining characters (Latin letters / digits / spaces / punctuation, etc.)
    return cjk + (other + 3) // 4        # CJK at 1 token each (conservative); others at ~4 chars/token, (x+3)//4 rounds up as an integer, zero dependency
