"""Reserved LLM news/hypothesis layer.

This module is a stub. It is disabled by default and has no authority to
change target weights or send orders. If enabled later, it may only attach
research hypotheses to the daily report for a human to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from paperbot import config


@dataclass
class Hypothesis:
    title: str
    rationale: str
    related_symbols: list[str] = field(default_factory=list)
    confidence: str = "low"  # low / medium / high, qualitative only
    suggested_test: str = ""


class LLMHypothesisLayer:
    """Interface for a future news-reading model.

    Contract:
    - Must not place, cancel, or size any order.
    - Must not override the strategy's target weights.
    - Output is advisory text only.
    """

    def __init__(self, enabled: bool | None = None):
        self.enabled = config.LLM_ENABLED if enabled is None else bool(enabled)

    def propose(self, date, context: dict) -> list[Hypothesis]:
        if not self.enabled:
            return []
        # Intentionally unimplemented: enable only after a researched prompt
        # and a human review step are added.
        raise NotImplementedError(
            "LLM layer is a reserved interface. It is off by default and "
            "must not decide trades. Implement propose() later if needed."
        )


def status_text() -> str:
    if config.LLM_ENABLED:
        return "LLM 新闻判断层已在配置中打开，但仍只能输出假设，不能下单。"
    return "LLM 新闻判断层默认关闭。它只能提出假设，不能决定交易。"
