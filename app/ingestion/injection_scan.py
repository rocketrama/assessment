"""Flags corpus text that reads like instructions to an AI system (TASK.pdf §2.1, §7.5).

A flag never removes text: a flagged chunk stays retrievable as information. The
defence is structural (the planner never sees corpus text, the composer wraps
it as untrusted data, tools only run from the user's plan); this flag adds a
signal for the composer, the logs and the eval.
"""

import re

_PATTERNS = [
    r"\bignore\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|rules|prompts?)\b",
    r"\bdisregard\b.{0,60}\b(instructions?|rules|polic(y|ies)|prompts?)\b",
    r"\b(note|message|instructions?|reminder)s?\s+(for|to)\s+(the\s+|any\s+)?(ai|assistant|chat\s?bot|llm|language model|model)\b",
    r"\b(ai|virtual)\s+assistants?\b.{0,80}\b(should|must|always|never|tell|include|reveal|share)\b",
    r"\byou\s+are\s+(now\s+)?(an?\s+)?(ai|assistant|chat\s?bot|language model)\b",
    r"\bsystem\s+prompt\b",
    r"\b(call|use|invoke|run)\s+the\s+\w+\s+tool\b",
    r"\b(apply|cancel|approve)_leave\b",
    r"\bdo\s+not\s+(tell|inform|show)\s+the\s+(user|employee)\b",
    r"\bthis\s+is\s+approved\s+by\s+(hr|management|it)\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.DOTALL)


def looks_like_injection(text: str) -> bool:
    return _INJECTION_RE.search(text) is not None
