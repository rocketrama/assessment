"""Flags corpus text that reads like instructions to an AI system (TASK.pdf §2.1, §7.5).

A flag never removes a chunk: a flagged chunk stays retrievable as information.
The defence is structural (the planner never sees corpus text, the composer wraps
it as untrusted data, tools only run from the user's plan). On top of that, the
instruction-like sentences of a flagged chunk are left out of what the answer
call is sent (without_injected_sentences), and the flag feeds the logs and eval.
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


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def without_injected_sentences(text: str) -> str:
    """The text with every sentence that reads like an instruction to an AI left out.

    Used on flagged chunks before they reach the answer call. Eval finding: told to
    ignore such text, the model still answered "Your line manager's salary band ...
    is not provided here as per policy" — it obeyed, but the planted request had
    reached the employee. A sentence like that carries no HR information, so
    leaving it out loses nothing; the rest of the chunk is sent as before."""
    lines = []
    for line in text.split("\n"):  # lines kept as they are: chunks hold tables and lists
        if not looks_like_injection(line):
            lines.append(line)
            continue
        kept = " ".join(s for s in _SENTENCE_END_RE.split(line) if not looks_like_injection(s))
        if kept.strip():
            lines.append(kept)
    return "\n".join(lines)
