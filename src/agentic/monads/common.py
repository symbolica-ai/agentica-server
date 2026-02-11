from pathlib import Path
from typing import TYPE_CHECKING, Generator

__all__ = ['JINJA_ENV_CACHE']


# === Utilities ===

if TYPE_CHECKING:
    from jinja2 import Environment

BASE_DIR = Path(__file__).parent
REPL_TXT_DIR = BASE_DIR / 'repl_tool' / 'multi_turn' / 'text'


JINJA_ENV_CACHE: dict[str, 'Environment'] = {}


def _find_matching_end(text: str, start: str, end: str, pos: int) -> int:
    """Find matching end tag position, handling nesting. Raises ValueError if not found."""
    depth = 1
    start_len, end_len = len(start), len(end)
    while depth > 0:
        next_start = text.find(start, pos)
        next_end = text.index(end, pos)
        # Prioritize start when: start comes first, OR they overlap and start is longer
        # (handles case where end is prefix of start, like ``` vs ```python)
        if next_start != -1 and (
            next_start < next_end or (next_start == next_end and start_len > end_len)
        ):
            depth += 1
            pos = next_start + start_len
        else:
            depth -= 1
            pos = next_end + end_len
    return pos - end_len


def text_between(text: str, start: str, end: str) -> Generator[str, None, None]:
    start_len = len(start)
    end_len = len(end)
    ptr = 0
    while True:
        try:
            start_pos = text.index(start, ptr)
            end_pos = _find_matching_end(text, start, end, start_pos + start_len)
            yield text[start_pos + start_len : end_pos]
            ptr = end_pos + end_len
        except ValueError:
            break


def text_not_between(text: str, start: str, end: str) -> Generator[str, None, None]:
    start_len = len(start)
    end_len = len(end)
    ptr = 0
    while True:
        try:
            start_pos = text.index(start, ptr)
        except ValueError:
            yield text[ptr:]  # No more starts, yield rest
            break
        # Check if matching end exists before yielding
        try:
            end_pos = _find_matching_end(text, start, end, start_pos + start_len)
        except ValueError:
            yield text[ptr:]  # Unmatched start, yield everything remaining
            break
        yield text[ptr:start_pos]
        ptr = end_pos + end_len
