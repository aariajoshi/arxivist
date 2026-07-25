"""
Shared utility: verify that a function satisfies Simon's algorithm's 2-to-1
promise for a claimed hidden period s.

Simon's algorithm assumes f(x) = f(x') iff x' in {x, x XOR s}. Small,
randomly-constructed cipher instances (permutations, toy block ciphers) can
accidentally violate this promise -- i.e. have EXTRA collisions beyond the
intended period -- especially at small block sizes where the birthday
paradox bites hard on a tiny domain. This is a combinatorial property of
the specific concrete instance, not something the paper discusses (it
implicitly assumes "generic" constructions satisfy the promise, which holds
with high probability only as n grows). This module lets callers verify (or
search for) instances that genuinely satisfy the promise before running
Simon's algorithm against them, rather than silently returning wrong/no
answers on degenerate instances.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable


def verify_two_to_one_promise(f: Callable[[int], int], s: int, n: int) -> bool:
    """Check that f(x) = f(x') holds iff x' in {x, x XOR s}, for all x in [0, 2^n).

    Args:
        f: the function under test, f: [0, 2^n) -> anything hashable.
        s: the claimed hidden period.
        n: bit-length of the domain.

    Returns:
        True iff f is genuinely 2-to-1 with period exactly {0, s} (or,
        degenerately, s=0 and f is injective -- not useful for Simon's
        algorithm, callers should exclude s=0 separately).
    """
    groups: dict = defaultdict(list)
    for x in range(2 ** n):
        groups[f(x)].append(x)
    for xs in groups.values():
        if len(xs) not in (0, 2):
            return False  # a size-1 group means f(x) != f(x XOR s) for that x -- promise violated
        if len(xs) == 2 and xs[1] != (xs[0] ^ s):
            return False
    return True
