import re
import pathlib
import pytest
import math
import main  # your module under test

SIGFIGS_THRESHOLD = 10
# Match ints/floats incl. scientific notation (not inf/nan)
NUM_RE = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')

def _tokenize(line: str):
    """Yield (is_number: bool, token: str) covering the whole line."""
    pos = 0
    for m in NUM_RE.finditer(line):
        if m.start() > pos:
            yield (False, line[pos:m.start()])
        yield (True, m.group(0))
        pos = m.end()
    if pos < len(line):
        yield (False, line[pos:])

def _first_diff_idx(a: str, b: str) -> int:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))

# ---------- Significant-figure helpers ----------

def _sigfigs_in_token(tok: str) -> int:
    """
    Infer significant-figure count from the *string* token.
    Rules:
      - Only the mantissa (before 'e'/'E') matters for counting.
      - Leading zeros are not significant.
      - If there's a decimal point, trailing zeros ARE significant.
      - If there's no decimal point, trailing zeros are NOT significant.
      - If the number is all zeros (e.g., '0' or '0.000'), return 1.
    """
    s = tok.lstrip("+-")
    if 'e' in s or 'E' in s:
        mantissa, _exp = re.split(r'[eE]', s, maxsplit=1)
    else:
        mantissa = s

    if '.' in mantissa:
        left, right = mantissa.split('.', 1)
        # Remove leading zeros from the left part
        left = left.lstrip('0')
        # Keep trailing zeros in right part (they are significant)
        # Remove leading zeros in right for counting, but if left empty,
        # we must still count zeros on the right as significant.
        # Combine and count digits
        digits = left + right
        # Remove non-digits just in case
        digits = re.sub(r'\D', '', digits)
        # Drop leading zeros total
        digits = digits.lstrip('0')
        if digits == '':
            # number like 0.000 or 0.0
            # Count digits in right (including zeros) if any, else 1
            return max(1, len(re.sub(r'\D', '', right)))
        return len(digits)
    else:
        # Integer: trailing zeros are NOT significant
        s2 = mantissa.lstrip('0')
        if s2 == '':
            return 1  # it's zero
        # remove trailing zeros
        s2 = s2.rstrip('0')
        return len(s2)

def _round_to_sigfigs(x: float, n: int) -> float:
    """Round x to n significant figures (returns float)."""
    if x == 0 or math.isnan(x) or math.isinf(x):
        return x
    # Use scientific notation rounding
    # fmt like '1.234e+03' with n sig figs -> n-1 after decimal
    s = f"{x:.{max(n-1, 0)}e}"
    return float(s)

def _compare_numbers_by_sigfig(tok_got: str, tok_ref: str, line_no: int, token_idx: int, fixed_sigfigs: int | None = None):
    """
    Compare two numeric tokens by significant figures.
    - If fixed_sigfigs is provided, use that for both.
    - Otherwise, infer sig figs from the reference token string.
    """
    v1 = float(tok_got)
    v2 = float(tok_ref)

    # Handle NaN/Inf explicitly (shouldn't match NUM_RE normally, but be safe)
    if any(map(math.isnan, (v1, v2))) or any(map(math.isinf, (v1, v2))):
        assert v1 == v2, (
            f"Line {line_no}, numeric token {token_idx}: special value mismatch.\n"
            f"  got: {tok_got}\n  ref: {tok_ref}"
        )
        return

    n = fixed_sigfigs if fixed_sigfigs is not None else _sigfigs_in_token(tok_ref)

    r1 = _round_to_sigfigs(v1, n)
    r2 = _round_to_sigfigs(v2, n)

    # Compare equality after sigfig rounding. If desired, allow a tiny float fuzz.
    if not math.isclose(r1, r2, rel_tol=0, abs_tol=0):
        raise AssertionError(
            f"Line {line_no}, numeric token {token_idx} mismatch at {n} significant figures:\n"
            f"  got: {tok_got} -> rounded({n} sf) = {r1}\n"
            f"  ref: {tok_ref} -> rounded({n} sf) = {r2}"
        )

# ---------- Line comparator using sig figs ----------

def _compare_lines(line1: str, line2: str, line_no: int, fixed_sigfigs: int | None = None):
    t1 = list(_tokenize(line1))
    t2 = list(_tokenize(line2))
    assert len(t1) == len(t2), (
        f"Line {line_no}: token count differs.\n"
        f"  got: {t1}\n  ref: {t2}"
    )
    for j, ((isnum1, tok1), (isnum2, tok2)) in enumerate(zip(t1, t2), start=1):
        if isnum1 and isnum2:
            _compare_numbers_by_sigfig(tok1, tok2, line_no, j, fixed_sigfigs=fixed_sigfigs)
        elif isnum1 != isnum2:
            raise AssertionError(
                f"Line {line_no}, token {j}: numeric/text kind mismatch:\n"
                f"  got: {tok1!r} ({'num' if isnum1 else 'text'})\n"
                f"  ref: {tok2!r} ({'num' if isnum2 else 'text'})"
            )
        else:
            if tok1 != tok2:
                k = _first_diff_idx(tok1, tok2)
                context_got = tok1[max(0,k-10):k+10]
                context_ref = tok2[max(0,k-10):k+10]
                raise AssertionError(
                    f"Line {line_no}, text token {j} differs at char {k}:\n"
                    f"  got: {tok1!r}\n"
                    f"  ref: {tok2!r}\n"
                    f"  context got: {context_got!r}\n"
                    f"  context ref: {context_ref!r}"
                )

# ---------- The test ----------

def test_main_SF1():
    # Produce results.out
    main.exec(1, list(range(1, 23)), None, None, True, None, None, None)

    here = pathlib.Path(__file__).resolve().parent
    res_file = here / "results.out"
    ref_file = here / "tests" / "verification" / "SF1_reference.out"

    res_lines = res_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()

    assert len(res_lines) == len(ref_lines), (
        f"Line count differs: got {len(res_lines)} vs ref {len(ref_lines)}"
    )

    for i, (a, b) in enumerate(zip(res_lines, ref_lines), start=1):
        _compare_lines(a, b, i, fixed_sigfigs=SIGFIGS_THRESHOLD)
