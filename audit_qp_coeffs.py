"""Cross-check the QP cosine-multiplier coefficients between the JSON certificate
and the Lean ``qpNumerators`` list.

The Lean theorem ``Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`` is bound
to a concrete 200-cosine multiplier ``G`` whose numerators (with common
denominator ``10**8``) are stored in ``lean/Sidon/MultiScale.lean`` as
``qpNumerators : List ℤ``. The source of truth for those numerators is the
JSON certificate ``multiscale_arcsine_1292.json`` produced by
``delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py``.

This script verifies, bit-for-bit, that the two lists agree. It is intentionally
stdlib-only so it can be run from a fresh checkout / CI without any setup.

Exit codes:
    0  -- 200/200 numerators match.
    1  -- File / parse / shape error (with a clear diagnostic).
    2  -- A coefficient mismatch was detected (with index + expected/actual).

Usage:
    python audit_qp_coeffs.py
    python audit_qp_coeffs.py --verbose
    python audit_qp_coeffs.py --cert path/to/cert.json --lean path/to/MultiScale.lean

The JSON entries are stored as reduced rationals (e.g. ``95403771/50000000``),
not necessarily with denominator ``10**8`` already. The script multiplies each
rational by ``10**8`` and asserts the result is an exact integer; that is the
common-denominator convention shared with the Lean side.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CERT = (
    REPO_ROOT
    / "delsarte_dual"
    / "grid_bound_alt_kernel"
    / "certificates"
    / "multiscale_arcsine_1292.json"
)
DEFAULT_LEAN = REPO_ROOT / "lean" / "Sidon" / "MultiScale.lean"

COMMON_DENOMINATOR = 10**8  # 10^8, per the JSON certificate and Lean convention.
N_COEFFS = 200

# Lean list header. The Unicode "ℤ" is intentional — that is the literal token
# used in the source file (see ``def qpNumerators : List ℤ := [`` ).
LEAN_LIST_HEADER = "def qpNumerators : List ℤ := ["

# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def die(code: int, msg: str) -> "Never":  # type: ignore[name-defined]
    """Print ``msg`` to stderr and exit with ``code``."""
    print(f"audit_qp_coeffs: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# JSON side
# ---------------------------------------------------------------------------


def load_json_numerators(cert_path: Path) -> tuple[list[int], str]:
    """Return ``(numerators, sha256_of_body)``.

    Reads ``body.G.coeffs_q`` (a list of 200 reduced-rational strings),
    multiplies each by ``10**8``, and verifies the result is an integer (i.e.
    every stored fraction's reduced denominator divides ``10**8``). The
    returned numerators are the canonical numerators over ``10**8``.
    """
    if not cert_path.is_file():
        die(1, f"JSON certificate not found: {cert_path}")

    try:
        with cert_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        die(1, f"failed to parse JSON certificate {cert_path}: {exc}")

    try:
        coeffs_q = data["body"]["G"]["coeffs_q"]
    except (KeyError, TypeError) as exc:
        die(1, f"JSON certificate missing body.G.coeffs_q: {exc}")

    sha256_of_body = data.get("sha256_of_body", "<missing sha256_of_body>")

    if not isinstance(coeffs_q, list):
        die(1, f"body.G.coeffs_q must be a list, got {type(coeffs_q).__name__}")
    if len(coeffs_q) != N_COEFFS:
        die(
            1,
            f"body.G.coeffs_q has {len(coeffs_q)} entries, expected {N_COEFFS}",
        )

    numerators: list[int] = []
    for idx, raw in enumerate(coeffs_q):
        if not isinstance(raw, str):
            die(
                1,
                f"body.G.coeffs_q[{idx}] is {type(raw).__name__}, expected str",
            )
        try:
            frac = Fraction(raw)
        except (ValueError, ZeroDivisionError) as exc:
            die(1, f"body.G.coeffs_q[{idx}] = {raw!r} is not a valid rational: {exc}")

        scaled = frac * COMMON_DENOMINATOR  # Still a Fraction.
        if scaled.denominator != 1:
            die(
                1,
                "body.G.coeffs_q[{i}] = {raw!r} does not have a denominator "
                "dividing 10^8 (scaled = {scaled})".format(
                    i=idx, raw=raw, scaled=scaled
                ),
            )
        numerators.append(scaled.numerator)

    return numerators, sha256_of_body


# ---------------------------------------------------------------------------
# Lean side
# ---------------------------------------------------------------------------

# Strip a trailing ``--``-to-end-of-line comment from a single Lean line.
# Lean 4 line-comments are ``--`` (anywhere, not inside a string -- the
# ``qpNumerators`` block contains no strings, so a plain split is safe).
_INT_RE = re.compile(r"-?\d+")


def load_lean_numerators(lean_path: Path) -> list[int]:
    """Extract the 200 integer literals from ``qpNumerators`` in ``lean_path``."""
    if not lean_path.is_file():
        die(1, f"Lean source file not found: {lean_path}")

    try:
        text = lean_path.read_text(encoding="utf-8")
    except OSError as exc:
        die(1, f"failed to read Lean file {lean_path}: {exc}")

    header_idx = text.find(LEAN_LIST_HEADER)
    if header_idx < 0:
        die(
            1,
            "could not locate `def qpNumerators : List ℤ := [` in "
            f"{lean_path}",
        )

    open_bracket_idx = text.find("[", header_idx)
    if open_bracket_idx < 0:
        die(1, "no `[` after qpNumerators header (malformed Lean file)")

    # Find the matching `]`. The qpNumerators body contains no nested
    # brackets, so the next unescaped `]` after the `[` is the closer. To be
    # robust we still scan forward and account for the (highly unlikely) case
    # of a comment containing `]` -- we strip line comments before searching.
    body_search_start = open_bracket_idx + 1
    close_bracket_idx = -1
    cursor = body_search_start
    while cursor < len(text):
        # Find the next candidate `]` and the next `--` (line comment start).
        next_close = text.find("]", cursor)
        next_comment = text.find("--", cursor)
        if next_close < 0:
            break
        if next_comment < 0 or next_close < next_comment:
            close_bracket_idx = next_close
            break
        # The `--` precedes the next `]`. Skip past the end of that comment
        # line and continue scanning.
        eol = text.find("\n", next_comment)
        if eol < 0:
            # Comment runs to end of file -- can't be the right list.
            break
        cursor = eol + 1
    if close_bracket_idx < 0:
        die(1, "could not find closing `]` for qpNumerators list")

    body = text[body_search_start:close_bracket_idx]

    # Strip every ``--``-to-end-of-line comment from the body so the integer
    # regex can't accidentally pick up integers embedded in comments.
    body_no_comments_lines: list[str] = []
    for line in body.splitlines():
        comment_idx = line.find("--")
        if comment_idx >= 0:
            line = line[:comment_idx]
        body_no_comments_lines.append(line)
    body_clean = "\n".join(body_no_comments_lines)

    # Extract signed integer literals.
    matches = _INT_RE.findall(body_clean)
    if len(matches) != N_COEFFS:
        die(
            1,
            f"found {len(matches)} integer literals inside qpNumerators, "
            f"expected exactly {N_COEFFS} (file: {lean_path})",
        )

    try:
        return [int(tok) for tok in matches]
    except ValueError as exc:  # pragma: no cover -- regex guarantees digits.
        die(1, f"failed to parse integer literal from qpNumerators: {exc}")


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------


def cross_check(json_nums: list[int], lean_nums: list[int]) -> None:
    """Compare the two lists; emit a diagnostic and exit on mismatch."""
    if len(json_nums) != len(lean_nums):  # Pre-checked, but defence in depth.
        die(
            1,
            f"length mismatch: JSON has {len(json_nums)}, Lean has {len(lean_nums)}",
        )

    mismatches: list[tuple[int, int, int]] = []
    for j in range(N_COEFFS):
        if json_nums[j] != lean_nums[j]:
            mismatches.append((j + 1, json_nums[j], lean_nums[j]))

    if mismatches:
        print(
            f"MISMATCH: {len(mismatches)}/{N_COEFFS} coefficient(s) differ "
            "between JSON certificate and Lean qpNumerators.",
            file=sys.stderr,
        )
        for one_based, expected, actual in mismatches[:20]:
            print(
                f"  a_{one_based}: expected (JSON) = {expected}, "
                f"actual (Lean) = {actual}, delta = {actual - expected}",
                file=sys.stderr,
            )
        if len(mismatches) > 20:
            print(
                f"  ... and {len(mismatches) - 20} more mismatches.",
                file=sys.stderr,
            )
        sys.exit(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bit-for-bit cross-check the 200 QP cosine-multiplier coefficients "
            "between the JSON certificate (body.G.coeffs_q, common denominator "
            "10^8) and the Lean qpNumerators list in Sidon/MultiScale.lean."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cert",
        type=Path,
        default=DEFAULT_CERT,
        help="Path to the JSON certificate.",
    )
    parser.add_argument(
        "--lean",
        type=Path,
        default=DEFAULT_LEAN,
        help="Path to the Lean source file containing qpNumerators.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="On success, also print the first and last 5 coefficients.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    json_nums, sha256_of_body = load_json_numerators(args.cert)
    lean_nums = load_lean_numerators(args.lean)
    cross_check(json_nums, lean_nums)

    print(f"OK: {N_COEFFS}/{N_COEFFS} coefficients match between JSON and Lean.")
    print(f"  JSON certificate : {args.cert}")
    print(f"  Lean source       : {args.lean}")
    print(f"  sha256_of_body    : {sha256_of_body}")
    if args.verbose:
        print("  First 5 numerators (a_1..a_5):")
        for j in range(5):
            print(f"    a_{j+1:>3} = {json_nums[j]}")
        print("  Last 5 numerators (a_196..a_200):")
        for j in range(N_COEFFS - 5, N_COEFFS):
            print(f"    a_{j+1:>3} = {json_nums[j]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
