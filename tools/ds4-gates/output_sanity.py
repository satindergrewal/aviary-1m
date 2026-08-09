#!/usr/bin/env python3
"""Grade a paged output against a static reference: DEGENERATE, DIVERGENT, or OK.

⚠⚠ WHY THIS EXISTS. `FINDINGS-paged-cross-request.md` is OPEN and its own "next step" is the reason
for this file, verbatim:

    "Every existing marker, warning and assert on the paged path passes while the output is wrong.
     So the first job is not to find the bug; it is to add a check that can *fail* when it happens,
     because none of the current ones can. This defect is not merely intermittent, it is
     uninstrumented."

Measured there: **2 of 6 runs FAIL at 224,992 tokens** with the champion active. That is a third of
runs producing wrong text inside the exact 256k-1M band the owner's bar is set on, and a failing
server's log is byte-comparable to a clean one -- same 88 static-path warnings, same 8 fallback
layers, same checkout accounting, different tokens.

⚠ AND THE NEEDLE GATE CANNOT SEE IT. Every parity run in this lane validates with a needle, and the
documented failures range from "fluent but different from static" to degenerate. A fluent-but-wrong
answer still contains the passcode, so it passes. The needle is a validity guard for the SPEED
number; it was never a corruption detector, and reading it as one is gate-shape-excludes-the-defect.

⚠ WHAT THIS DELIBERATELY DOES NOT DO: fail on byte inequality. MEASURED 2026-08-09, static-vs-static
over 200 tokens: **a 67% false-fail rate**. Two runs of the SAME arm disagree two times in three, so
a hash comparison would cry wolf on two thirds of healthy runs and the alarm would be ignored inside
a day -- which is worse than no alarm. Divergence is therefore GRADED, not asserted, and only
DEGENERACY -- which no healthy sample exhibits -- is treated as a failure.

usage:  output_sanity.py <static.json|-> <paged.json>       # llama-server /completion responses
        output_sanity.py --text <static.txt> <paged.txt>
exit 0 = OK · 1 = DEGENERATE (a real defect) · 3 = DIVERGENT (report, do not fail) · 2 = VOID
"""
import json, sys, re, unicodedata

# ── detectors ────────────────────────────────────────────────────────────────
# Each targets a failure OBSERVED in the findings doc, not an imagined one. The three recorded
# samples are:
#     ' H and 和Q of/路 ） NOTE –'      mixed-script word salad
#     '. - - - - - '                    punctuation repetition
#     ' is the 1 - - - -'               ditto, with a fluent prefix
# A detector that cannot fire on those is decoration.

def _tokens(t):
    return re.findall(r"\S+", t)

def rep_ratio(t):
    """Fraction of whitespace tokens that repeat the immediately preceding token.

    '. - - - - -' scores 0.8; ordinary prose runs ~0.0-0.05 ('the the' is rare and short)."""
    w = _tokens(t)
    if len(w) < 8:
        return None                       # too short to be evidence either way
    return sum(1 for a, b in zip(w, w[1:]) if a == b) / (len(w) - 1)

def ngram_loop(t, k=4):
    """Fraction of k-gram positions occupied by a k-gram that occurs more than once.

    ⚠⚠ THE DETECTOR THIS FILE SHIPPED WITHOUT, AND REAL DATA CAUGHT IT WITHIN THE HOUR.
    `rep_ratio` only sees ADJACENT identical tokens. Fed the actual 512-token completion from a
    256k parity arm --

        'MAGENTA-7742. Remember it.7742. Remember it.7742. Remember it.7742. Remember it.'

    -- it scored **0.00** and the whole file returned OK. Whitespace tokens are
    `it.7742.` `Remember` `it.7742.` `Remember`: **no two neighbours are equal**, so an
    adjacent-duplicate test is blind to every loop of period 2 or more, which is what looping
    models actually produce. `'. - - - -'` was period ONE and that is the only reason the original
    detector worked on it.

    ⇒ A repeated k-gram catches any period up to k. Prose reuses short phrases, so the threshold is
      set well above ordinary reuse rather than at zero.

    ⚠ I predicted this file would FALSE-FIRE on `ignore_eos` output. It did the opposite: it stayed
      silent on output that is unambiguously degenerate. **The prediction was wrong in the direction
      that matters** -- I would have trusted a green.
    """
    w = _tokens(t)
    if len(w) < k * 4:
        return None
    grams = [tuple(w[i:i+k]) for i in range(len(w) - k + 1)]
    seen = {}
    for g in grams:
        seen[g] = seen.get(g, 0) + 1
    repeated = sum(n for g, n in seen.items() if n > 1)
    return repeated / len(grams)

def alnum_ratio(t):
    """Fraction of non-space characters that are letters or digits.

    '. - - - - -' scores ~0.0. Prose runs 0.75-0.95. Punctuation storms are the tell."""
    c = [ch for ch in t if not ch.isspace()]
    if len(c) < 24:
        return None
    return sum(1 for ch in c if ch.isalnum()) / len(c)

def _scripts_of(s):
    out = set()
    for ch in s:
        if not ch.isalpha():
            continue
        try:
            out.add(unicodedata.name(ch).split()[0])
        except ValueError:
            pass
    return out

def intra_token_mix(t):
    """Count of whitespace tokens containing letters from TWO OR MORE different scripts.

    ⚠⚠ THIS DETECTOR EXISTS BECAUSE MY FIXTURE WAS MORE BROKEN THAN THE REAL DEFECT. The first
    version of this file "caught 3 of 3" recorded failures -- but only because I had synthesised
    longer, more mixed samples than the ones in the findings doc. Fed the LITERAL recorded string,

        ' H and 和Q of/路 ） NOTE –'

    it graded DIVERGENT, not DEGENERATE: seven tokens is under the repetition detector's minimum,
    twenty-three characters is under the alphanumeric one's, and Latin+Han is only two scripts. **A
    test fixture built to be caught proves the fixture, not the instrument** -- gate-shape-excludes-
    the-defect, on the very file written to end that.

    What actually distinguishes that string is script mixing INSIDE single tokens: `和Q`, `of/路`.
    Healthy multilingual output switches script at word boundaries, so this fires at seven tokens
    where every length-based detector is still blind. Threshold is 2 tokens, not 1, so an isolated
    transliteration or a stray glyph is not a defect."""
    return sum(1 for w in _tokens(t) if len(_scripts_of(w)) >= 2)

def script_mix(t):
    """Count of distinct Unicode scripts among alphabetic characters.

    ⚠ CALIBRATED TO ALLOW BILINGUAL OUTPUT. Two scripts is normal for a multilingual model answering
    in one language with a quoted term in another. Three or more in a short span is the ' H and 和Q
    of/路 ）' shape. This fires on MIXTURE, never on a non-Latin script by itself -- a fully Chinese
    answer scores 1 and is not a defect."""
    scripts = set()
    for ch in t:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        scripts.add(name)
    return len(scripts)

# ⚠ THRESHOLDS ARE A CLAIM ABOUT HEALTHY OUTPUT AND ARE MARKED AS SUCH. They are set well outside
# anything prose produces, so a fire means "no healthy sample looks like this", not "this differs
# from the reference". Tighten only with a measured false-positive rate in hand -- the 67% figure
# above is what happens when a bound is set by intuition.
REP_MAX      = 0.35   # prose ~0.00-0.05; the recorded failures ~0.6-0.8
ALNUM_MIN    = 0.45   # prose 0.75-0.95; punctuation storms ~0.0-0.2
SCRIPTS_MAX  = 3      # 1-2 normal (incl. bilingual); 3+ is the word-salad shape
INTRA_MAX    = 1      # >1 token with mixed scripts INSIDE it; fires at 7 tokens, length-independent
# ⚠ 0.60 IS DELIBERATELY HIGH. Ordinary prose repeats 4-grams occasionally (lists, refrains, code),
# and a low bar here would fire on structured output. The real 512-token loop measured on this box
# scores near 1.0, so the separation is wide and the threshold does not need to be tight.
NGRAM_MAX    = 0.60
#
# ⚠⚠ MEASURED FALSE-POSITIVE REGIME -- READ THIS BEFORE POINTING THIS FILE AT LONG OUTPUT.
# My false-positive controls were all ONE-LINERS, so they could not test the regime the detector
# actually runs in. Built long-form controls and measured:
#
#     text                       tokens  4-gram  TTR    verdict
#     REAL LOOP (a paged arm)      129    0.97   0.047  DEGENERATE   correct
#     long prose                   158    0.00   0.703  clean        correct
#     structured table (40 rows)   560    0.71   0.086  DEGENERATE   **FALSE POSITIVE**
#     the same code block x8       336    1.00   0.083  DEGENERATE   arguably correct
#
# A model asked to list forty layers and their status produces exactly the third row. **The detector
# cannot tell "the model is looping" from "the model is producing a legitimately repetitive
# structure."**
#
# ⇒ I tried the obvious co-condition -- require a low type-token ratio as well -- and **MEASURED that
#   it does not separate them**: the real loop is 0.047 and the table is 0.086, a factor of 1.8, and
#   a table with fewer distinct values lands below the loop. **Reporting the failed fix matters more
#   than the successful one**: without this note the next reader will try the same thing.
#
# ⇒ THE RESOLUTION IS SCOPE, NOT THRESHOLD. Run this on an ANSWER, never on free-running `ignore_eos`
#   filler -- `--prefix`, or a second short request with ignore_eos off. In answer-length text the
#   one-liner controls above ARE representative and the detector is sound. Outside that scope its
#   verdict is not trustworthy, and no threshold I can measure makes it so.

def degeneracy(t):
    """Return (verdict, [reasons]). Only fires on shapes healthy output does not produce."""
    reasons = []
    r = rep_ratio(t)
    if r is not None and r > REP_MAX:
        reasons.append(f"token repetition {r:.2f} > {REP_MAX} (adjacent duplicates)")
    a = alnum_ratio(t)
    if a is not None and a < ALNUM_MIN:
        reasons.append(f"alphanumeric fraction {a:.2f} < {ALNUM_MIN} (punctuation storm)")
    s = script_mix(t)
    if s > SCRIPTS_MAX:
        reasons.append(f"{s} distinct scripts mixed (> {SCRIPTS_MAX}) -- word-salad shape")
    g = ngram_loop(t)
    if g is not None and g > NGRAM_MAX:
        reasons.append(f"{g*100:.0f}% of 4-grams are repeated (> {NGRAM_MAX*100:.0f}%) -- the text is "
                       f"looping with a period an adjacent-duplicate test cannot see")
    m = intra_token_mix(t)
    if m > INTRA_MAX:
        reasons.append(f"{m} tokens mix scripts INTERNALLY (> {INTRA_MAX}), e.g. Han inside a Latin "
                       f"word -- healthy output switches script at word boundaries")
    return (len(reasons) > 0), reasons

def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

# ── main ─────────────────────────────────────────────────────────────────────
def load(p, as_text):
    if as_text:
        return open(p, encoding="utf-8", errors="replace").read()
    d = json.load(open(p, encoding="utf-8", errors="replace"))
    if "error" in d:
        raise ValueError(f"server returned an error: {str(d['error'])[:120]}")
    return d.get("content", "")

def main(argv):
    as_text = "--text" in argv
    # ⚠⚠ --prefix EXISTS BECAUSE OF A COLLISION BETWEEN TWO CORRECT FIXES.
    # `ignore_eos` makes n_predict a floor, which is required for a readable decode rate. It also
    # makes the model run PAST its answer, and what follows a completed answer is a loop: the real
    # 512-token completion from a 256k parity arm scores **97% repeated 4-grams**. So a degeneracy
    # check over the full completion VOIDs every healthy run of the gate it is meant to protect.
    # ⇒ Grade the ANSWER, not the filler. Either grade a prefix, or send a second short request with
    #   ignore_eos off -- the parity gate's needle guard already works that way and is the cleaner
    #   shape. Two fixes that are each right and jointly wrong is how three defects got built today.
    prefix = 0
    if "--prefix" in argv:
        i = argv.index("--prefix")
        try: prefix = int(argv[i+1])
        except Exception: prefix = 0
        argv = argv[:i] + argv[i+2:]
    args = [a for a in argv[1:] if a != "--text"]
    if len(args) != 2:
        print(__doc__.strip().splitlines()[-3], file=sys.stderr)
        return 2
    try:
        ref, got = load(args[0], as_text), load(args[1], as_text)
    except Exception as e:
        print(f"  output-sanity VOID: {e}")
        return 2

    if prefix > 0:
        ref, got = ref[:prefix], got[:prefix]
        print(f"  (grading the first {prefix} characters only -- see the --prefix note in the source)")

    # ⚠ AN EMPTY SAMPLE MUST NOT GRADE AS CLEAN. Zero bytes passes every detector above by having
    # nothing to detect -- the instrument-examined-nothing failure, which this directory exists to
    # prevent and has already shipped twice today (lint with 0 gates, privacy guard with 0 lines).
    if not got.strip() or not ref.strip():
        print(f"  output-sanity VOID: an arm produced NO TEXT "
              f"(reference {len(ref)} chars, sample {len(got)} chars). Nothing was graded.")
        return 2

    bad, reasons = degeneracy(got)
    refbad, refreasons = degeneracy(ref)

    # ⚠ GRADE THE REFERENCE TOO. An error-only probe says what broke, never what it should have been:
    # if the STATIC arm is also degenerate the problem is the prompt, the model or the harness, and
    # blaming the paged path would be a wrong attribution with a real fix attached to it.
    if refbad:
        print("  output-sanity VOID: the STATIC REFERENCE is itself degenerate -- "
              + "; ".join(refreasons))
        print("    A paged verdict from a broken reference is not a verdict. Fix the reference arm first.")
        return 2

    cp = common_prefix(ref, got)
    same = (ref == got)
    print(f"  output-sanity: ref {len(ref)} chars, sample {len(got)} chars, "
          f"common prefix {cp} ({100.0*cp/max(len(ref),1):.0f}%)")

    if bad:
        print("  ⇒ **DEGENERATE**: " + "; ".join(reasons))
        print("    This is the FINDINGS-paged-cross-request signature: bookkeeping clean, output wrong.")
        print("    Keep the server log -- every existing marker passes on this failure, so the log is")
        print("    only useful when paired with the sample that proves the run was bad.")
        print(f"    sample: {got[:160]!r}")
        return 1

    if same:
        print("  ⇒ OK: byte-identical to the static reference.")
        return 0

    print("  ⇒ DIVERGENT (reported, NOT failed): text differs but shows no degeneracy signature.")
    print("    ⚠ Static-vs-static over 200 tokens diverges 67% of the time on this box, so difference")
    print("      alone is not evidence of a defect. Grade the top-2 logprob gap at the first")
    print("      differing token before calling it anything (gate_grade_divergence in _gate_common.sh).")
    print(f"    ref@{cp}: {ref[cp:cp+80]!r}")
    print(f"    got@{cp}: {got[cp:cp+80]!r}")
    return 3

if __name__ == "__main__":
    sys.exit(main(sys.argv))
