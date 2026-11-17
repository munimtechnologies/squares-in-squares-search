# squares-in-squares-search

Open search for dense packings of n unit squares in the smallest square, by MV2 of Munim, Inc.
The code and text were made with AI agents working in his name. Licence: Apache-2.0 (code),
CC BY 4.0 (text and result tables).

## Why
Packing n unit squares into the smallest square is a classic open problem: only a few dozen n are
proven optimal, and the best known side lengths s(n) (kept by David Ellsworth at
https://kingbird.myphotos.cc/packing/squares_in_squares.html, formerly Erich Friedman's page)
keep improving: more than 20 records for n <= 310 changed in 2026 alone. Kaggle's
Squares in Squares competition (n = 1 to 1,000, $50,000) made it a race. The aim here is a new
best-known packing for some n, found by an open method that anyone can re-run.

## What is here
- `sqpack.py`: a from-scratch optimizer (separating-line penalty, L-BFGS-B continuation, strict
  floating-point legalisation, basin hopping) and a verifier.
- `test_sqpack.py`: tests (gradient vs finite differences, verifier, a 2-square run).
- `PREREGISTRATION-sq1.md`: the first pre-registered run (calibration on 10 n with known records,
  then 18 target n), with its gate and predictions written before the code ran.
- `wait_public.py`: the run only starts once the pre-registration is public here.

Run: `pip install -r requirements.txt && python -m unittest -v test_sqpack && python sqpack.py --out-dir out`.

## Status
- 2026-11-17: code and pre-registration sq1 published; the run is queued. No results yet.

## Honest limits
- The verifier is floating point (gaps must exceed 1e-12). Any candidate record will be re-checked
  in exact arithmetic before it is called a record.
- MV2 is not currently a Kaggle entrant. If he enters, this code will also be shared on the
  competition's Kaggle forum, as Kaggle's public code-sharing rule requires.
