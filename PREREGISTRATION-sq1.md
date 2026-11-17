# Pre-registration sq1: can a from-scratch optimizer reach or beat the best known packings of unit squares in a square?

MV2 of Munim, Inc. Written 2026-11-17 by AI agents working in his name, before any run of the code.
Licence: CC BY 4.0 (this file); code Apache-2.0.

## The question
How small a square holds n non-overlapping unit squares (rotations allowed)? Only a few dozen n are
proven optimal; the best known side lengths s(n) for n up to 324 are kept by David Ellsworth
(https://kingbird.myphotos.cc/packing/squares_in_squares.html, formerly Erich Friedman's page).
Kaggle's Squares in Squares competition (n = 1 to 1,000; scored by the sum of L/sqrt(n) - 1)
has made this a race in 2026. The long-run question for this repo is whether an open, from-scratch
method finds packings smaller than the best known ones. This first run (sq1) is a calibration plus
a first attempt.

## Method (fixed; code in `sqpack.py` at the commit that adds this file)
Penalty method with one separating line per pair of squares, L-BFGS-B continuation
(mu = 1e1 .. 1e10), strict float legalisation (every close pair separated by a gap > 1e-12 on a face
normal; all corners inside the box), basin hopping (relocate 1-3 squares / jiggle all / swap two;
stop a chain after 8 hops without gain). Random starts only: no known packing is used as a seed.
Seed for n is 1000 + n. Two worker processes, one thread each.

- Calibration set C (10): n = 5, 10, 11, 17, 18, 19, 26, 27, 28, 29; 300 s each.
- Target set T (18): n = 37, 39, 41, 50, 51, 53, 54, 55, 66, 68, 69, 70, 71, 83, 85, 86, 87, 88;
  600 s each. T mixes n improved in 2025-2026 (still moving) with older records.
- Records: the values in `sqpack.RECORDS`, copied from Ellsworth's page on 2026-11-17.

## Definitions
- reproduced(n): best legal L <= record + 1e-6.
- candidate(n): best legal L < record - 1e-9. A candidate is NOT a claimed record: before any claim,
  (a) re-read the record on Ellsworth's page that day, (b) re-verify the coordinates independently in
  exact rational arithmetic after rounding, (c) a human-readable figure is checked. Only then may it
  be reported to the page maintainer, in a message reviewed before it is sent.

## Gate and predictions (graded by hand from `/out/sq1_summary.json`)
- G1 (gate): at least 4 of the 10 calibration n are reproduced. If G1 fails, the method is declared
  not competitive and every target number is reported as "no claim", whatever it shows.
- P1: at least 4 of 10 calibration n reproduced (= G1). Our guess: 5 and 10 yes; 11, 17, 18, 19 unlikely
  in 300 s from random starts.
- P2: zero candidates in T. (Records for n <= 88 have been searched hard in 2025-2026 by annealing and
  AI-assisted optimisers, so we expect none.)
- P3: at least 12 of the 18 target n end below the trivial side ceil(sqrt(n)).
- P4: for at least 9 of the 18 target n, best L is within 1% of the record.

## What happens next
If G1 passes: sq2 seeds the optimizer from structured starts and targets n in 100-324 where the
record is a classical pattern that has not been re-optimised. If G1 fails: fix the method
(a different hop set or local polish) and re-run sq1 as sq1b with the same sets, labelled as a re-run.
Every result, including "no candidate", is published here.
