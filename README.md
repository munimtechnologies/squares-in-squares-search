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
- 2026-11-17 (later): a prior-art check before planning a second run (sq2) found that both
  planned sq2 directions are already taken:
  - **Excess-two packings.** Kevin Fang's note "Packing n^2 - n + 1 unit squares in a square of
    side less than n, for every n >= 15" (draft v0.9, 2026-10-18, doi:10.5281/zenodo.23463481,
    certificates at https://github.com/TheSnakeFang/squarepack-certs) lists s(n^2 - n + 2) < n
    as known for every n >= 18 except n = 24, 25, 26, 27 and 30. Packings below side n for all
    five of those counts (554, 602, 652, 704 and 872 squares) were submitted on 2026-10-19 in
    the Squares Project's registration issue https://github.com/jlevy/squares/issues/686 (Darius
    Muglich), which also cites earlier reference packings for each. Registration there is pending.
  - **Old Goebel-pattern records at n = 226-299.** Joost de Winter's packings were adopted as the
    Squares Project's best at n = 227 and 260 (jlevy/squares PR #704, 2026-10-20), and large
    tables of packings for n > 324 have been submitted by several groups (jlevy/squares #686,
    #719).
  - The Squares Project (https://github.com/jlevy/squares, squaremath.org) is now the registry
    for new results on this problem. Construction records here are a crowded race among
    well-equipped, AI-assisted groups, so sq2 will not be started on those targets.
- sq1 still runs as a calibration of the method. Its result will be published here whatever it
  shows, graded against the pre-registration.

## Honest limits
- The verifier is floating point (gaps must exceed 1e-12). Any candidate record will be re-checked
  in exact arithmetic before it is called a record.
- MV2 is not currently a Kaggle entrant. If he enters, this code will also be shared on the
  competition's Kaggle forum, as Kaggle's public code-sharing rule requires.
