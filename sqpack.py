"""sqpack: search for dense packings of n unit squares in the smallest square container.

MV2 of Munim, Inc. (made with AI agents working in his name). Apache-2.0.

Method (from scratch, no third-party packing code):
  * each square i has centre (x_i, y_i) and angle th_i; the container is [0, L]^2;
  * every pair (i, j) carries a separating line with unit normal (cos phi, sin phi) and offset t;
    the penalty sum max(0, u.a - t)^2 over corners a of i plus sum max(0, t - u.b)^2 over corners b
    of j is zero exactly when the line separates the two (convex) squares;
  * walls add sum max(0, -c)^2 + max(0, c - L)^2 over corner coordinates c;
  * minimise L + mu * penalty with L-BFGS-B for mu = 1e1 .. 1e10 (continuation), then
    "legalise": scale centres apart by (1 + d) for the smallest d in 0, 1e-13 .. 1e-3 that makes a
    strict check pass (every close pair separated by a gap > margin on some face normal, all corners
    inside the box);
  * basin hopping: relocate 1-3 squares, jiggle all, or swap two, re-solve, keep if L shrinks.
Limits: the verifier is floating point (gaps must exceed 1e-12); any record claim needs an
independent exact check before it is reported anywhere.
"""
import argparse
import json
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

VX = np.array([0.5, -0.5, -0.5, 0.5])
VY = np.array([0.5, 0.5, -0.5, -0.5])

MUS = [10.0 ** k for k in range(1, 11)]
MUS_HOP = [10.0 ** k for k in range(3, 11)]

# Best known side lengths, copied 2026-11-17 from David Ellsworth's Squares in Squares page
# https://kingbird.myphotos.cc/packing/squares_in_squares.html (values as displayed there).
RECORDS = {
    5: 2.70710678118654, 10: 3.70710678118654, 11: 3.87708359002281, 17: 4.67553009360455,
    18: 4.82287565553229, 19: 4.88561808316412, 26: 5.62132034355964, 27: 5.70710678118654,
    28: 5.82444461667405, 29: 5.93383346267692, 37: 6.59861960924436, 39: 6.81072208306864,
    41: 6.92669309446880, 50: 7.57142857142857, 51: 7.70079923541701, 53: 7.82287565553229,
    54: 7.84666719284348, 55: 7.94577100750391, 66: 8.65685424949238, 68: 8.7987961402601,
    69: 8.82719465572973, 70: 8.88166675700900, 71: 8.94407155757031, 83: 9.63475764863108,
    85: 9.74264068711928, 86: 9.82287565553229, 87: 9.83881526994826, 88: 9.88815305375857,
}
CALIBRATION = [5, 10, 11, 17, 18, 19, 26, 27, 28, 29]
TARGETS = [37, 39, 41, 50, 51, 53, 54, 55, 66, 68, 69, 70, 71, 83, 85, 86, 87, 88]
REPRODUCE_TOL = 1e-6
RECORD_MARGIN = 1e-9
GATE_MIN_REPRODUCED = 4


def offsets(th):
    c = np.cos(th)[:, None]
    s = np.sin(th)[:, None]
    return c * VX - s * VY, s * VX + c * VY


def corners(x, y, th):
    ox, oy = offsets(th)
    return x[:, None] + ox, y[:, None] + oy


def all_pairs(n):
    return np.triu_indices(n, 1)


def init_lines(x, y, th, I, J):
    """For each pair pick the face normal (of either square, either direction) that best separates
    them; return its angle phi and the mid-gap offset t (an exact separator if one exists)."""
    cx, cy = corners(x, y, th)
    P = len(I)
    best_sep = np.full(P, -np.inf)
    best_phi = np.zeros(P)
    best_t = np.zeros(P)
    for base in (th[I], th[J]):
        for extra in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
            phi = base + extra
            ux = np.cos(phi)[:, None]
            uy = np.sin(phi)[:, None]
            hi = (ux * cx[I] + uy * cy[I]).max(1)
            lo = (ux * cx[J] + uy * cy[J]).min(1)
            sep = lo - hi
            better = sep > best_sep
            best_sep = np.where(better, sep, best_sep)
            best_phi = np.where(better, phi, best_phi)
            best_t = np.where(better, 0.5 * (hi + lo), best_t)
    return best_phi, best_t


def fg(z, n, I, J, mu):
    """Objective L + mu * penalty and its exact gradient.
    z = [x(n), y(n), th(n), L, phi(P), t(P)]."""
    P = len(I)
    x = z[:n]
    y = z[n:2 * n]
    th = z[2 * n:3 * n]
    L = z[3 * n]
    phi = z[3 * n + 1:3 * n + 1 + P]
    t = z[3 * n + 1 + P:3 * n + 1 + 2 * P]
    ox, oy = offsets(th)
    cx = x[:, None] + ox
    cy = y[:, None] + oy
    ux = np.cos(phi)[:, None]
    uy = np.sin(phi)[:, None]
    cxi, cyi, cxj, cyj = cx[I], cy[I], cx[J], cy[J]
    vi = np.maximum(ux * cxi + uy * cyi - t[:, None], 0.0)
    vj = np.maximum(t[:, None] - (ux * cxj + uy * cyj), 0.0)
    bx = np.maximum(-cx, 0.0)
    ax = np.maximum(cx - L, 0.0)
    by = np.maximum(-cy, 0.0)
    ay = np.maximum(cy - L, 0.0)
    E = ((vi * vi).sum() + (vj * vj).sum() + (bx * bx).sum() + (ax * ax).sum()
         + (by * by).sum() + (ay * ay).sum())
    gi = 2.0 * vi
    gj = -2.0 * vj
    gt = -gi.sum(1) - gj.sum(1)
    gphi = ((gi * (-uy * cxi + ux * cyi)).sum(1) + (gj * (-uy * cxj + ux * cyj)).sum(1))
    gcx = 2.0 * (ax - bx)
    gcy = 2.0 * (ay - by)
    gL = -2.0 * (ax.sum() + ay.sum())
    gix = gi * ux
    giy = gi * uy
    gjx = gj * ux
    gjy = gj * uy
    for k in range(4):
        gcx[:, k] += (np.bincount(I, weights=gix[:, k], minlength=n)
                      + np.bincount(J, weights=gjx[:, k], minlength=n))
        gcy[:, k] += (np.bincount(I, weights=giy[:, k], minlength=n)
                      + np.bincount(J, weights=gjy[:, k], minlength=n))
    gx = gcx.sum(1)
    gy = gcy.sum(1)
    gth = (gcx * (-oy) + gcy * ox).sum(1)
    f = float(L + mu * E)
    g = np.concatenate([mu * gx, mu * gy, mu * gth, [1.0 + mu * gL], mu * gphi, mu * gt])
    return f, g


def solve(x, y, th, L, mus, I, J, maxiter=1500):
    from scipy.optimize import minimize
    n = len(x)
    P = len(I)
    bounds = [(None, None)] * (3 * n) + [(1.0, None)] + [(None, None)] * (2 * P)
    for mu in mus:
        phi, t = init_lines(x, y, th, I, J)
        z0 = np.concatenate([x, y, th, [L], phi, t])
        r = minimize(fg, z0, args=(n, I, J, mu), jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-12, "maxcor": 30})
        z = r.x
        x = z[:n].copy()
        y = z[n:2 * n].copy()
        th = z[2 * n:3 * n].copy()
        L = float(z[3 * n])
    return x, y, th, L


def verify(x, y, th, L, margin=0.0):
    """Strict check: all corners in [0, L]^2 and every pair of squares whose centres are closer than
    sqrt(2) has a gap > margin along one of the four face normals (separating axis theorem)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    th = np.asarray(th, dtype=float)
    n = len(x)
    cx, cy = corners(x, y, th)
    wall = float(min(cx.min(), cy.min(), L - cx.max(), L - cy.max()))
    I, J = np.triu_indices(n, 1)
    d2 = (x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2
    keep = d2 < 2.0 + 1e-6
    I = I[keep]
    J = J[keep]
    if len(I):
        gap = np.full(len(I), -np.inf)
        for base in (th[I], th[J]):
            for extra in (0.0, 0.5 * np.pi):
                ux = np.cos(base + extra)[:, None]
                uy = np.sin(base + extra)[:, None]
                pi_ = ux * cx[I] + uy * cy[I]
                pj = ux * cx[J] + uy * cy[J]
                sep = np.maximum(pj.min(1) - pi_.max(1), pi_.min(1) - pj.max(1))
                gap = np.maximum(gap, sep)
        min_gap = float(gap.min())
    else:
        min_gap = float("inf")
    ok = bool(wall >= 0.0 and min_gap > margin)
    return {"ok": ok, "wall": wall, "min_gap": min_gap, "close_pairs": int(len(I))}


def fit_box(x, y, th, margin):
    cx, cy = corners(x, y, th)
    x2 = x - cx.min() + margin
    y2 = y - cy.min() + margin
    L = float(max(cx.max() - cx.min(), cy.max() - cy.min()) + 2.0 * margin)
    return x2, y2, L


def legalize(x, y, th, margin=1e-12):
    xm = float(x.mean())
    ym = float(y.mean())
    for d in [0.0] + [10.0 ** -k for k in range(13, 2, -1)]:
        xs = xm + (x - xm) * (1.0 + d)
        ys = ym + (y - ym) * (1.0 + d)
        x2, y2, L = fit_box(xs, ys, th, margin)
        rep = verify(x2, y2, th, L, margin)
        if rep["ok"]:
            rep["scale_d"] = d
            return x2, y2, th.copy(), L, rep
    return None


def perturb(rng, x, y, th, L):
    x = x.copy()
    y = y.copy()
    th = th.copy()
    n = len(x)
    r = rng.random()
    if r < 0.5 or n < 2:
        k = int(rng.integers(1, min(3, n) + 1))
        idx = rng.choice(n, size=k, replace=False)
        hi = max(L - 0.5, 0.5)
        x[idx] = rng.uniform(0.5, hi, k)
        y[idx] = rng.uniform(0.5, hi, k)
        th[idx] = rng.uniform(0.0, 0.5 * np.pi, k)
    elif r < 0.8:
        s = 0.02 + 0.08 * rng.random()
        x += rng.normal(0.0, s, n)
        y += rng.normal(0.0, s, n)
        th += rng.normal(0.0, s, n)
    else:
        i, j = rng.choice(n, size=2, replace=False)
        x[i], x[j] = x[j], x[i]
        y[i], y[j] = y[j], y[i]
    return x, y, th


def search(n, seconds, seed, record=None, max_stall=8):
    rng = np.random.default_rng(seed)
    I, J = all_pairs(n)
    t0 = time.time()
    best = None
    chains = hops = accepted = legal_fail = 0
    while chains == 0 or time.time() - t0 < seconds:
        chains += 1
        L0 = 1.25 * math.sqrt(n) + 1.0
        x = rng.uniform(0.5, L0 - 0.5, n)
        y = rng.uniform(0.5, L0 - 0.5, n)
        th = rng.uniform(0.0, 0.5 * np.pi, n)
        x, y, th, L = solve(x, y, th, L0, MUS, I, J)
        cur = legalize(x, y, th)
        if cur is None:
            legal_fail += 1
            continue
        if best is None or cur[3] < best[3]:
            best = cur
        stall = 0
        while stall < max_stall and time.time() - t0 < seconds:
            hops += 1
            px, py, pth = perturb(rng, cur[0], cur[1], cur[2], cur[3])
            px, py, pth, pL = solve(px, py, pth, cur[3], MUS_HOP, I, J)
            new = legalize(px, py, pth)
            if new is not None and new[3] < cur[3] - 1e-12:
                cur = new
                accepted += 1
                stall = 0
                if new[3] < best[3]:
                    best = new
            else:
                if new is None:
                    legal_fail += 1
                stall += 1
    trivial = math.ceil(math.sqrt(n) - 1e-12)
    out = {"n": n, "seed": seed, "seconds": round(time.time() - t0, 1), "chains": chains,
           "hops": hops, "accepted_hops": accepted, "legalize_failures": legal_fail,
           "trivial": trivial, "record": record}
    if best is None:
        out.update({"best_L": None, "reproduced": False, "beats_record": False,
                    "beats_trivial": False})
        return out
    bx, by, bth, bL, rep = best
    out.update({
        "best_L": bL,
        "kaggle_style_score": bL / math.sqrt(n) - 1.0,
        "delta_vs_record": (bL - record) if record is not None else None,
        "reproduced": bool(record is not None and bL <= record + REPRODUCE_TOL),
        "beats_record": bool(record is not None and bL < record - RECORD_MARGIN),
        "beats_trivial": bool(bL < trivial - RECORD_MARGIN),
        "verify": rep,
        "packing": {"L": bL, "x": bx.tolist(), "y": by.tolist(), "th": bth.tolist()},
    })
    return out


def _task(args):
    n, seconds, seed = args
    return search(n, seconds, seed, RECORDS.get(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cpu", type=int, default=2)
    ap.add_argument("--cal-seconds", type=float, default=300.0)
    ap.add_argument("--target-seconds", type=float, default=600.0)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    res_dir = os.path.join(a.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    tasks = [(n, a.target_seconds, 1000 + n) for n in TARGETS] + \
            [(n, a.cal_seconds, 1000 + n) for n in CALIBRATION]
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.cpu) as ex:
        futs = {ex.submit(_task, t): t[0] for t in tasks}
        for fu in as_completed(futs):
            n = futs[fu]
            try:
                r = fu.result()
            except Exception as e:  # keep going; report the failure
                r = {"n": n, "error": "%s: %s" % (type(e).__name__, e)}
            results[n] = r
            with open(os.path.join(res_dir, "n%03d.json" % n), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
            print(json.dumps({k: r.get(k) for k in ("n", "best_L", "record", "delta_vs_record",
                                                    "reproduced", "beats_record", "chains",
                                                    "hops", "error")}), flush=True)
    cal_rep = [n for n in CALIBRATION if results.get(n, {}).get("reproduced")]
    new_rec = sorted(n for n, r in results.items() if r.get("beats_record"))
    summary = {
        "prereg": "PREREGISTRATION-sq1.md",
        "wall_seconds": round(time.time() - t0, 1),
        "calibration_reproduced": cal_rep,
        "calibration_reproduced_count": len(cal_rep),
        "gate_G1_pass": len(cal_rep) >= GATE_MIN_REPRODUCED,
        "targets_reproduced": [n for n in TARGETS if results.get(n, {}).get("reproduced")],
        "candidate_new_records_unchecked": new_rec,
        "beats_trivial": sorted(n for n, r in results.items() if r.get("beats_trivial")),
        "errors": sorted(n for n, r in results.items() if r.get("error")),
        "table": [{k: results.get(n, {}).get(k) for k in ("n", "record", "best_L", "delta_vs_record",
                                                         "chains", "hops")}
                  for n in CALIBRATION + TARGETS],
    }
    with open(os.path.join(a.out_dir, "sq1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
