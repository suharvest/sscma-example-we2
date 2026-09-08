#!/usr/bin/env python3
"""Independent re-implementation of the two geometry predicates shipped in
sscma/extension/counter/pc_counter.hpp, cross-checked against a float
reference over a dense grid and over the same synthetic trajectories the C++
host harness (pc_selftest.cpp) drives.  Pure logic, no device needed."""

W = H = 240


def pc_side(l, x, y):                       # mirrors pc_side()
    x1, y1, x2, y2 = l
    v = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return 1 if v > 0 else (-1 if v < 0 else 0)


def pc_point_in_poly(px, py, x, y):         # mirrors pc_point_in_poly()
    n = len(px)
    inside = False
    j = n - 1
    for i in range(n):
        if (py[i] > y) != (py[j] > y):
            dy = py[j] - py[i]
            lhs = (x - px[i]) * dy
            rhs = (y - py[i]) * (px[j] - px[i])
            if (lhs < rhs) if dy > 0 else (lhs > rhs):
                inside = not inside
        j = i
    return inside


def poly_ref(px, py, x, y):                 # float reference (Franklin)
    n = len(px)
    inside = False
    j = n - 1
    for i in range(n):
        if (py[i] > y) != (py[j] > y):
            if x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]:
                inside = not inside
        j = i
    return inside


def crossings(line, pts):
    """Line crossing state machine: first side seeds, side==0 holds, sign flip counts."""
    ab = ba = 0
    prev = None
    for (x, y) in pts:
        s = pc_side(line, x, y)
        if s == 0:
            continue
        if prev is None:
            prev = s
            continue
        if s != prev:
            if s > 0:
                ab += 1
            else:
                ba += 1
            prev = s
    return ab, ba


fails = 0


def check(what, got, want):
    global fails
    ok = got == want
    if not ok:
        fails += 1
    print("  [%s] %-52s got=%s want=%s" % ("PASS" if ok else "FAIL", what, got, want))


print("A) point in polygon vs float reference, every pixel of a 240x240 frame")
polys = [
    ([60, 180, 180, 60], [60, 60, 180, 180]),        # axis aligned square
    ([120, 200, 120, 40], [30, 120, 210, 120]),      # diamond
    ([20, 220, 160, 60], [40, 20, 200, 150]),        # irregular quad
    ([60, 180, 180, 60], [180, 180, 60, 60]),        # square, reversed winding
]
mismatch = 0
for px, py in polys:
    for y in range(H):
        for x in range(W):
            if pc_point_in_poly(px, py, x, y) != poly_ref(px, py, x, y):
                mismatch += 1
check("mismatching pixels over 4 polygons", mismatch, 0)

print("B) line side of a left-to-right mid line")
line = (0, 120, 240, 120)
check("above the line is positive", pc_side(line, 120, 10), 1)
check("below the line is negative", pc_side(line, 120, 230), -1)
check("on the line is zero", pc_side(line, 120, 120), 0)

print("C) one point walking top -> bottom through the line")
check("(in, out)", crossings(line, [(120, y) for y in range(20, 230, 10)]), (0, 1))

print("D) then bottom -> top again")
pts = [(120, y) for y in range(20, 230, 10)] + [(120, y) for y in range(220, 10, -10)]
check("(in, out)", crossings(line, pts), (1, 1))

print("E) a point that stops on the line and turns back")
pts = [(120, y) for y in range(20, 130, 10)] + [(120, y) for y in range(120, 10, -10)]
check("(in, out)", crossings(line, pts), (0, 0))

print("F) a point entering and leaving a region")
px, py = [60, 180, 180, 60], [60, 60, 180, 180]
seq = [pc_point_in_poly(px, py, x, 120) for x in range(10, 240, 10)]
enters = sum(1 for a, b in zip([False] + seq, seq) if (not a) and b)
check("edge triggered enters", enters, 1)
check("occupancy at the end", seq[-1], False)

print("\n%s (%d failing check(s))" % ("ALL CHECKS PASSED" if not fails else "FAILED", fails))
raise SystemExit(1 if fails else 0)
