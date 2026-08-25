/*
 * Host self test for the people counting logic.
 * It includes the *real* firmware headers (sscma/extension/counter/*.hpp), so
 * the line side test, the point in polygon test, the tracker association and
 * the counting state machine under test are byte for byte the code that ships.
 */
#include <cstdio>
#include <forward_list>

#include "sscma/extension/counter/pc_counter.hpp"

using namespace sscma::extension::counter;

static const int W = 240, H = 240;

static el_box_t mk(int cx, int cy, int w, int h) {
    el_box_t b{};
    b.x = (uint16_t)cx; b.y = (uint16_t)cy; b.w = (uint16_t)w; b.h = (uint16_t)h;
    b.score = 80; b.target = 0;
    return b;
}

static int failures = 0;
static void check(const char* what, long got, long want) {
    bool ok = got == want;
    if (!ok) ++failures;
    printf("  [%s] %-46s got=%ld want=%ld\n", ok ? "PASS" : "FAIL", what, got, want);
}

int main() {
    /* ---------------- 1. pure geometry ---------------- */
    printf("1) geometry primitives\n");
    /* line direction (x1,y1)->(x2,y2); positive side = left of the direction
     * vector. For a left-to-right line that is the top half of the frame, so a
     * top->bottom walk is positive->negative = count_ba = "out". */
    {
        pc_line_t l{}; l.x1 = 0; l.y1 = 120; l.x2 = 240; l.y2 = 120; l.enabled = 1;
        check("pc_side above the line (y=10) is positive",  pc_side(l, 120, 10),  1);
        check("pc_side below the line (y=230) is negative", pc_side(l, 120, 230), -1);
        check("pc_side exactly on the line",    pc_side(l, 120, 120),  0);

        int16_t px[4] = {60, 180, 180, 60};
        int16_t py[4] = {60, 60, 180, 180};
        check("in poly, center",        pc_point_in_poly(px, py, 4, 120, 120), 1);
        check("in poly, outside left",  pc_point_in_poly(px, py, 4,  30, 120), 0);
        check("in poly, outside above", pc_point_in_poly(px, py, 4, 120,  30), 0);
        check("in poly, outside right", pc_point_in_poly(px, py, 4, 220, 120), 0);
    }

    /* ---------------- 2. one target crossing a line downwards ---------------- */
    printf("2) single target crossing the horizontal mid line top -> bottom\n");
    PcCounter& c = pc_counter();
    c.defaults();
    /* horizontal line at y=500/1000 spanning the frame, left to right */
    c.set_line(0, 0, 500, 1000, 500);
    for (int y = 20; y <= 220; y += 10) {
        std::forward_list<el_box_t> boxes;
        boxes.push_front(mk(120, y, 40, 40));
        c.update(boxes, W, H);
    }
    check("count_ab (in)  after one downward pass", c.line(0).count_ab, 0);
    check("count_ba (out) after one downward pass", c.line(0).count_ba, 1);

    printf("3) the same target walking back bottom -> top\n");
    for (int y = 220; y >= 20; y -= 10) {
        std::forward_list<el_box_t> boxes;
        boxes.push_front(mk(120, y, 40, 40));
        c.update(boxes, W, H);
    }
    check("count_ab (in)  after the return pass", c.line(0).count_ab, 1);
    check("count_ba (out) unchanged",             c.line(0).count_ba, 1);

    /* ---------------- 4. entering and leaving a region ---------------- */
    printf("4) single target entering and leaving a quadrilateral region\n");
    c.defaults();
    {
        /* square from (250,250) to (750,750) in normalised units */
        const int16_t roi[8] = {250, 250, 750, 250, 750, 750, 250, 750};
        c.set_roi(0, roi);
    }
    long cur_inside_peak = 0;
    for (int x = 10; x <= 230; x += 10) {
        std::forward_list<el_box_t> boxes;
        boxes.push_front(mk(x, 120, 40, 40));
        c.update(boxes, W, H);
        if (c.roi(0).current > cur_inside_peak) cur_inside_peak = c.roi(0).current;
    }
    check("entered after one pass through the region", c.roi(0).enter_count, 1);
    check("peak occupancy during the pass",            cur_inside_peak,      1);
    check("occupancy after leaving",                   c.roi(0).current,     0);

    printf("5) the same target re-enters the region\n");
    for (int x = 230; x >= 10; x -= 10) {
        std::forward_list<el_box_t> boxes;
        boxes.push_front(mk(x, 120, 40, 40));
        c.update(boxes, W, H);
    }
    check("entered after the second pass", c.roi(0).enter_count, 2);

    /* ---------------- 6. two targets, opposite directions ---------------- */
    printf("6) two targets crossing the line in opposite directions\n");
    c.defaults();
    c.set_line(0, 0, 500, 1000, 500);
    for (int k = 0; k <= 20; ++k) {
        std::forward_list<el_box_t> boxes;
        boxes.push_front(mk(80,  20 + k * 10, 40, 40));   /* downwards */
        boxes.push_front(mk(180, 220 - k * 10, 40, 40));  /* upwards   */
        c.update(boxes, W, H);
    }
    check("count_ab (in)",  c.line(0).count_ab, 1);
    check("count_ba (out)", c.line(0).count_ba, 1);

    /* ---------------- 7. a target that stops on the line does not count ------ */
    printf("7) a target that walks onto the line and back does not count\n");
    c.defaults();
    c.set_line(0, 0, 500, 1000, 500);
    for (int y = 20; y <= 120; y += 10) {
        std::forward_list<el_box_t> boxes;
        boxes.push_front(mk(120, y, 40, 40));
        c.update(boxes, W, H);
    }
    for (int y = 120; y >= 20; y -= 10) {
        std::forward_list<el_box_t> boxes;
        boxes.push_front(mk(120, y, 40, 40));
        c.update(boxes, W, H);
    }
    check("count_ab", c.line(0).count_ab, 0);
    check("count_ba", c.line(0).count_ba, 0);

    /* ---------------- 8. identity is kept across a detection drop ----------- */
    printf("8) track identity survives dropped detections (max_miss=8)\n");
    c.defaults();
    c.set_line(0, 0, 500, 1000, 500);
    int32_t first_id = -1, last_id = -1;
    for (int k = 0; k < 24; ++k) {
        std::forward_list<el_box_t> boxes;
        const int y = 20 + k * 9;
        const bool dropped = (k == 10 || k == 11); /* detector misses two frames */
        if (!dropped) boxes.push_front(mk(120, y, 40, 40));
        c.update(boxes, W, H);
        if (!dropped) {
            if (first_id < 0 && c.box_track_id(0) > 0) first_id = c.box_track_id(0);
            if (c.box_track_id(0) > 0) last_id = c.box_track_id(0);
        }
    }
    check("same track id before and after the gap", last_id, first_id);
    check("still exactly one crossing",             c.line(0).count_ba, 1);

    /* ---------------- 9. counts json ---------------- */
    printf("9) serialised counts field\n  %s\n", c.counts_json());

    printf("\n%s (%d failing check(s))\n", failures ? "FAILED" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
