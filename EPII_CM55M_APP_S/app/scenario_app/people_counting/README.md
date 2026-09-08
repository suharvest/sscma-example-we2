# people_counting

Standalone scenario app: SSCMA object detection plus the people counting
extension (IoU tracker + line/region counters).

It is a copy of the `sscma` app with one extra define, `SSCMA_PEOPLE_COUNTING`.
All counting code inside the `sscma_micro` submodule is guarded by that macro,
so `sscma` and `sscma_face` build without any of it.

## What the define changes

| File (in `library/sscma_micro`) | Guarded by `SSCMA_PEOPLE_COUNTING` |
| --- | --- |
| `sscma/main_task.hpp` | `callback/counter.hpp` include, `counter_load_config()`, the seven `AT+CNT*` registrations |
| `sscma/callback/invoke.hpp` | `pc_counter.hpp` include, `pc_on_results()` call |
| `sscma/utility.hpp` | `pc_counter.hpp` include, the 7th `track_id` box field, the `counts` object |

Without the define the `boxes` array goes back to the original six fields
(`x, y, w, h, score, target`) and no `counts` key is emitted.

Sources live in:

```
library/sscma_micro/sscma/extension/counter/pc_tracker.hpp
library/sscma_micro/sscma/extension/counter/pc_counter.hpp
library/sscma_micro/sscma/callback/counter.hpp
```

`sscma/extension/counter` is on this app's `LIB_SSCMA_MICRO_DIR` list, so
adding a `.cpp` there is picked up automatically (the extension is currently
header-only).

## Build

```bash
export PATH="/bin:/usr/bin:/opt/homebrew/bin:$PATH"
cd EPII_CM55M_APP_S
# makefile: APP_TYPE = people_counting
TARGET=GROVE_VISION_AI_V2 gmake clean && TARGET=GROVE_VISION_AI_V2 gmake -j8
```

`TARGET` selects the linker script under `linker/`:
`GROVE_VISION_AI_V2` → `grove.ld`, `SENSECAP_WATCHER` → `watcher.ld`,
`SENSECAP_A1102` → `a1102.ld`. On macOS use `gmake`, not BSD `make`.

## AT commands

All geometry is normalised to 0..1000, so the configuration survives a
resolution change. The AT parser splits on commas and does not accept a `0x`
prefix — use decimal. Send all coordinates as 0 to disable a line or region.

| Command | Arguments |
| --- | --- |
| `AT+CNTLINE=` | `IDX,X1,Y1,X2,Y2` |
| `AT+CNTLINE?` | — |
| `AT+CNTROI=` | `IDX,X1,Y1,X2,Y2,X3,Y3,X4,Y4` (fixed quadrilateral) |
| `AT+CNTROI?` | — |
| `AT+CNTCFG=` | `IOU_Q10,MAX_MISS,MIN_HITS,ANCHOR_MODE` |
| `AT+CNTCFG?` | — |
| `AT+CNTRST` | — (resets counters, keeps configuration) |

Details: `docs/people_counting_handoff.md`.
