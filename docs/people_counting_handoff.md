# 人流计数固件 — 交接文档

**日期**：2026-08-25
**硬件**：Seeed Grove Vision AI Module V2（Himax WE2 / Cortex-M55 + Ethos-U55）
**状态**：功能链路已在真机跑通；**真人计数精度尚未验证**（镜头被挡，见 §5）

---

## 1. 代码在哪

| 位置 | 内容 |
|---|---|
| `EPII_CM55M_APP_S/makefile` | `APP_TYPE = people_counting` |
| `EPII_CM55M_APP_S/app/scenario_app/people_counting/` | **独立 scenario app**，自带 `.mk` / linker / README |
| `EPII_CM55M_APP_S/app/scenario_app/sscma/linker/grove.ld` | linker 修复（`sscma` app 自身的 bug，独立于本功能） |
| `EPII_CM55M_APP_S/library/sscma_micro/`（submodule） | 分支 **`feature/people-counting`** |
| `model_zoo/swift_yolo_person_192_padded.tflite` | 检测模型（已补 padding，见 §6） |
| `tools/people_counting/` | 验证脚本 |
| `docs/people_counting_handoff.md` | 本文档 |

### submodule 里的改动（分支 `feature/people-counting`）

```
A  sscma/extension/counter/pc_tracker.hpp   273 行  IoU 贪心跟踪 + 生命周期
A  sscma/extension/counter/pc_counter.hpp   449 行  越线/区域计数 + 归一化换算 + JSON
A  sscma/callback/counter.hpp               172 行  KV 持久化 + 7 条 AT 命令
M  sscma/callback/invoke.hpp                :463    推理后调 pc_on_results()
M  sscma/utility.hpp                        :185    boxes 追加第 7 字段 track_id
                                            :409    追加 counts 字段
M  sscma/main_task.hpp                      :301-323 注册命令 + 开机加载配置
```

### 编译期隔离

计数代码全部包在 `#ifdef SSCMA_PEOPLE_COUNTING` 里，该宏**只在** `people_counting.mk:4` 定义。
`sscma` / `sscma_face` 不受影响（已用 `nm` 验证产物里无 `PcTracker` / `pc_counter` / `counter_cmd` 符号）。

| APP_TYPE | ROM | 说明 |
|---|---|---|
| `people_counting` | 259,272 B (98.90%) | 含计数 |
| `sscma` | 253,464 B (96.69%) | 不含，差 5,808 B |

> ⚠️ **代码全部未提交。** 接手第一件事：在 submodule 里 commit 到 `feature/people-counting`，
> 并把主仓库的 submodule 指针、`makefile`、新 app 目录一起提交。

> ⚠️ **一个 workaround**：`app/main.c:316` 用 `#ifdef SSCMA` 选入口且硬编码 `#include "sscma.h"`。
> 新 app 里放了一行 shim `people_counting/sscma.h` 转发到 `people_counting.h`。
> 正规做法是给 `app/main.c` 加一个 `SSCMA_PEOPLE_COUNTING` 分支——但那是所有 app 共享的文件，本次没动。

---

## 2. 构建

```bash
cd EPII_CM55M_APP_S
export PATH="/bin:/usr/bin:/opt/homebrew/bin:$PATH"   # macOS：避开 ~/.rpty 的 bash 包装脚本
# makefile 里 APP_TYPE = people_counting
TARGET=GROVE_VISION_AI_V2 gmake clean
TARGET=GROVE_VISION_AI_V2 gmake -j8
```

**必须带 `TARGET=GROVE_VISION_AI_V2`**（`library.mk:111-117` 默认是 `SENSECAP_A1102`）。
macOS 用 `gmake` 不是 `make`。

### linker 修复（不改就编不出来）

`app/scenario_app/sscma/linker/grove.ld`：
```
CM55M_S_EL_ALLOC : LENGTH = 0x34200000 - 0x34054000   // 1712 KB
                            ^^^^^^^^^^ 原来是 0x3416A000（1112 KB）
```
原因：`sscma_micro/porting/himax/we2/el_config_porting.h:53` 的
`CONFIG_SSCMA_TENSOR_ARENA_SIZE = 1110*1024`，光 arena 就 1110KB，1112KB 的区域装不下。
`sscma_face/linker/grove.ld` 早就是 1712KB，只是没回灌给 `sscma`。

### 生成镜像
```bash
cd ../we2_image_gen_local
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen_macOS_arm64 project_case1_blp_wlcsp.json
# 产物 output_case1_sec_wlcsp/output.img
```

---

## 3. 烧录

```bash
uv run --with pyserial --with xmodem python xmodem/xmodem_send.py \
  --port=/dev/cu.wchusbserial<你的序列号> --baudrate=921600 --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="model_zoo/swift_yolo_person_192_padded.tflite 0x700000 0x00000" \
  --model="<任意小文件> 0x980000 0x00000"      # 末尾牺牲品，见 §6
```

**踩坑（全部实测，别重复踩）**
- 模型必须用**补过 padding** 的版本，否则尾部约 6KB 传不进去（§6）
- `xmodem_send.py` 的地址**必须带 `0x`**；而 **AT 命令的参数必须十进制**（解析器遇 `'x'` 静默截断）
- **model-only 刷写不支持**，必须 `--file` + `--model` 同一 session
- 固件传输可能停在 99.x% 的 EOT 握手，**但已提交成功**，别据此判失败
- 传输要几分钟，**日志真正冻结 >5 分钟才判失败**
- **同一时刻只能有一个进程碰串口**，并发会报 `multiple access on port`
- `cu.usbmodem*` 和 `cu.wchusbserial*` 是**同一物理口的两个节点**，任选其一

---

## 4. AT 命令

坐标一律**归一化 0..1000**（不是像素，换分辨率不失效）；**参数必须十进制**；
参数个数必须与声明完全一致（解析器 `_argc = 逗号数 + 1`），所以 ROI 固定四边形。

```
AT+CNTLINE=<idx>,<x1>,<y1>,<x2>,<y2>                        配置越线（idx 0-3）
AT+CNTLINE?
AT+CNTROI=<idx>,<x1>,<y1>,<x2>,<y2>,<x3>,<y3>,<x4>,<y4>     配置区域（四边形，idx 0-3）
AT+CNTROI?
AT+CNTCFG=<iou_q10>,<max_miss>,<min_hits>,<anchor_mode>     跟踪参数
AT+CNTCFG?
AT+CNTRST                                                    清零计数（不清配置）
```
禁用某条线/区域：坐标全传 0。配置存 FlashDB KV，**重启不丢**（已验证）。

检测阈值用 sscma 自带的 `AT+TSCORE=<0-100>` / `AT+TIOU=<0-100>`（默认 50 / 45），
**不要**再造一套。

### 输出格式（向后兼容）
```json
{"type":1,"name":"INVOKE","code":0,"data":{
  "count":12, "perf":[7,48,0],
  "boxes":[[x,y,w,h,score,cls,track_id]],
  "counts":{"lines":[{"id":0,"in":14,"out":11}],
            "rois":[{"id":0,"cur":3,"entered":22}]},
  "resolution":[240,240]}}
```
- `boxes` 前 6 个字段**顺序含义未变**，`track_id` 是**追加**的第 7 个。
  已核对 Seeed 官方 ESP32 组件 `sscma-client`（`sscma_client_ops.c:1394-1399`）按下标读 0..5，
  多出的第 7 个会被忽略 → **ESP32 侧不改也能跑**，要用 track_id 加一行读 index 6 即可。
- `track_id = -1` 表示该框还是 TENTATIVE（未达 `min_hits`），不参与计数。
- **`boxes` 的 x,y 是框中心，不是左上角**（`core/algorithm/el_algorithm_yolo.cpp:193`）。

---

## 5. 已验证 / 未验证

### 已验证（真机）
| 项 | 结果 |
|---|---|
| 7 条 AT 命令 | 全部正常 |
| KV 持久化 | `AT+RST` 后配置完整存活 |
| 推理性能 | `perf [6-7, 48-49, 0]` ms → **13 fps 实测**（1182 帧 / 91 s） |
| 跟踪开销 | 加功能前后 `perf` 无变化 |
| 状态机 | 单帧检出给 `track_id=-1` 不计数（正确） |
| 稳定性 | 连续 90 s 无崩溃/重启 |
| 算法逻辑 | 两套独立自测全通过（见 `tools/people_counting/pc_selftest.{py,cpp}`）：<br>单目标双向穿线、走到线上退回不计数、进出区域、重复进入、双目标反向对穿、<br>丢失 8 帧后 track_id 不变且只计一次、240×240 全像素多边形判定 0 失配 |

### ⚠️ 未验证（接手要做的）
1. **真人穿越计数** —— 最后一次 90 秒采集时**镜头被挡**（抓图确认几乎全黑），
   1182 帧只检出 1 个框。**功能链路是通的，精度一个数都没有。**
2. **吊顶俯视下的 person 召回** —— 当前模型 Swift-YOLO 是常规视角数据训的，
   俯视是分布外，召回可能大幅下降。这是整个方案最大的未知数。
3. **跟踪参数** —— `iou_q10=307 / max_miss=8 / min_hits=3` 全是纸面默认值，未用真实语料标定。
4. **ESP32 侧转发** —— 板上挂的 ESP32 如何把 `counts` 上行，未做。

---

## 6. 模型

```
model_zoo/swift_yolo_person_192_padded.tflite      1,699,920 B（含 16KB padding）
来源: https://files.seeedstudio.com/sscma/model_zoo/detection/person/swift_yolo_nano_person_192_int8_vela.tflite
      （Seeed-Studio/sscma-model-zoo → detection/person/swift_yolo_nano_192.json）
```
Swift-YOLO Nano，**person 单类**，192×192 输入，mAP 92.6% (INT8)，Vela 3.9.0，
输出 `[1,2268,6]` = `[x,y,w,h,score,cls]`，被 sscma 识别为 **algorithm type 3 (YOLO)**。

### 为什么要 padding
xmodem 刷模型时**稳定在距文件末尾约 6KB（47~48 个 128B 块）处卡死**，
同一文件跨运行可复现，且**即使该模型不在 session 末位也照卡**；固件传输从不出现。
补 16KB 零字节后，真实数据在卡死点之前传完，模型按 header 里的 size 加载，忽略尾部多余字节。
实测补后传到 99.95%（13275/13281），真实数据 100% 完整。

### 换模型注意
sscma **不看 magic，看输出张量形状**运行时探测（`el_algorithm_delegate.cpp:39-62`）。
- `[1,N,5+C]` → YOLO（type 3）✅
- `[1,BC,ibox_len]` 且 5≤BC≤84 → YOLOV8
- **`model_zoo/tflm_yolov8_od/yolov8n_od_192_delete_transpose` 不可用** ——
  输出被拆成 `[1,4,756]`+`[1,756,80]` 两个张量（为 `tflm_yolov8_od` 的自定义后处理改的），
  sscma 识别不了，`AT+INVOKE` 返回 code 5。
- `model_zoo/tflm_yolo11_od/*_nopost` 同样不可用（3 个原始 grid 头）。

---

## 7. ⚠️ ROM 只剩 1.1%，下个功能大概率撞墙

```
CM55M_S_APP_ROM:  259272 B / 256 KB = 98.90%
```
加计数功能**之前**基线就是 **99.50%**（只剩 1312 B）。
为腾空间，`sscma/callback/invoke.hpp` 和 `sscma/utility.hpp` 顶部加了
`#pragma GCC optimize("Os")`（行为中性），净结果比加功能前还小 1560 B。
**这是应急手段，瘦身之后应该撤掉。**

### 已量化的瘦身空间（`.map` 实测，总计 60~75 KB）

| 项 | 预计 | 风险 | 做法 |
|---|---|---|---|
| A. 砍 7 个用不到的算法 | 30~40 KB | 低 | `invoke.hpp:202-282` 和 `el_algorithm_delegate.cpp:39-104` 用 `#if` 圈掉；只留 `AlgorithmYOLO`。**没有现成配置宏**，要改代码 |
| B. 砍 FatFS | ~20 KB | 低 | `sscma.mk` 的 `LIB_SEL` 去掉 `fatfs`（`ff.o` 单个 19,037 B） |
| C. 砍 WiFi/MQTT | 6~10 KB | 低 | Grove Vision V2 无网络模组（`CONFIG_EL_NETWORK_SPI_AT` 指外接 AT 模组） |
| D. 砍 ACTION 表达式解释器 | 4~6 KB | 低 | `set_action` + Lexer/Parser |
| E. 裁 TFLite 算子 | ~18 KB | **高** | `el_config_porting.h:56-69` 有现成宏。`MEAN`(reduce_common 9,051B)、`BATCH_MATMUL`(8,941B) 最大。**Vela 全 NPU 理论上用不到，但有算子 fallback 就跑不了，必须逐个删逐个真机验** |

`sscma.o` 单个文件占全部 `.text/.rodata` 的 24%（157,280 / 655,790 B），
其中「模型/算法委派」类符号 64,280 B —— 就是 A 的来源（模板按算法类型逐个实例化）。

---

## 8. 验证脚本

```bash
cd /Users/harvest/project/grove_vision_2/sscma-example-we2
uv run --with pyserial python tools/people_counting/grab_frame.py          # 抓一帧 JPEG，先确认镜头对着场景
uv run --with pyserial python tools/people_counting/verify_at_commands.py  # 7 条 AT 命令 + 持久化
uv run --with pyserial python tools/people_counting/capture_counting.py    # 90s 计数采集 + 统计
uv run python tools/people_counting/pc_selftest.py                         # 纯逻辑自测，不需要设备
```
**脚本里的串口路径写死了，接手先改成自己的。**

> **每次实测前先跑 `grab_frame.py`。** 上一轮就是因为没先看画面，白跑了 90 秒。

---

## 9. 建议的下一步顺序

1. **提交代码**（submodule 分支 + 主仓库指针）
2. **`grab_frame.py` 确认视野** → 平视跑一遍 `capture_counting.py`，拿到第一组真人计数数据
3. **ROM 瘦身 A→D**，每步单独编译记录 Memory region 表；完成后撤掉两处 `#pragma Os`
4. **吊顶俯视实测** —— 召回是关键指标。如果崩了，换**头部检测**模型
   （SCUT-HEAD / Brainwash / HollywoodHeads，或 Roboflow 上的 overhead 数据集）。
   俯视遮挡少，IoU 关联反而更容易，难点全在检测这一层。
5. **用真实语料调 tracker 参数**
6. **ESP32 侧转发**：本地已有 `~/project/esp32s3/sscma-example-esp32`（Seeed 官方），
   `examples/sscma_client_monitor` 是监听示例（UART1 @921600），
   `sscma_client_proxy` 是转发示例。注意示例里 `TX=21/RX=20/reset=GPIO5` 是 XIAO ESP32S3 编号，
   实际接线要按板子改。

---

## 10. 关键决策记录（避免重复论证）

- **不迁 SSCMA-Micro 2.0**：2.0 的 `extension/bytetrack` + `extension/counter` 是**死代码**
  （`BYTETRACK_SSCMA_SOURCES`/`COUNTER_SSCMA_SOURCES` 从未赋值、Eigen 从未 include、AT 层零调用）；
  且 2.0 的空 AT 骨架就要 330KB `.text`，超 256KB ROM 区 66KB。评估副本在 `~/project/grove_vision_2/sscma2_eval/`。
- **不上 ByteTrack / 卡尔曼**：依赖 Eigen，1MB 固件 + 256KB ROM 装不下。俯视基本无遮挡，IoU 贪心足够。
- **越线判定逻辑抄自 2.0 的 `counter.cpp`**（叉积判 side + side 翻转），但数据结构重写为定长静态数组。
- **实现是 header-only**：`sscma.mk` 的 `LIB_SSCMA_MICRO_DIR` 逐个列举编译目录，不含 `extension/`。
  新 app 的 `people_counting.mk` 已把 `sscma_micro/sscma/extension/counter` 加进该列表，
  所以**后续可以正常写 `.cpp`**，不必再受 header-only 限制。
- **`gate_dist` 写死 `width/4`**（240 下 60px）不可配，因为 AT 解析器要求参数个数固定，`AT+CNTCFG` 没位置。
