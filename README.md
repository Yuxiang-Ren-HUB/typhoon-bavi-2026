# 台风巴威（2026）WRF模拟

用 WRF-ARW 对 2026 年第 9 号台风"巴威"（Bavi）登陆浙闽沿海过程及其后续 10 天演变的区域数值模拟。
起报时刻该台风尚未登陆，属于实时预报性质的模拟，而非事后个例复盘。

## 背景与目的

巴威登陆前，本次模拟希望回答两个问题：

1. **登陆过程**：精确的登陆时间、地点、强度、大风/降雨落区是什么样？（Run A）
2. **登陆之后**：台风减弱变性后，其残余水汽/环流是否会影响华北、西北内陆
   （北京、石家庄、郑州、济南、兰州）？（Run B）

为兼顾登陆过程的精细结构和延伸期的计算成本，把模拟拆成两组独立运行，而不是用一组超长时间的嵌套模拟。

## Run A / Run B 配置对比

| | Run A（登陆段） | Run B（延伸段） |
|---|---|---|
| 起报时间 | 2026-07-10 18Z | 2026-07-10 18Z（同起报，独立运行） |
| 时效 | 84 小时 | 240 小时（10 天） |
| 网格 | d01 15km（全国范围）+ d02 5km（登陆走廊，嵌套） | 仅 d01 15km（全国范围） |
| 格点数 | d01: 300×250，d02: 301×256 | 300×250（与 Run A d01 相同区域） |
| 垂直层数 | 45 层，模式顶 50hPa | 同左 |
| 积云参数化 | d01 用 Kain-Fritsch，d02 关闭（显式对流） | Kain-Fritsch |
| 输出间隔 | 60 分钟 | 180 分钟 |
| 目的 | 精细刻画登陆时刻的风眼、眼墙、沿海地形降雨 | 换取时效，看减弱后的系统能走多远、影响哪里 |

物理方案（两组一致）：WSM6 微物理、RRTMG 长波/短波辐射、YSU 边界层、Noah 陆面、
Monin-Obukhov 近地层方案，d02 关闭 SST 更新（该 WRF 编译版本 `sst_update` 功能有 bug）。

驱动场为 **NCEP GFS 0.25°**（NOMADS 实时下载），计算平台为本地 **14 核工作站**（`nproc_x=7 × nproc_y=2` 显式域分解）。

## 目录结构

```
namelists/
  RunA_landfall/       Run A 的 namelist.wps + namelist.input
  RunB_extended/        Run B 的 namelist.wps + namelist.input
scripts/
  check_domains.py                后处理前核验两组模拟网格范围/嵌套关系
  postprocess_runA_manual.py       Run A 路径追踪 + 登陆走廊(d02)范围的风雨图
  postprocess_runA_d01_wide.py     Run A 全国范围(d01)风雨图，核查风雨是否已推进到内陆
  postprocess_runB_wide.py         Run B 全国范围风雨图 + 参考城市累计降雨量时间序列
  make_landfall_video.py           Run A 登陆过程动画（风速+气压+路径 | 累计降雨 两联图）
  make_runB_videos.py              Run B 三段全国范围动画（见下）
results/
  bavi_runA_track.png / .csv       Run A 模拟路径与强度（SLP极小值追踪）
  bavi_runA_wind_swath.png         Run A 登陆走廊(d02)逐时次最大10m风速
  bavi_runA_rain_total.png         Run A 登陆走廊(d02)累计降雨
  bavi_runA_d01_*_wide.png         Run A 全国范围(d01)风雨图
  bavi_runA_landfall.mp4           Run A 登陆过程动画
  domain_check.png                 两组模拟网格范围核验图
  bavi_runB_rain_total_wide.png    Run B 全国范围10天累计降雨
  bavi_runB_wind_swath_wide.png    Run B 全国范围10天大风落区
  bavi_runB_city_rain_timeseries.png  5个参考城市累计降雨量随时间演变
  bavi_runB_track.csv              Run B 路径追踪（第4天起不再具有台风路径物理意义，见下方局限说明）
  bavi_runB_wind_track.mp4         Run B 全国风场+气压+追踪路径演变（10天，~20s）
  bavi_runB_rain_rate.mp4          Run B 全国3小时降雨率演变——可直接看到雨带向内陆移动的过程
  bavi_runB_rain_cumulative.mp4    Run B 全国累计降雨演变
```

后处理脚本均不依赖 `wrf-python`（该库在 Python 3.12+/numpy 2.x 环境下因 `numpy.distutils`
被移除而无法安装），SLP/风速/降雨等诊断量直接用 `netCDF4` 读取原始变量手算。

## Run A：登陆过程结果

模拟路径从东南洋面（124–125°E 附近，最低气压一度达 955.7hPa）向西北移动，
在浙闽交界登陆，登陆后迅速减弱北上，最终在苏皖交界附近气旋性环流趋于停滞、持续减弱。

### 与实况（中央气象台）检验

| 指标 | 实况 | Run A 模拟 | 判定 |
|---|---|---|---|
| 登陆时间 | 07-11 23:20 北京时（15:20 UTC） | 减弱速率在 14–16 UTC 明显放缓 | 吻合 |
| 登陆地点 | 浙江台州玉环坎门（28.05°N, 121.27°E） | 约 27.9°N, 121.2°E（插值） | 偏差约 18km |
| 登陆强度 | 955hPa · 40m/s（13级） | ≈963hPa · 30 出头 m/s | 偏弱约 8hPa |
| 沿海累计雨量 | 300–500mm（浙闽沿海） | >400mm（沿海山地/台湾山区） | 量级吻合 |

强度偏弱是 5km 网格分辨率下的典型系统性偏差——未做资料同化或涡旋 Bogus 初始化。

## Run B：10 天延伸预报结果

Run A 结束时（84 小时）风雨尚未推进到内陆。Run B 用单层 15km、10 天时效换取更大的时间/空间覆盖，
结果显示**内陆五座参考城市 10 天内均获得有意义的累计降雨，但大风影响没有推进内陆**：

| 城市 | 10天累计降雨 |
|---|---|
| 石家庄（河北） | 109.3mm |
| 郑州（河南） | 71.6mm |
| 北京 | 71.3mm |
| 济南（山东） | 63.2mm |
| 兰州 | 53.6mm |

降雨呈两波：第 0–1 天是模拟起始时当地已有的独立天气系统；真正与巴威相关的降雨出现在
第 2–7.5 天，是台风减弱后残余环流被高空槽卷吸北上、与冷空气结合产生的降水，而非台风本体
直接扫过。大风核心（35–40 m/s）在整个 10 天里始终锁定在浙闽沿海到台湾海峡一带，上述五座
内陆城市逐时次最大 10m 风速均在 10–20 m/s，不构成大风灾害性影响。

## 已知局限

- 海平面气压用 `PSFC/HGT/T2` 标准气压订正公式近似给出，足够用于路径追踪，但非业务级精确算法。
- Run A 路径追踪前 2 个时次（起报后 0–1 小时）因初始猜测点偏差不可信。
- Run B 的 SLP 极小值追踪算法在台风失去闭合环流后（约第 4 天起）会漂移到附近任意弱低压，
  之后的"路径"不再具有台风路径的物理意义，`bavi_runB_track.mp4`/`.csv` 中对应时段仅供参考。
  同理，`bavi_runB_wind_track.mp4` 中第4天后的路径线也不应被解读为台风本体位置。
- Run B 为换取 10 天时效，牺牲了 5km 嵌套网格分辨率，且未做资料同化，其降雨落区/量级
  只能作定性参考，不代表逐小时精确预报。
- 该 WRF 编译版本 `sst_update`（`io_form_auxinput4`）功能存在 bug（wrflowinp 读取失败），
  两组模拟均已禁用海温更新（`sst_update=0`）。

## 复现

```bash
# WPS: geogrid -> ungrib(GFS 0.25°) -> metgrid
# 之后：real.exe -> mpirun --allow-run-as-root -np 14 wrf.exe
# namelist 见 namelists/RunA_landfall/ 与 namelists/RunB_extended/

# 后处理（Python 3.12，需 netCDF4/numpy/matplotlib/cartopy/pandas）
python scripts/postprocess_runA_d01_wide.py
python scripts/postprocess_runB_wide.py
python scripts/make_runB_videos.py   # 需要 ffmpeg 拼接帧到 mp4
```
