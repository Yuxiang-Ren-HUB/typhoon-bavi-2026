"""
Run A (登陆段) 后处理: 台风路径+强度、大风落区、累计降雨
不依赖wrf-python(与当前环境numpy 2.x/Python 3.12不兼容)，直接用netCDF4读原始变量手算：
  - 海平面气压: 用PSFC/HGT/T2标准气压订正公式近似(供追踪风暴中心用，非wrf-python的精确算法)
  - 10m风速: sqrt(U10^2+V10^2) -- 这个量本身与网格/地理坐标系旋转无关，不需要额外订正
  - 累计降雨: RAINC+RAINNC (原始变量，直接相加)

pip install netCDF4 numpy pandas matplotlib cartopy
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- config ----------------
WRFOUT_DIR = r"F:\WRF_backup\RunA_output"
DOMAIN = "d02"
FIRST_GUESS = (27.5, 121.2)   # 起报时刻(2026-07-10 18Z)附近的台风位置估计, 已知登陆点附近
SEARCH_RADIUS_DEG = 1.5
MAP_EXTENT = [112, 132, 18, 34]
OUT_PREFIX = "bavi_runA"


def approx_slp(psfc, hgt, t2):
    """standard barometric reduction, hPa. psfc: Pa, hgt: m, t2: K"""
    g = 9.80665
    Rd = 287.05
    tv = t2 + 0.0065 * hgt / 2.0
    return psfc / 100.0 * np.exp(g * hgt / (Rd * tv))


files = sorted(glob.glob(os.path.join(WRFOUT_DIR, f"wrfout_{DOMAIN}_*")))
files = [f for f in files if os.path.isfile(f)]
if not files:
    raise SystemExit(f"no wrfout_{DOMAIN}_* files found in {WRFOUT_DIR}")
print(f"found {len(files)} wrfout files for {DOMAIN}")

# static fields from first file
ds0 = nc.Dataset(files[0])
lats = ds0.variables['XLAT'][0, :, :].astype(float)
lons = ds0.variables['XLONG'][0, :, :].astype(float)
ny, nx = lats.shape
ds0.close()

# ---------------- pass 1: build full time series (track needs sequential access) ----------------
all_times = []
all_slp = []
all_wspd10 = []
rain_cumulative_last = None

for fpath in files:
    ds = nc.Dataset(fpath)
    times_raw = ds.variables['Times'][:]
    ntime = times_raw.shape[0]
    psfc = ds.variables['PSFC'][:]        # (t, ny, nx), Pa
    t2 = ds.variables['T2'][:]            # (t, ny, nx), K
    hgt = ds.variables['HGT'][0, :, :]    # (ny, nx), static, m
    u10 = ds.variables['U10'][:]
    v10 = ds.variables['V10'][:]
    rainc = ds.variables['RAINC'][:]
    rainnc = ds.variables['RAINNC'][:]

    for t in range(ntime):
        time_str = times_raw[t].tobytes().decode().strip()
        all_times.append(time_str)
        slp = approx_slp(psfc[t], hgt, t2[t])
        all_slp.append(slp)
        wspd = np.sqrt(u10[t] ** 2 + v10[t] ** 2)
        all_wspd10.append(wspd)

    rain_cumulative_last = rainc[-1] + rainnc[-1]
    ds.close()
    print(f"processed {os.path.basename(fpath)} ({ntime} times)")

all_slp = np.array(all_slp)          # (T, ny, nx)
all_wspd10 = np.array(all_wspd10)    # (T, ny, nx)
ntimes = all_slp.shape[0]

# ---------------- track: min SLP within search radius of previous center ----------------
track = []
center = FIRST_GUESS
for t in range(ntimes):
    slp_t = all_slp[t]
    dist = np.sqrt((lats - center[0]) ** 2 + (lons - center[1]) ** 2)
    mask = dist <= SEARCH_RADIUS_DEG
    slp_masked = np.where(mask, slp_t, np.nan)
    if np.all(np.isnan(slp_masked)):
        print(f"t={t} ({all_times[t]}): no points within {SEARCH_RADIUS_DEG}deg of {center}, stopping track")
        break
    idx = np.nanargmin(slp_masked)
    j, i = np.unravel_index(idx, slp_masked.shape)
    center = (float(lats[j, i]), float(lons[j, i]))
    min_slp = float(slp_t[j, i])
    max_wind = float(np.nanmax(np.where(mask, all_wspd10[t], np.nan)))
    track.append({"time": all_times[t], "lat": center[0], "lon": center[1],
                  "min_slp_hpa": min_slp, "max_wind_near_center_ms": max_wind})

df = pd.DataFrame(track)
df.to_csv(f"{OUT_PREFIX}_track.csv", index=False)
print(df.to_string())

# ================================================================
# 1) track + intensity map
# ================================================================
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(MAP_EXTENT)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
sc = ax.scatter(df["lon"], df["lat"], c=df["min_slp_hpa"], cmap="turbo_r", s=30,
                 transform=ccrs.PlateCarree(), zorder=5)
ax.plot(df["lon"], df["lat"], "-", color="black", linewidth=1, transform=ccrs.PlateCarree())
plt.colorbar(sc, ax=ax, label="近似海平面气压 (hPa)")
ax.set_title("台风巴威（2026）Run A 模拟路径与强度")
plt.savefig(f"{OUT_PREFIX}_track.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ================================================================
# 2) domain-wide max 10m wind swath (大风落区)
# ================================================================
max_wind_swath = np.nanmax(all_wspd10, axis=0)

fig2 = plt.figure(figsize=(8, 8))
ax2 = plt.axes(projection=ccrs.PlateCarree())
ax2.set_extent(MAP_EXTENT)
ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linestyle=":")
cf2 = ax2.contourf(lons, lats, max_wind_swath, levels=np.arange(0, 65, 5),
                    cmap="viridis", transform=ccrs.PlateCarree(), extend="max")
plt.colorbar(cf2, ax=ax2, label="逐时次最大10m风速 (m/s)")
ax2.set_title("台风巴威（2026）Run A 大风落区（全程逐时次最大10m风速）")
plt.savefig(f"{OUT_PREFIX}_wind_swath.png", dpi=200, bbox_inches="tight")
plt.close(fig2)

# ================================================================
# 3) accumulated rainfall (降雨落区)
# ================================================================
fig3 = plt.figure(figsize=(8, 8))
ax3 = plt.axes(projection=ccrs.PlateCarree())
ax3.set_extent(MAP_EXTENT)
ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linestyle=":")
cf3 = ax3.contourf(lons, lats, rain_cumulative_last, levels=[0, 10, 25, 50, 100, 150, 250, 400, 600],
                    cmap="viridis", transform=ccrs.PlateCarree(), extend="max")
plt.colorbar(cf3, ax=ax3, label="累计降雨量 (mm)")
ax3.set_title("台风巴威（2026）Run A 累计降雨预报")
plt.savefig(f"{OUT_PREFIX}_rain_total.png", dpi=200, bbox_inches="tight")
plt.close(fig3)

print("done:", f"{OUT_PREFIX}_track.csv/png, _wind_swath.png, _rain_total.png")
