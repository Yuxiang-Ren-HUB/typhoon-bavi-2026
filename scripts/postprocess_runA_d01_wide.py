"""
Run A - d01 全国范围后处理 (覆盖河南/河北/山东/兰州/北京等内陆区域)
同 postprocess_runA_manual.py 的算法，换成d01(15km,全中国范围)数据源
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
WRFOUT_DIR = r"F:\WRF_backup\RunA_output_d01"
DOMAIN = "d01"
FIRST_GUESS = (27.5, 121.2)
SEARCH_RADIUS_DEG = 2.0   # d01是15km网格，稍微放宽一点
MAP_EXTENT = [95, 145, 10, 45]   # 全中国范围，覆盖北京、兰州
OUT_PREFIX = "bavi_runA_d01"


def approx_slp(psfc, hgt, t2):
    g = 9.80665
    Rd = 287.05
    tv = t2 + 0.0065 * hgt / 2.0
    return psfc / 100.0 * np.exp(g * hgt / (Rd * tv))


# rename any leftover colon-corrupted filenames first (same MSYS PUA issue as before)
for f in os.listdir(WRFOUT_DIR):
    clean = f.replace('', '')
    if clean != f:
        os.rename(os.path.join(WRFOUT_DIR, f), os.path.join(WRFOUT_DIR, clean))
        print(f"renamed -> {clean}")

files = sorted(glob.glob(os.path.join(WRFOUT_DIR, f"wrfout_{DOMAIN}_*")))
files = [f for f in files if os.path.isfile(f)]
if not files:
    raise SystemExit(f"no wrfout_{DOMAIN}_* files found in {WRFOUT_DIR}")
print(f"found {len(files)} wrfout files for {DOMAIN}")

ds0 = nc.Dataset(files[0])
lats = ds0.variables['XLAT'][0, :, :].astype(float)
lons = ds0.variables['XLONG'][0, :, :].astype(float)
ds0.close()

all_times = []
all_slp = []
all_wspd10 = []
rain_cumulative_last = None

for fpath in files:
    ds = nc.Dataset(fpath)
    times_raw = ds.variables['Times'][:]
    ntime = times_raw.shape[0]
    psfc = ds.variables['PSFC'][:]
    t2 = ds.variables['T2'][:]
    hgt = ds.variables['HGT'][0, :, :]
    u10 = ds.variables['U10'][:]
    v10 = ds.variables['V10'][:]
    rainc = ds.variables['RAINC'][:]
    rainnc = ds.variables['RAINNC'][:]

    for t in range(ntime):
        time_str = times_raw[t].tobytes().decode().strip()
        all_times.append(time_str)
        all_slp.append(approx_slp(psfc[t], hgt, t2[t]))
        all_wspd10.append(np.sqrt(u10[t] ** 2 + v10[t] ** 2))

    rain_cumulative_last = rainc[-1] + rainnc[-1]
    ds.close()
    print(f"processed {os.path.basename(fpath)} ({ntime} times)")

all_slp = np.array(all_slp)
all_wspd10 = np.array(all_wspd10)
ntimes = all_slp.shape[0]

# ---------------- track ----------------
track = []
center = FIRST_GUESS
for t in range(ntimes):
    slp_t = all_slp[t]
    dist = np.sqrt((lats - center[0]) ** 2 + (lons - center[1]) ** 2)
    mask = dist <= SEARCH_RADIUS_DEG
    slp_masked = np.where(mask, slp_t, np.nan)
    if np.all(np.isnan(slp_masked)):
        print(f"t={t} ({all_times[t]}): tracking stopped")
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

# ================================================================
# wide-area rainfall map (the actual ask: show inland extent)
# ================================================================
fig3 = plt.figure(figsize=(11, 9))
ax3 = plt.axes(projection=ccrs.PlateCarree())
ax3.set_extent(MAP_EXTENT)
ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linestyle=":")
ax3.add_feature(cfeature.STATES if hasattr(cfeature, 'STATES') else cfeature.BORDERS, linestyle=":", linewidth=0.4)
cf3 = ax3.contourf(lons, lats, rain_cumulative_last, levels=[0, 10, 25, 50, 100, 150, 250, 400, 600],
                    cmap="viridis", transform=ccrs.PlateCarree(), extend="max")
ax3.plot(df["lon"], df["lat"], "-", color="red", linewidth=1.2, transform=ccrs.PlateCarree(), label="模拟路径")
# reference cities
for name, la, lo in [("北京", 39.9, 116.4), ("兰州", 36.1, 103.7), ("郑州", 34.75, 113.65), ("石家庄", 38.05, 114.5), ("济南", 36.65, 117.0)]:
    ax3.plot(lo, la, 'ko', markersize=4, transform=ccrs.PlateCarree())
    ax3.annotate(name, (lo, la), fontsize=10, xytext=(4, 4), textcoords="offset points")
plt.colorbar(cf3, ax=ax3, label="累计降雨量 (mm)", shrink=0.8)
ax3.legend(loc="lower left")
ax3.set_title("台风巴威（2026）Run A 全国范围累计降雨预报 (d01, 15km)")
plt.savefig(f"{OUT_PREFIX}_rain_total_wide.png", dpi=200, bbox_inches="tight")
plt.close(fig3)

# ================================================================
# wide-area wind swath
# ================================================================
max_wind_swath = np.nanmax(all_wspd10, axis=0)
fig2 = plt.figure(figsize=(11, 9))
ax2 = plt.axes(projection=ccrs.PlateCarree())
ax2.set_extent(MAP_EXTENT)
ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linestyle=":")
cf2 = ax2.contourf(lons, lats, max_wind_swath, levels=np.arange(0, 65, 5),
                    cmap="viridis", transform=ccrs.PlateCarree(), extend="max")
plt.colorbar(cf2, ax=ax2, label="逐时次最大10m风速 (m/s)", shrink=0.8)
ax2.set_title("台风巴威（2026）Run A 全国范围大风落区 (d01, 15km)")
plt.savefig(f"{OUT_PREFIX}_wind_swath_wide.png", dpi=200, bbox_inches="tight")
plt.close(fig2)

print("done:", f"{OUT_PREFIX}_rain_total_wide.png, _wind_swath_wide.png, _track.csv")
