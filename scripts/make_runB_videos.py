"""
Run B (d01, 15km, 10天/240h) 全国范围动画: 三段视频
  1) 风速+海平面气压+追踪路径 (bavi_runB_wind_track.mp4)
  2) 逐时次间隔降雨率 (3小时累计，展示雨带移动) (bavi_runB_rain_rate.mp4)
  3) 累计降雨量演变 (bavi_runB_rain_cumulative.mp4)
81帧 @ 4fps ≈ 20秒
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

WRFOUT_DIR = r"F:\WRF_backup\RunB_output"
DOMAIN = "d01"
BASE_DIR = r"C:\Users\Admin\AppData\Local\Temp\claude\f-------Python\1d35fbaf-7c90-4d0c-8cbe-1f0bcb5a6404\scratchpad\WRF_Bavi2026"
MAP_EXTENT = [95, 145, 10, 45]
FIRST_GUESS = (27.5, 121.2)
SEARCH_RADIUS_DEG = 2.0
CITIES = [("北京", 39.9, 116.4), ("兰州", 36.1, 103.7), ("郑州", 34.75, 113.65),
          ("石家庄", 38.05, 114.5), ("济南", 36.65, 117.0)]

FRAME_DIR_WIND = os.path.join(BASE_DIR, "frames_runB_wind")
FRAME_DIR_RATE = os.path.join(BASE_DIR, "frames_runB_rate")
FRAME_DIR_CUM = os.path.join(BASE_DIR, "frames_runB_cum")
for d in (FRAME_DIR_WIND, FRAME_DIR_RATE, FRAME_DIR_CUM):
    os.makedirs(d, exist_ok=True)


def approx_slp(psfc, hgt, t2):
    g = 9.80665
    Rd = 287.05
    tv = t2 + 0.0065 * hgt / 2.0
    return psfc / 100.0 * np.exp(g * hgt / (Rd * tv))


for f in os.listdir(WRFOUT_DIR):
    clean = f.replace('', '')
    if clean != f:
        os.rename(os.path.join(WRFOUT_DIR, f), os.path.join(WRFOUT_DIR, clean))

files = sorted(glob.glob(os.path.join(WRFOUT_DIR, f"wrfout_{DOMAIN}_*")))
files = [f for f in files if os.path.isfile(f)]
print(f"found {len(files)} files")

ds0 = nc.Dataset(files[0])
lats = ds0.variables['XLAT'][0, :, :].astype(float)
lons = ds0.variables['XLONG'][0, :, :].astype(float)
ds0.close()

# ---------------- pass 1: load everything + build track ----------------
all_times, all_slp, all_wspd, all_rain = [], [], [], []
track_lat, track_lon, track_valid = [], [], []
center = FIRST_GUESS
track_alive = True
for fpath in files:
    ds = nc.Dataset(fpath)
    times_raw = ds.variables['Times'][:]
    psfc = ds.variables['PSFC'][:]
    t2 = ds.variables['T2'][:]
    hgt = ds.variables['HGT'][0, :, :]
    u10 = ds.variables['U10'][:]
    v10 = ds.variables['V10'][:]
    rainc = ds.variables['RAINC'][:]
    rainnc = ds.variables['RAINNC'][:]
    for t in range(times_raw.shape[0]):
        all_times.append(times_raw[t].tobytes().decode().strip())
        slp = approx_slp(psfc[t], hgt, t2[t])
        all_slp.append(slp)
        all_wspd.append(np.sqrt(u10[t] ** 2 + v10[t] ** 2))
        all_rain.append(rainc[t] + rainnc[t])
        if track_alive:
            dist = np.sqrt((lats - center[0]) ** 2 + (lons - center[1]) ** 2)
            mask = dist <= SEARCH_RADIUS_DEG
            slp_masked = np.where(mask, slp, np.nan)
            if not np.all(np.isnan(slp_masked)):
                idx = np.nanargmin(slp_masked)
                j, i = np.unravel_index(idx, slp_masked.shape)
                center = (float(lats[j, i]), float(lons[j, i]))
        track_lat.append(center[0])
        track_lon.append(center[1])
    ds.close()
    print(f"loaded {os.path.basename(fpath)}")

ntimes = len(all_times)
print(f"total frames: {ntimes}")
# track stops being physically meaningful ~day 4 (system fully dissipated) per earlier static analysis;
# still drawn for continuity, matches static report caveat
TRACK_CUTOFF = min(int(4.0 * 24 / 3), ntimes)  # ~day4 at 3h spacing

RAIN_LEVELS_CUM = [0, 10, 25, 50, 100, 150, 250, 400, 600, 800, 1000]
RAIN_LEVELS_RATE = [0, 1, 2, 5, 10, 20, 40, 60, 90]

# ================================================================
# video 1: wind + SLP + track
# ================================================================
for t in range(ntimes):
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    cf = ax.contourf(lons, lats, all_wspd[t], levels=np.arange(0, 65, 5),
                      cmap="viridis", transform=ccrs.PlateCarree(), extend="max")
    cs = ax.contour(lons, lats, all_slp[t], levels=np.arange(940, 1025, 4),
                     colors='white', linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=6, fmt='%d')
    tcut = min(t + 1, TRACK_CUTOFF)
    ax.plot(track_lon[:tcut], track_lat[:tcut], '-', color='red', linewidth=1.3,
            transform=ccrs.PlateCarree())
    if t < TRACK_CUTOFF:
        ax.plot(track_lon[t], track_lat[t], 'o', color='red', markersize=8,
                markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
    for name, la, lo in CITIES:
        ax.plot(lo, la, 'ks', markersize=4, transform=ccrs.PlateCarree())
        ax.annotate(name, (lo, la), fontsize=9, xytext=(4, 4), textcoords="offset points")
    plt.colorbar(cf, ax=ax, label="10m风速 (m/s)", shrink=0.8)
    ax.set_title(f"台风巴威（2026）Run B 风场+气压演变  |  {all_times[t]} UTC", fontsize=12)
    plt.savefig(os.path.join(FRAME_DIR_WIND, f"frame_{t:03d}.png"), dpi=125, bbox_inches="tight")
    plt.close(fig)
    if t % 10 == 0:
        print(f"[wind] rendered {t}/{ntimes}")

# ================================================================
# video 2: rain rate (delta between consecutive outputs, 3h accumulation)
# ================================================================
for t in range(ntimes):
    rate = all_rain[t] - all_rain[t - 1] if t > 0 else np.zeros_like(all_rain[0])
    rate = np.clip(rate, 0, None)
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    cf = ax.contourf(lons, lats, rate, levels=RAIN_LEVELS_RATE,
                      cmap="viridis", transform=ccrs.PlateCarree(), extend="max")
    for name, la, lo in CITIES:
        ax.plot(lo, la, 'ks', markersize=4, transform=ccrs.PlateCarree())
        ax.annotate(name, (lo, la), fontsize=9, xytext=(4, 4), textcoords="offset points")
    plt.colorbar(cf, ax=ax, label="3小时降雨量 (mm)", shrink=0.8)
    ax.set_title(f"台风巴威（2026）Run B 3小时降雨率  |  {all_times[t]} UTC", fontsize=12)
    plt.savefig(os.path.join(FRAME_DIR_RATE, f"frame_{t:03d}.png"), dpi=125, bbox_inches="tight")
    plt.close(fig)
    if t % 10 == 0:
        print(f"[rate] rendered {t}/{ntimes}")

# ================================================================
# video 3: cumulative rainfall growing over time
# ================================================================
for t in range(ntimes):
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    cf = ax.contourf(lons, lats, all_rain[t], levels=RAIN_LEVELS_CUM,
                      cmap="viridis", transform=ccrs.PlateCarree(), extend="max")
    for name, la, lo in CITIES:
        ax.plot(lo, la, 'ks', markersize=4, transform=ccrs.PlateCarree())
        ax.annotate(name, (lo, la), fontsize=9, xytext=(4, 4), textcoords="offset points")
    plt.colorbar(cf, ax=ax, label="累计降雨量 (mm，起报以来)", shrink=0.8)
    ax.set_title(f"台风巴威（2026）Run B 累计降雨演变  |  {all_times[t]} UTC", fontsize=12)
    plt.savefig(os.path.join(FRAME_DIR_CUM, f"frame_{t:03d}.png"), dpi=125, bbox_inches="tight")
    plt.close(fig)
    if t % 10 == 0:
        print(f"[cum] rendered {t}/{ntimes}")

print("all frames rendered. ntimes =", ntimes)
