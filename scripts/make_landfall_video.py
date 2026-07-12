"""
Run A (d02, 5km) 台风登陆动画：10m风速填色 + 海平面气压等压线 + 移动路径
逐帧存PNG，再用ffmpeg拼接成mp4
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

WRFOUT_DIR = r"F:\WRF_backup\RunA_output_d01"
DOMAIN = "d01"
FRAME_DIR = r"C:\Users\Admin\AppData\Local\Temp\claude\f-------Python\1d35fbaf-7c90-4d0c-8cbe-1f0bcb5a6404\scratchpad\WRF_Bavi2026\frames_landfall"
MAP_EXTENT = [105, 130, 18, 40]
FIRST_GUESS = (27.5, 121.2)
SEARCH_RADIUS_DEG = 1.5

os.makedirs(FRAME_DIR, exist_ok=True)


def approx_slp(psfc, hgt, t2):
    g = 9.80665
    Rd = 287.05
    tv = t2 + 0.0065 * hgt / 2.0
    return psfc / 100.0 * np.exp(g * hgt / (Rd * tv))


files = sorted(glob.glob(os.path.join(WRFOUT_DIR, f"wrfout_{DOMAIN}_*")))
files = [f for f in files if os.path.isfile(f)]
print(f"found {len(files)} files")

ds0 = nc.Dataset(files[0])
lats = ds0.variables['XLAT'][0, :, :].astype(float)
lons = ds0.variables['XLONG'][0, :, :].astype(float)
ds0.close()

# ---------------- pass 1: gather all times + track (for drawing the trailing path) ----------------
all_times, all_slp, all_wspd, track_lat, track_lon = [], [], [], [], []
center = FIRST_GUESS
for fpath in files:
    ds = nc.Dataset(fpath)
    times_raw = ds.variables['Times'][:]
    psfc = ds.variables['PSFC'][:]
    t2 = ds.variables['T2'][:]
    hgt = ds.variables['HGT'][0, :, :]
    u10 = ds.variables['U10'][:]
    v10 = ds.variables['V10'][:]
    for t in range(times_raw.shape[0]):
        all_times.append(times_raw[t].tobytes().decode().strip())
        slp = approx_slp(psfc[t], hgt, t2[t])
        wspd = np.sqrt(u10[t] ** 2 + v10[t] ** 2)
        all_slp.append(slp)
        all_wspd.append(wspd)
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

# ---------------- pass 2: render frames ----------------
for t in range(ntimes):
    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    cf = ax.contourf(lons, lats, all_wspd[t], levels=np.arange(0, 65, 5),
                      cmap="viridis", transform=ccrs.PlateCarree(), extend="max")
    cs = ax.contour(lons, lats, all_slp[t], levels=np.arange(940, 1015, 4),
                     colors='white', linewidths=0.6, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

    ax.plot(track_lon[:t + 1], track_lat[:t + 1], '-', color='red', linewidth=1.5,
             transform=ccrs.PlateCarree())
    ax.plot(track_lon[t], track_lat[t], 'o', color='red', markersize=9,
             markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())

    if t == 0:
        cb = plt.colorbar(cf, ax=ax, label="10m风速 (m/s)", shrink=0.75)

    ax.set_title(f"台风巴威（2026）Run A 模拟  |  {all_times[t]} UTC")
    plt.savefig(os.path.join(FRAME_DIR, f"frame_{t:03d}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    if t % 10 == 0:
        print(f"rendered frame {t}/{ntimes}")

print("all frames rendered")
