"""
Sanity-check Run A / Run B domain placement BEFORE running geogrid.exe.
Pure matplotlib/cartopy, no WPS binaries needed.

pip install matplotlib cartopy
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def d01_extent_km(ref_lat, ref_lon, dx_m, e_we, e_sn):
    """approximate lon/lat box of the parent (d01) mercator WPS domain, centered on ref_lat/ref_lon"""
    import numpy as np
    width_km = dx_m * (e_we - 1) / 1000
    height_km = dx_m * (e_sn - 1) / 1000
    dlat = height_km / 111.0
    dlon = width_km / (111.0 * np.cos(np.radians(ref_lat)))
    return (ref_lon - dlon / 2, ref_lon + dlon / 2, ref_lat - dlat / 2, ref_lat + dlat / 2)


def nest_extent_from_parent(d01_box, e_we_d01, e_sn_d01, i_parent_start, j_parent_start,
                             parent_grid_ratio, e_we_nest, e_sn_nest):
    """proper offset-based footprint of a nest within its parent, using i/j_parent_start
    (WPS convention: nest start indices are 1-based parent grid-cell coordinates)"""
    l, r, b, t = d01_box
    dlon_per_cell = (r - l) / (e_we_d01 - 1)
    dlat_per_cell = (t - b) / (e_sn_d01 - 1)
    cells_x = (e_we_nest - 1) // parent_grid_ratio
    cells_y = (e_sn_nest - 1) // parent_grid_ratio
    nl = l + (i_parent_start - 1) * dlon_per_cell
    nr = l + (i_parent_start - 1 + cells_x) * dlon_per_cell
    nb = b + (j_parent_start - 1) * dlat_per_cell
    nt = b + (j_parent_start - 1 + cells_y) * dlat_per_cell
    return (nl, nr, nb, nt)


# --- values must match namelist.wps in RunA_landfall / RunB_extended ---
d01 = d01_extent_km(ref_lat=27.0, ref_lon=121.0, dx_m=15000, e_we=300, e_sn=250)
d02 = nest_extent_from_parent(d01, e_we_d01=300, e_sn_d01=250,
                               i_parent_start=110, j_parent_start=80,
                               parent_grid_ratio=3, e_we_nest=301, e_sn_nest=256)

# --- best-track waypoints to overlay: fill in from latest CMA/JTWC advisory ---
# format: (time_label, lat, lon)
track_waypoints = [
    ("07-10 17Z", 23.5, 128.5),   # ~830km SE of Zhejiang-Fujian border, per CMA bulletin
    ("07-11 20Z landfall(fcst)", 27.5, 121.2),  # Wenling-Xiapu coast
    ("07-13 (dissipate, fcst)", 31.0, 118.0),   # west of Shanghai
    ("北京", 39.9, 116.4),
    ("兰州", 36.1, 103.7),
]

fig = plt.figure(figsize=(9, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([90, 145, 5, 50])
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")

for name, (l, r, b, t), color in [("d01 (15km)", d01, "blue"), ("d02 (5km)", d02, "red")]:
    ax.plot([l, r, r, l, l], [b, b, t, t, b], color=color, linewidth=2, label=name, transform=ccrs.PlateCarree())

lons = [w[2] for w in track_waypoints]
lats = [w[1] for w in track_waypoints]
ax.plot(lons, lats, "o-", color="black", transform=ccrs.PlateCarree())
for label, lat, lon in track_waypoints:
    ax.annotate(label, (lon, lat), fontsize=8, xytext=(3, 3), textcoords="offset points")

ax.legend(loc="lower left")
ax.set_title("Run A/B 预设网格范围 vs 台风巴威(2026)预报路径— 跑geogrid前先核对")
plt.savefig("domain_check.png", dpi=200, bbox_inches="tight")
print("saved domain_check.png -- 确认d02红框是否完整包住黑色路径点后再继续")
