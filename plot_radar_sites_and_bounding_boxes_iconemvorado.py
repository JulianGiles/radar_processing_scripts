#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:13:28 2026

@author: jgiles
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
from matplotlib.lines import Line2D

# Format: (Name, Center_Lat, Center_Lon, Lon_Min, Lon_Max, Lat_Min, Lat_Max)
stations = [
    ("PRO", 52.6486, 13.8581, 9.41,  18.31, 49.95, 55.35),
    ("UMD", 52.1600, 11.1761, 6.77,  15.59, 49.46, 54.86),
    ("TUR", 48.5853, 9.7828,  5.69,  13.87, 45.89, 51.29),
    ("AFY", 38.4017, 30.4192, 26.97, 33.87, 35.70, 41.10),
    ("ANK", 39.7986, 32.9714, 29.45, 36.49, 37.10, 42.50),
    ("GZT", 37.1372, 37.1372, 33.75, 40.53, 34.44, 39.84),
    ("HTY", 36.3178, 35.7881, 32.43, 39.15, 33.62, 39.02),
    ("SVS", 39.7656, 36.8544, 33.33, 40.37, 37.07, 42.47)
]

fig = plt.figure(figsize=(12, 10))
# Using PlateCarree so the lat/lon bounding boxes appear as perfect rectangles
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
ax.set_extent([0, 45, 30, 60], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.6)
ax.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--')

geodesic = cgeo.Geodesic()

for name, lat, lon, lon_min, lon_max, lat_min, lat_max in stations:
    # 1. Plot the radar site center
    ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())
    ax.text(lon + 0.3, lat + 0.3, name, transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold')

    # 2. Plot the calculated Lat/Lon bounding box
    ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            color='red', linewidth=1.5, transform=ccrs.PlateCarree())

    # 3. Plot the true 300 km radius circle using cartopy's geodesic tool
    circle_points = geodesic.circle(lon=lon, lat=lat, radius=300_000) # radius in meters
    ax.plot(circle_points[:, 0], circle_points[:, 1], color='blue', linestyle='--', transform=ccrs.Geodetic())

# Custom legend
legend_elements = [
    Line2D([0], [0], color='red', lw=1.5, label='Calculated ICON Lat/Lon Box'),
    Line2D([0], [0], color='blue', lw=1.5, linestyle='--', label='True 300 km Radius'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=6, label='Radar Site')
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=1)

plt.title('Radar Sites: 300 km Coverage vs ICON Output Grid')
plt.show()