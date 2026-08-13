#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:27:04 2026

@author: jgiles
"""

"""
Manually run this in the console for the script to work
%matplotlib qt
"""
import os
import sys

# List all possible paths where this script might live
possible_paths = [
    '/home/jgiles/Scripts/python/radar_processing_scripts',              # Office PC
    '/p/scratch/detectrea2/giles1/radar_processing_scripts',             # JUWELS Scratch
]

# Find the one that exists on the current machine and add it
for script_dir in possible_paths:
    if os.path.exists(script_dir):
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        break  # Stop checking once we find the right one

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.widgets import Slider, Button
import wradlib as wrl

import utils
import radarmet


# 1. Load the dataset
file_path = '/automount/realpep//upload/jgiles/dmi/final_ppis/2017/2017-04/2017-04-13/HTY/MON_YAZ_B/1.5/MON_YAZ_B-allmoms-1.5-2017-04-13-HTY-h5netcdf.nc'
data = xr.open_dataset(file_path)

# 2. Parameterize the variables
dbzh_name = 'DBZH'
zdr_name = 'ZDR_EC_OC'
phi_name = 'PHIDP_OC'
rho_name = 'RHOHV'

variables = [dbzh_name, zdr_name, phi_name, rho_name]
max_idx = len(data.time) - 1

# 3. Set up the figure and a 2x2 grid of axes
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3)
axes = axes.flatten()

time_idx = 0
meshes = []

# 4. Loop through the axes and variables to plot the initial timestep
for i, mom in enumerate(variables):
    ax = axes[i]

    # Get ticks, cmap, and norm using your custom dictionary logic
    ticks = radarmet.visdict14[mom.split("_")[0]]["ticks"]
    cmap = mpl.colormaps.get_cmap("miub2")

    # Alternative cmap
    # cmap0 = mpl.colormaps.get_cmap("SpectralExtended")
    # cmap = mpl.colors.ListedColormap(cmap0(np.linspace(0, 1, len(ticks))), N=len(ticks)+1)

    norm = utils.get_discrete_norm(ticks, cmap.N, extend="both")

    # Plot using wradlib and capture the mesh
    mesh = data.isel(time=time_idx)[mom].wrl.vis.plot(
        ax=ax,
        cmap=cmap,
        norm=norm,
        extend="both"
    )

    meshes.append(mesh)
    ax.set_title(mom)

# Add a single timestamp title for the whole figure
time_title = fig.suptitle(f"Time: {data.time[time_idx].values}", fontsize=14, fontweight='bold')

# 5. Set up the UI elements (Slider and Buttons)
# We adjust the widths to fit the buttons on the left and right of the slider
ax_prev = plt.axes([0.1, 0.05, 0.08, 0.03])
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
ax_next = plt.axes([0.8, 0.05, 0.08, 0.03])

btn_prev = Button(ax_prev, 'Previous')
btn_next = Button(ax_next, 'Next')

time_slider = Slider(
    ax=ax_slider,
    label='Timestep Index',
    valmin=0,
    valmax=max_idx,
    valinit=time_idx,
    valstep=1
)

# 6. Define the update functions
def update(val):
    idx = int(time_slider.val)

    for i, mom in enumerate(variables):
        new_data = data.isel(time=idx)[mom].values
        meshes[i].set_array(new_data.ravel())

    time_title.set_text(f"Time: {data.time[idx].values}")
    fig.canvas.draw_idle()

def step_forward(event):
    current_val = int(time_slider.val)
    if current_val < max_idx:
        # Calling set_val automatically triggers the slider's 'update' function
        time_slider.set_val(current_val + 1)

def step_backward(event):
    current_val = int(time_slider.val)
    if current_val > 0:
        time_slider.set_val(current_val - 1)

# 7. Connect the UI elements to their functions
time_slider.on_changed(update)
btn_prev.on_clicked(step_backward)
btn_next.on_clicked(step_forward)

plt.show()