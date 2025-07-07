#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:48:42 2025

@author: jgiles

This scripts recursively downloads all available soundings
from the University of Wyoming database for a given list of dates.
"""
import os
import io
import re
import time
import random
import requests
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
BASE_URL = "http://weather.uwyo.edu/wsgi/sounding"  # ← changed from cgi-bin to wsgi
SRC = "FM35"  # ← needed for WSGI endpoint
REGION = "naconf"
TYPE_STR = "TEXT:CSV"
BASE_SAVE_DIR = "/automount/ags/jgiles/soundings_wyoming/"
CSV_SAVE_DIR = os.path.join(BASE_SAVE_DIR, "csv")

def parse_sounding_to_netcdf(csv_text, station, date_obj, hour):
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return f"Could not parse CSV for {station} {date_obj.strftime('%Y%m%d')}{hour}"

    if df.empty or "geopotential height_m" not in df.columns:
        return f"No valid data found for {station} {date_obj.strftime('%Y%m%d')}{hour}"

    # Column mapping to match UWYO NetCDF variable style
    column_map = {
        "pressure_hPa": "PRES",
        "geopotential height_m": "HGHT",
        "temperature_C": "TEMP",
        "dew point temperature_C": "DWPT",
        "relative humidity_%": "RELH",
        "mixing ratio_g/kg": "MIXR",
        "wind direction_degree": "WDIR",
        "wind speed_m/s": "WSPD",
        "ice point temperature_C": "IPT",  # Not always in original table, optional
        "humidity wrt ice_%": "HICE"       # Optional
    }

    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    df = df.dropna(subset=["HGHT"])  # Ensure vertical axis is valid

    ds = xr.Dataset()
    ds.coords["HGHT"] = ("HGHT", df["HGHT"].values)

    for col in df.columns:
        if col != "HGHT":
            ds[col] = ("HGHT", df[col].values)

    # Metadata
    ds.attrs["Station number"] = int(station)
    ds.attrs["Observation time"] = date_obj.strftime("%Y-%m-%d %H:00:00")
    ds.attrs["Variables"] = list(df.columns)

    # Save NetCDF
    year_dir = os.path.join(BASE_SAVE_DIR+"/"+station, date_obj.strftime("%Y"))
    os.makedirs(year_dir, exist_ok=True)
    filename = f"{station}_{date_obj.strftime('%Y%m%d')}{hour}.nc"
    filepath = os.path.join(year_dir, filename)
    ds.to_netcdf(filepath)

    return f"Saved NetCDF: {filepath}"


def download_and_convert(station, date_obj, hour):
    """Downloads a single sounding from WSGI API and converts it to NetCDF."""
    datetime_str = f"{date_obj.strftime('%Y-%m-%d')} {hour}:00:00"
    params = {
        "datetime": datetime_str,
        "id": station,
        "src": SRC,
        "type": TYPE_STR
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)

        # Ensure CSV headers are present
        if not response.ok or "pressure_hPa" not in response.text:
            return f"No data or bad format for {station} at {datetime_str}"

        # Now parse directly from the CSV string to NetCDF
        result = parse_sounding_to_netcdf(response.text, station, date_obj, hour)

        return result

    except Exception as e:
        return f"Error for {station} on {date_obj.strftime('%Y%m%d')}{hour}: {e}"

    finally:
        time.sleep(random.uniform(0.5, 1.5))


def download_multiple_dates(date_list, station_list,
                            hours=["00", "03", "06", "09", "12", "15", "18", "21"],
                            max_workers=10):
    """Downloads and converts soundings for multiple dates and stations."""
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for date_str in date_list:
            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                print(f"Invalid date: {date_str}")
                continue

            for hour in hours:
                for station in station_list:
                    task = executor.submit(download_and_convert, station, date_obj, hour)
                    tasks.append(task)

        for future in as_completed(tasks):
            print(future.result())

# ----------------------------
# ✅ Example usage:
station_list = [
    # "17351", # Adana/Bolge, Turkey
    # "17240", # ISPARTA, TURKEY
    "17196", # KAYSERI, TURKEY
]

date_list_afy = ['20160514',
 '20160515',
 '20160629',
 '20160805',
 '20160913',
 '20170602',
 '20170618',
 '20170721',
 '20170925',
 '20170929',
 '20180614',
 '20180615',
 '20180617',
 '20180619',
 '20180625',
 '20180731',
 '20180804',
 '20180805',
 '20180821',
 '20180827',
 '20200809',
 '20200924',
 '20201019']

date_list_svs = ['20170413',
 '20170618',
 '20170619',
 '20170816',
 '20171001',
 '20171028',
 '20171029',
 '20171105',
 '20171223',
 '20180308',
 '20180324',
 '20180328',
 '20180329',
 '20180501',
 '20180613',
 '20180615',
 '20180616',
 '20180618',
 '20180619',
 '20180622',
 '20180625',
 '20180910',
 '20181001',
 '20181008',
 '20181009',
 '20181021',
 '20181024',
 '20181025',
 '20190407',
 '20190502',
 '20190505',
 '20190506',
 '20190510',
 '20190511',
 '20190512',
 '20190516',
 '20190522',
 '20190524',
 '20190531',
 '20190609',
 '20190610',
 '20190612',
 '20190620',
 '20190622',
 '20190711',
 '20190714',
 '20190715',
 '20190717',
 '20190718',
 '20190914',
 '20190915',
 '20190925',
 '20191101',
 '20191129',
 '20200312',
 '20200313',
 '20200314',
 '20200315',
 '20200405',
 '20200415',
 '20200421',
 '20200422',
 '20200430',
 '20200501',
 '20200503',
 '20200504',
 '20200505',
 '20200507',
 '20200508',
 '20200522',
 '20200608',
 '20200610',
 '20200612',
 '20200613',
 '20200618',
 '20201002',
 '20201020',
 '20201103',
 '20201104',
 '20201105']

download_multiple_dates(date_list_svs, station_list)

