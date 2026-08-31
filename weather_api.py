# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 17:27:58 2025
@author: dream
"""

# Name: Mackenzie Jackson
# Module 3 - Individual Project Assignment - Data Gathering
# "Weather API"
#
# Citations:
# Date: 10/13/2025
#   1. Open-Meteo Historical Weather Archive API
#   Adapted From: https://archive-api.open-meteo.com/v1/archive
#
#   2. Gates Bolton Analytics - API Tutorial
#   Based On: https://gatesboltonanalytics.com/?page_id=254
#
#   3. Pandas Data Frame- Writing/Appending to a file
#   Referenced: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
#
#   4. Find if path exists using OS library
#   Referenced: https://docs.python.org/3/library/os.path.html

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import time
import os

#file that will store retrieved data
output_file = "climate_data_2.csv"

#Read coordinates dataset
coords = pd.read_csv("country_coords.csv")  # Must have Country, Latitude, Longitude columns


#  Resume point – adjust to where last run stopped (reached max daily API requests)
start_country = "Uzbekistan"
start_year = 2019
start_found = False

# Setup Open-Meteo client with cache and retry logic
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Begin to process data
results = []

# Loop through each country and year
for _, row in coords.iterrows():
    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    for year in range(1995, 2021):
        
        if not start_found:
            if country == start_country and year == start_year:
                start_found = True
                print(f"Resuming from {country}, {year}")
            else:
                continue

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "daily": [
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "sunshine_duration",
                "shortwave_radiation_sum"
            ],
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "mm"
        }

        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()

            # Extract and transform daily data into yearly sum/averages
            df = pd.DataFrame({
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "temperature_mean_f": daily.Variables(0).ValuesAsNumpy(),
                "temperature_max_f": daily.Variables(1).ValuesAsNumpy(),
                "temperature_min_f": daily.Variables(2).ValuesAsNumpy(),
                "precipitation_mm": daily.Variables(3).ValuesAsNumpy(),
                "sunshine_duration_s": daily.Variables(4).ValuesAsNumpy(),
                "shortwave_radiation_mj": daily.Variables(5).ValuesAsNumpy()
            })

            summary = {
                "country": country,
                "year": year,
                "avg_temp_mean_f": df["temperature_mean_f"].mean(),
                "avg_temp_max_f": df["temperature_max_f"].mean(),
                "avg_temp_min_f": df["temperature_min_f"].mean(),
                "total_precip_mm": df["precipitation_mm"].sum(),
                "avg_sunshine_hrs": df["sunshine_duration_s"].sum() / 3600,
                "total_radiation_mj": df["shortwave_radiation_mj"].sum()
            }

            results.append(summary)
            print(f"Retrieved data for {country}, {year}")

            # Save every successful request return 
            # checks if file path exists, if it doesn't, add header (for first session)
            pd.DataFrame(results).to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
            results = []  # clear list after each write

            time.sleep(5)  # small delay between calls to not overload API

        #I was having a difficult time overloading the API, so threw an excepton to waite longer and retry        
        except Exception as e:
            print(f"Error for {country}, {year}: {e}")
            print("Waiting 2 minutes before retry...")
            time.sleep(120)
            continue

print("\n Data collection complete.")