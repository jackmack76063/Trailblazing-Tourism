# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:01:06 2025

@author: dream
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

allTrailsFile = "Global_Alltrails.csv"
trailsdf = pd.read_csv(allTrailsFile)

print("\nFirst five rows (Before Cleaning):\n")
print(trailsdf.head())

trailsdf['TrailCount'] = trailsdf['TrailCount'].str.replace(',', '').astype(int)
trailsdf['TrailCount'] = trailsdf['TrailCount'].astype(int)

print("\nDatatypes:\n")
print(trailsdf.dtypes)

# Sort alphabetically by country
trailsdf = trailsdf.sort_values('Country')

# Check for missing values
print("\nMissing Values:\n")
print(trailsdf.isnull().sum())

trailsdf.to_csv('AllTrails_Cleaned.csv', index=False)

###################################################
###
### Normalized AllTrails Dataframe (Min/Max)
###
###################################################

print("\nNormalizing AllTrails Dataframe...\n")

# Select numeric columns to scale
normalize_cols = [
    'TrailCount',
    'HardPercent',
    'EasyPercent',
    'AvgLen_mi',
    'AvgElevGain_ft',
    'AvgRating',
    'ReviewCount'
]

# Copy dataframe to preserve original
trails_scaled = trailsdf.copy()

# Initialize scaler
scaler = MinMaxScaler()
trails_scaled[normalize_cols] = scaler.fit_transform(trails_scaled[normalize_cols])

# Show before/after for documentation
print("\nOriginal AllTrails Data (First 5 Rows):\n")
print(trailsdf.head())

print("\nAfter Normalization (First 5 Rows):\n")
print(trails_scaled.head())

# Drop the Country column — only keep quantitative features
trails_kmeans = trails_scaled[normalize_cols].copy()

# Display ready-to-cluster data
print("\nAllTrails Data Prepared for KMeans (First 5 Rows):\n")
print(trails_kmeans.head())

trails_kmeans.to_csv('AllTrails_kmeans.csv', index=False)
print("\nAllTrails data saved as 'AllTrails_kmeans.csv' for clustering.\n")