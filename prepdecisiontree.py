# -*- coding: utf-8 -*-
"""
Creating labeled version of AllTrails dataset for supervised modeling later.
"""

import pandas as pd

# Read the already cleaned dataset (not the normalized one)
trailsfile = "AllTrails_Cleaned.csv"
alltrailsdf = pd.read_csv(trailsfile)

print("\nBefore labeling:\n")
print(alltrailsdf.head())

# -------------------------
# Create TrailLevel label using binning
# -------------------------

bins = [-1, 500, 5000, float('inf')]
labels = ['Low Trails', 'Moderate Trails', 'High Trails']

alltrailsdf['TrailLevel'] = pd.cut(alltrailsdf['TrailCount'], bins=bins, labels=labels)

print("\nAfter labeling:\n")
print(alltrailsdf[['Country', 'TrailCount', 'TrailLevel']].head())

# -------------------------
# Save labeled version
# -------------------------

alltrailsdf.to_csv("AllTrails_Labeled.csv", index=False)

print("\nLabeled dataset saved as: AllTrails_Labeled.csv\n")