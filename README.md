# 🥾 Trailblazing Tourism: A Data Science Look at Global Hiking Destinations

A data science project analyzing what drives international hiking tourism — combining trail data, tourism arrivals, and 25 years of climate history to understand why people travel where they do.

## Overview

International tourism represents roughly 10% of the global economy, and hiking is one of its fastest-growing segments. This project asks: **what actually predicts a country's hiking tourism** — trail difficulty, climate, accessibility, or something else? To answer it, I gathered and merged three independent datasets, built a full cleaning/normalization pipeline, and applied both unsupervised (K-Means clustering) and supervised (decision tree) machine learning models.

I grew up in Moab, Utah, a town built around outdoor tourism, and wanted to explore the patterns behind why people travel for hiking the way they do.

## Research Questions

- Which countries attract the most hiking-focused international tourists?
- Which countries attract the most difficult/advanced hikes — and the most beginner-friendly ones?
- Can trail features (length, elevation, difficulty) predict a trail's popularity?
- Do tourism flows correlate with climate patterns (temperature, precipitation, sunshine)?

## Data Sources

| Source | Data |
|---|---|
| [AllTrails Trek Data (Kaggle)](https://www.kaggle.com/datasets/ashishgatreddi/alltrails-trek-data) | Trail count, difficulty, length, elevation, ratings by country |
| [World Bank Open Data](https://data.worldbank.org/indicator/ST.INT.ARVL) | International tourist arrivals by country, 1995–2020 |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-forecast-api) | Daily temperature, precipitation, and sunshine data, gathered live per country/year |

## Pipeline

**1. Data Gathering**
Pulled 25 years of climate data per country via the Open-Meteo API, handling rate limits with retry logic and resumable checkpoints across multiple runs.

**2. Data Cleaning**
Cleaned and standardized three independent datasets — resolving mismatched country names, handling missing values via interpolation, and filtering to a consistent country set across all sources.

**3. Normalization**
Applied Min-Max scaling to prep numeric features for clustering.

**4. Unsupervised Learning — K-Means Clustering**
Clustered trails by difficulty, length, and popularity to identify natural groupings in the global trail landscape.

**5. Supervised Learning — Decision Tree**
Built a classifier to predict a country's trail density category (Low / Moderate / High) from trail features.

**6. Final Merged Analysis**
Combined all three cleaned datasets to compare climate, trail availability, and tourism volume across the top and bottom 10 most-visited countries.

## Key Findings

- Climate influences **when** people travel more than **whether** they travel — even countries with extreme rainfall or temperatures still draw visitors.
- Countries with large trail systems tend to have more easy-to-moderate trails, shaped by elevation and length, rather than a high proportion of extreme/difficult terrain.
- There's a clear relationship between trail infrastructure, accessibility, and overall tourism volume.

## Tech Stack

`Python` · `pandas` · `scikit-learn` (K-Means, Decision Trees) · `seaborn` / `matplotlib` · `Open-Meteo API`

## Full Report

See [`CS332_Full_Report.pdf`](./CS332_Full_Report.pdf) for the complete write-up, including all visualizations, methodology detail, and source citations.

---
*Built for CS 332: Intro to Data Science at Oregon State University.*
