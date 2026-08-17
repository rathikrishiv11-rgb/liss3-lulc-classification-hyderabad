# LISS-3 LULC Classification — Hyderabad

Research internship project at the Lab for Spatial Informatics (LSI), 
IIIT Hyderabad, under Prof. Kuldeep Ramchandra Kurte, mentored by Sumedha Basu.

## Overview

Random Forest-based Land Use Land Cover (LULC) classification pipeline 
using LISS-3 satellite imagery across multiple seasonal scenes of Hyderabad, 
India, with training-data validation and an independent spatial test set.

## Study Area

- Location: Hyderabad, Telangana, India
- Data source: NRSC Bhoonidhi portal
- CRS: EPSG:32644 (UTM Zone 44N)

### January 2023 Scene

- Longitude: 78.012578 to 79.792674
- Latitude: 17.112110 to 18.666139

### December 2023 Scene

- Longitude: 77.984374 to 79.761541
- Latitude: 17.112133 to 18.667451

- Scenes processed: 12 LISS-3 scenes from 2023
- Training models: January 2023 and December 2023

## Methodology

- Bands used: Green, Red, NIR, SWIR
- Derived indices: NDVI, NDWI, NDBI
- Classifier: Random Forest (scikit-learn)
- Training data: Hand-drawn polygons
- Classes: Water, Built-up, Vegetation, Other, Clouds
- Independent spatially separate test polygons used to evaluate 
  generalisation without spatial overlap with training data

## Results

### January 2023 — Validation

- Overall Accuracy: 97.40%
- Kappa Coefficient: 0.9644

### January 2023 — Independent Spatial Test

- Overall Accuracy: 92.71%
- Kappa Coefficient: 0.8982

### December 2023 — Validation

- Overall Accuracy: 99.70%
- Kappa Coefficient: 0.9953

### Key Findings

- January showed Built-up/Other spectral confusion.
- Independent spatial testing produced lower accuracy than the 
  pixel-split validation, indicating the importance of spatially 
  separate testing.
- Single-scene models showed limited generalisation when applied 
  to different seasonal scenes.
- Other and Clouds were the most difficult classes to generalise 
  across unfamiliar scenes.

## Class Schema

| Class ID | Class |
|----------|-------|
| 0 | Water |
| 1 | Built-up |
| 2 | Vegetation |
| 3 | Other |
| 4 | Clouds |

## Training Data Summary

### January (Jan2023_b)

- Shapefile: `training_samples_v2.shp`
- CRS: EPSG:32644
- Water (0): 5 polygons, 35,576 pixels
- Built-up (1): 9 polygons, 21,175 pixels
- Vegetation (2): 7 polygons, 34,738 pixels
- Other (3): 17 polygons, 15,464 pixels
- Clouds (4): 24 polygons, 1,435 pixels
- Notes: Cross-checked against Dynamic World for class accuracy.
  Built-up/Other confusion was identified during evaluation.

### December (Dec2023_b)

- Shapefile: Training polygons used for December classification
- CRS: EPSG:32644
- Water (0): 6 polygons, 49,306 pixels
- Built-up (1): 8 polygons, 15,792 pixels
- Vegetation (2): 7 polygons, 22,380 pixels
- Other (3): 10 polygons, 1,301 pixels
- Clouds (4): 20 polygons, 6,618 pixels

### Independent Test Set (Jan2023_b)

- Shapefile: `test_samples_jan.shp`
- Spatially separate from training polygons
- Used to evaluate model performance on independent data
- Water: 6,372 pixels
- Built-up: 6,682 pixels
- Vegetation: 6,632 pixels
- Other: 1,315 pixels
- Clouds: 373 pixels

## Known Issues / Findings

1. Built-up/Other confusion in January due to spectral overlap.
2. Single-scene models showed limited cross-season generalisation.
3. The December model strongly over-predicted Clouds on unfamiliar scenes.
4. Other and Clouds remained the hardest classes to generalise.

## Repository Structure

```text
LISS3-LULC-Classification/
│
├── Code/
│   ├── liss3_pipeline.py
│
└── README.md
