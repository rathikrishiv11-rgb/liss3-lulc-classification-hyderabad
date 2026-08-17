# LISS-3 LULC Classification — Hyderabad

Research internship project at the Lab for Spatial Informatics (LSI), 
IIIT Hyderabad, under Prof. Kuldeep Kurte, mentored by Sumedha Basu.

## Overview
Random Forest-based Land Use Land Cover classification pipeline 
using LISS-3 satellite imagery across multiple seasonal scenes 
of Hyderabad, India, with rigorous training data validation.

## Study Area
- Location: Hyderabad, Telangana, India
- Bounding box (EPSG:32644): [insert your actual bounds here]
- Scenes used: January 2023, December 2023 (LISS-3, path/row [X])
- Data source: NRSC Bhoonidhi portal

## Methodology
- Bands used: Green, Red, NIR, SWIR (LISS-3)
- Derived indices: NDVI, NDWI, NDBI
- Classifier: Random Forest (scikit-learn), class_weight='balanced'
- Training data: hand-drawn polygons per class, cross-validated 
  against Dynamic World
- Independent spatially-separate test set used to avoid train/
  validation leakage

## Results
- January model: [accuracy]%, kappa [value]
- December model: [accuracy]%, kappa [value]
- Independent test kappa: [value]
- Key finding: single-scene models show limited generalisation 
  across seasons; training class balance directly biases 
  behaviour on unfamiliar scenes

## Reproducing This Work
1. Download LISS-3 imagery for the bounding box above from 
   Bhoonidhi (bhoonidhi.nrsc.gov.in)
2. Place BAND2-5.tif files per the folder structure in 
   `src/liss3_pipeline.py`
3. Draw training polygons in QGIS following the class schema 
   below
4. Run the pipeline

## Class Schema
0 = Water, 1 = Built-up, 2 = Vegetation, 3 = Other, 4 = Clouds

## Tech Stack
Python, rasterio, scikit-learn, geopandas, QGIS, Google Earth Engine

## Acknowledgements
This work was carried out under the guidance of Prof. Kuldeep 
Ramchandra Kurte and with extensive mentorship from Sumedha Basu 
at the Lab for Spatial Informatics, IIIT Hyderabad.
