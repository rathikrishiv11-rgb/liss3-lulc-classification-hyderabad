import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from rasterio.mask import mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, cohen_kappa_score

root_path = r"C:\Users\HP\OneDrive\Attachments\Desktop\LISS3 Extracted"

scenes = {
    "Jan2023_a":  ("Jan2023",  "RA303JAN2023031515010000060PSANSTUC00GTDF"),
    "Jan2023_b":  ("Jan2023",  "RA327JAN2023031856010000060PSANSTUC00GTDF"),
    "Feb2023":    ("Feb2023",  "BH_RA320FEB2023032197010000060PSANSTUC00GTDF"),
    "Mar2023":    ("Mar2023",  "RA316MAR2023032538010000060PSANSTUC00GTDF"),
    "May2023":    ("May2023",  "BH_RA327MAY2023033561010000060PSANSTUC00GTDF"),
    "June2023":   ("June2023", "RA320JUN2023033902010000060PSANSTUC00GTDF"),
    "July2023":   ("July2023", "RA314JUL2023034243010000060PSANSTUC00GTDF"),
    "Aug2023":    ("Aug2023",  "RA331AUG2023034925010000060PSANSTUC00GTDF"),
    "Sept2023":   ("Sept2023", "RA324SEP2023035266010000060PSANSTUC00GTDF"),
    "Nov2023":    ("Nov2023",  "RA311NOV2023035948010000060PSANSTUC00GTDF"),
    "Dec2023_a":  ("Dec2023",  "RA305DEC2023036289010000060PSANSTUC00GTDF"),
    "Dec2023_b":  ("Dec2023",  "RA329DEC2023036630010000060PSANSTUC00GTDF"),
}

# ===============================
# STEP 1 — Load All Scenes
# ===============================

def load_scene(month_folder, scene_folder):
    scene_path = os.path.join(root_path, month_folder, scene_folder)
    try:
        with rasterio.open(os.path.join(scene_path, "BAND2.tif")) as src:
            green = src.read(1).astype(np.float32)
            profile = src.profile
        with rasterio.open(os.path.join(scene_path, "BAND3.tif")) as src:
            red = src.read(1).astype(np.float32)
        with rasterio.open(os.path.join(scene_path, "BAND4.tif")) as src:
            nir = src.read(1).astype(np.float32)
        with rasterio.open(os.path.join(scene_path, "BAND5.tif")) as src:
            swir = src.read(1).astype(np.float32)
        return green, red, nir, swir, profile
    except Exception as e:
        print(f"Error loading {scene_folder}: {e}")
        return None, None, None, None, None

all_composites = {}
all_profiles = {}

for label, (month_folder, scene_folder) in scenes.items():
    print(f"Processing {label} ({month_folder})...")
    green, red, nir, swir, profile = load_scene(month_folder, scene_folder)
    if green is None:
        continue

    eps = 1e-8
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)

    composite = np.stack([green, red, nir, swir, ndvi, ndwi, ndbi], axis=0).astype(np.float32)
    all_composites[label] = composite
    all_profiles[label] = profile
    print(f"{label} composite shape: {composite.shape}")

print(f"\nSuccessfully processed {len(all_composites)} out of {len(scenes)} scenes\n")

# ===============================
# STEP 2 — No-Data Mask Function
# ===============================

def get_valid_mask(composite):
    return composite[0] != 0

# ===============================
# STEP 3 — Export Jan2023_b GeoTIFF For QGIS
# ===============================

def export_geotiff(label, scene_info, composite, out_name):
    month_folder, scene_folder = scene_info
    scene_path = os.path.join(root_path, month_folder, scene_folder)
    with rasterio.open(os.path.join(scene_path, "BAND2.tif")) as src:
        prof = src.profile.copy()
    prof.update(count=7, dtype='float32')
    out_path = os.path.join(root_path, out_name)
    with rasterio.open(out_path, 'w', **prof) as dst:
        for i in range(7):
            dst.write(composite[i], i+1)
    print(f"Saved {out_name}")
    return out_path

def export_classified_geotiff(classified_array, reference_scene_path, output_name, root_path):
    """
    Saves a classified array as a georeferenced GeoTIFF, 
    using the same CRS/transform as the reference composite.
    """
    with rasterio.open(reference_scene_path) as src:
        profile = src.profile.copy()
    
    profile.update(
        count=1,
        dtype='int8',
        nodata=-1
    )
    
    output_path = os.path.join(root_path, output_name)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(classified_array, 1)
    
    print(f"Saved classified GeoTIFF: {output_path}")

jan_b_path = export_geotiff("Jan2023_b", scenes["Jan2023_b"], all_composites["Jan2023_b"], "jan2023_composite.tif")
dec_b_path = export_geotiff("Dec2023_b", scenes["Dec2023_b"], all_composites["Dec2023_b"], "dec2023_composite.tif")
may_path_export = export_geotiff("May2023", scenes["May2023"], all_composites["May2023"], "may2023_composite.tif")
sept_path_export = export_geotiff("Sept2023", scenes["Sept2023"], all_composites["Sept2023"], "sept2023_composite.tif")

# ===============================
# STEP 4 — Load Fresh Training Polygons (Jan2023_b ONLY)
# Draw these in QGIS on training_samples_v2.shp before running this
# ===============================

training_gdf_v2 = gpd.read_file(
    r"C:\Users\HP\OneDrive\Attachments\Desktop\LISS3 Extracted\training_samples_v2.shp"
)

with rasterio.open(jan_b_path) as src:
    target_crs = src.crs

training_gdf_v2 = training_gdf_v2.to_crs(target_crs)

print("Total polygons:", len(training_gdf_v2))
print("Class distribution:\n", training_gdf_v2['class_id'].value_counts())

# ===============================
# STEP 5 — Extract Training Pixels + Verify 100+ Per Class
# ===============================

X_train = []
y_train = []

for idx, row in training_gdf_v2.iterrows():
    class_id = row['class_id']
    geom = [row['geometry']]

    with rasterio.open(jan_b_path) as src:
        try:
            out_image, _ = mask(src, geom, crop=True)
            pixels = out_image.reshape(out_image.shape[0], -1).T
            valid_pixels = pixels[~np.all(pixels == 0, axis=1)]
            for pixel in valid_pixels:
                X_train.append(pixel)
                y_train.append(class_id)
        except Exception as e:
            print(f"Error processing polygon {idx} (class {class_id}): {e}")

X_train = np.array(X_train)
y_train = np.array(y_train)

class_names_map = {0: 'Water', 1: 'Built-up', 2: 'Vegetation', 3: 'Other', 4: 'Clouds'}

print(f"\nTotal training pixels: {len(X_train)}")
print("Number of features per pixel:", X_train.shape[1], "(should be 7)")
print("\n--- Per-Class Pixel Count Check ---")
for cls in sorted(np.unique(y_train)):
    count = np.sum(y_train == cls)
    status = "OK" if count >= 100 else "NEED MORE — GO BACK TO QGIS"
    print(f"  {class_names_map.get(cls, cls)}: {count} pixels [{status}]")

# ==========================
# STEP 6 — Train + Validate
# ==========================

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
rf.fit(X_tr, y_tr)
y_val_pred = rf.predict(X_val)

print("\n--- Validation Accuracy (Jan2023_b, clean training data) ---")
present = sorted(np.unique(y_val))
names = [class_names_map[i] for i in present]
print(classification_report(y_val, y_val_pred, labels=present, target_names=names))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y_val_pred, labels=present)

print("\nConfusion Matrix (rows = True, columns = Predicted)")
print("Classes:", names)
print(cm)

kappa = cohen_kappa_score(y_val, y_val_pred)
print(f"\nOverall Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Kappa Coefficient: {kappa:.4f}")

# Find WHERE in the validation set Other is being misclassified as Built-up
misclassified_mask = (y_val == 3) & (y_val_pred == 1)  # true=Other, predicted=Built-up
print(f"Number of Other→Built-up misclassifications: {misclassified_mask.sum()}")

# Get their spectral values to understand what they look like
misclassified_pixels = X_val[misclassified_mask]
print("Mean spectral values of confused pixels (green,red,nir,swir,ndvi,ndwi,ndbi):")
print(misclassified_pixels.mean(axis=0))

print("\nCompare to average Other pixel:")
other_pixels = X_val[y_val == 3]
print(other_pixels.mean(axis=0))

print("\nCompare to average Built-up pixel:")
builtup_pixels = X_val[y_val == 1]
print(builtup_pixels.mean(axis=0))

# Final model trained on all available clean data
rf_final = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
rf_final.fit(X_train, y_train)
print("\nFeature importances (green,red,nir,swir,ndvi,ndwi,ndbi):")
print(rf_final.feature_importances_)

# ===============================
# STEP 7 — Classify Jan2023_b (the scene you trained on) — PRIMARY DELIVERABLE
# ===============================

jan_b_composite = all_composites["Jan2023_b"]
n_bands, height, width = jan_b_composite.shape
valid_mask_jan_b = get_valid_mask(jan_b_composite)

X_full = jan_b_composite.reshape(n_bands, -1).T
y_pred = rf_final.predict(X_full)
classified_jan_b = y_pred.reshape(height, width).astype(np.int8)
classified_jan_b[~valid_mask_jan_b] = -1

print("\n--- Jan2023_b Classification (own training scene) ---")
valid_pixels = classified_jan_b[classified_jan_b != -1]
total = len(valid_pixels)
unique, counts = np.unique(valid_pixels, return_counts=True)
for cls, count in zip(unique, counts):
    pct = (count/total)*100
    print(f"  {class_names_map[int(cls)]}: {pct:.2f}%")

export_classified_geotiff(classified_jan_b, jan_b_path, "classified_Jan2023_b.tif", root_path)

# Apply the existing Jan2023_b-trained model to a new scene, no retraining
def apply_model_to_scene(scene_label, model, all_composites, get_valid_mask, class_names_map, root_path):
    composite = all_composites[scene_label]
    n_bands, height, width = composite.shape
    valid_mask = get_valid_mask(composite)
    
    X_full = composite.reshape(n_bands, -1).T
    y_pred = model.predict(X_full)
    classified = y_pred.reshape(height, width).astype(np.int8)
    classified[~valid_mask] = -1
    
    valid_pixels = classified[classified != -1]
    total = len(valid_pixels)
    unique, counts = np.unique(valid_pixels, return_counts=True)
    print(f"\n--- {scene_label} (using Jan2023_b-trained model) ---")
    for cls, count in zip(unique, counts):
        pct = (count/total)*100
        print(f"  {class_names_map[int(cls)]}: {pct:.2f}%")
    
    return classified

# Try it on your "too much cloud" and "very clear" picks
classified_sept = apply_model_to_scene("Sept2023", rf_final, all_composites, get_valid_mask, class_names_map, root_path)
classified_may = apply_model_to_scene("May2023", rf_final, all_composites, get_valid_mask, class_names_map, root_path)

# ===============================
# STEP 8 — Visualise Jan2023_b: Reference vs Classified, Side by Side
# ===============================

colors = ['black', 'blue', 'red', 'green', 'yellow', 'white']
cmap = ListedColormap(colors)
class_labels = {-1: 'No Data', 0: 'Water', 1: 'Built-up', 2: 'Vegetation', 3: 'Other', 4: 'Clouds'}

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

rgb = np.stack([jan_b_composite[2], jan_b_composite[1], jan_b_composite[0]], axis=-1).astype(np.float32)
rgb_norm = np.zeros_like(rgb)
for i in range(3):
    band = rgb[:,:,i]
    valid = band[band != 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
        rgb_norm[:,:,i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)

axes[0].imshow(rgb_norm)
axes[0].set_title('Reference — Jan2023_b (NIR-Red-Green)', fontsize=12, fontweight='bold')
axes[0].axis('off')

display = classified_jan_b.astype(np.int16) + 1
axes[1].imshow(display, cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
axes[1].set_title('LULC Classification — Jan2023_b (Clean Training Data)', fontsize=12, fontweight='bold')
axes[1].axis('off')

patches = [
    mpatches.Patch(color='black', label='No Data'),
    mpatches.Patch(color='blue', label='Water'),
    mpatches.Patch(color='red', label='Built-up'),
    mpatches.Patch(color='green', label='Vegetation'),
    mpatches.Patch(color='yellow', label='Other'),
    mpatches.Patch(color='white', label='Clouds')
]
axes[1].legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.savefig(os.path.join(root_path, 'classified_Jan2023_b_v2_clean.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved classified_Jan2023_b_v2_clean.png")

# ===============================
# INDEPENDENT TEST SET — January (Water, Built-up, Vegetation only)
# Other and Clouds skipped per Sumedha — insufficient separate area
# ===============================

test_gdf_jan = gpd.read_file(
    r"C:\Users\HP\OneDrive\Attachments\Desktop\LISS3 Extracted\test_samples_jan_clean.shp"
)

with rasterio.open(jan_b_path) as src:
    test_gdf_jan = test_gdf_jan.to_crs(src.crs)

print("\n--- Independent Test Polygons (Water/Built-up/Vegetation) ---")
print("Test polygons:", len(test_gdf_jan))
print("Test class distribution:\n", test_gdf_jan['class_id'].value_counts())

X_test_jan = []
y_test_jan = []

for idx, row in test_gdf_jan.iterrows():
    class_id = row['class_id']
    geom = [row['geometry']]
    with rasterio.open(jan_b_path) as src:
        try:
            out_image, _ = mask(src, geom, crop=True)
            pixels = out_image.reshape(out_image.shape[0], -1).T
            valid_pixels = pixels[~np.all(pixels == 0, axis=1)]
            for pixel in valid_pixels:
                X_test_jan.append(pixel)
                y_test_jan.append(class_id)
        except Exception as e:
            print(f"Error processing test polygon {idx}: {e}")

X_test_jan = np.array(X_test_jan)
y_test_jan = np.array(y_test_jan)

print(f"\nTotal independent test pixels: {len(X_test_jan)}")
for cls in sorted(np.unique(y_test_jan)):
    print(f"  {class_names_map.get(cls, cls)}: {np.sum(y_test_jan==cls)} pixels")

# Train on ALL existing Jan2023_b training data (full set, no internal split)
rf_indep = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
rf_indep.fit(X_train, y_train)

# Predict only on the independent test pixels
y_pred_indep = rf_indep.predict(X_test_jan)

print("\n--- INDEPENDENT Test Set Results (Jan2023_b, spatially separate polygons) ---")
present_indep = sorted(np.unique(y_test_jan))
names_indep = [class_names_map[i] for i in present_indep]
print(classification_report(y_test_jan, y_pred_indep, labels=present_indep, target_names=names_indep))

cm_indep = confusion_matrix(y_test_jan, y_pred_indep, labels=present_indep)
print("\nConfusion Matrix (Independent Test):")
print(names_indep)
print(cm_indep)

print(f"\nIndependent Test Accuracy: {accuracy_score(y_test_jan, y_pred_indep):.4f}")
print(f"Independent Test Kappa: {cohen_kappa_score(y_test_jan, y_pred_indep):.4f}")

print("\n--- Comparison ---")
print(f"Original validation (pixel-split) kappa: 0.9718")
print(f"Independent (spatially separate) test kappa: {cohen_kappa_score(y_test_jan, y_pred_indep):.4f}")

dec_label = "Dec2023_b"
# ===============================
# Load Training Polygons — December
# ===============================

training_gdf_dec = gpd.read_file(
    r"C:\Users\HP\OneDrive\Attachments\Desktop\LISS3 Extracted\training_samples_winter.shp"
)

with rasterio.open(dec_b_path) as src:
    dec_crs = src.crs

training_gdf_dec = training_gdf_dec.to_crs(dec_crs)

print("\n--- December Training Data ---")
print("Total polygons:", len(training_gdf_dec))
print("Class distribution:\n", training_gdf_dec['class_id'].value_counts())

# ===============================
# Extract Training Pixels — December
# ===============================

X_train_dec = []
y_train_dec = []

for idx, row in training_gdf_dec.iterrows():
    class_id = row['class_id']
    geom = [row['geometry']]
    with rasterio.open(dec_b_path) as src:
        try:
            out_image, _ = mask(src, geom, crop=True)
            pixels = out_image.reshape(out_image.shape[0], -1).T
            valid_pixels = pixels[~np.all(pixels == 0, axis=1)]
            for pixel in valid_pixels:
                X_train_dec.append(pixel)
                y_train_dec.append(class_id)
        except Exception as e:
            print(f"Error processing polygon {idx}: {e}")

X_train_dec = np.array(X_train_dec)
y_train_dec = np.array(y_train_dec)

class_names_map = {0: 'Water', 1: 'Built-up', 2: 'Vegetation', 3: 'Other', 4: 'Clouds'}

print(f"\nTotal training pixels: {len(X_train_dec)}")
print("Number of features per pixel:", X_train_dec.shape[1])
print("\n--- Per-Class Pixel Count Check ---")
for cls in sorted(np.unique(y_train_dec)):
    count = np.sum(y_train_dec == cls)
    status = "OK" if count >= 100 else "NEED MORE — GO BACK TO QGIS"
    print(f"  {class_names_map.get(cls, cls)}: {count} pixels [{status}]")

# ===============================
# Train + Validate — December
# ===============================

X_tr_dec, X_val_dec, y_tr_dec, y_val_dec = train_test_split(
    X_train_dec, y_train_dec, test_size=0.2, random_state=42, stratify=y_train_dec
)

rf_dec = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
rf_dec.fit(X_tr_dec, y_tr_dec)
y_val_pred_dec = rf_dec.predict(X_val_dec)

print("\n--- Validation Accuracy (December, clean training data) ---")
present_dec = sorted(np.unique(y_val_dec))
names_dec = [class_names_map[i] for i in present_dec]
print(classification_report(y_val_dec, y_val_pred_dec, labels=present_dec, target_names=names_dec))

cm_dec = confusion_matrix(y_val_dec, y_val_pred_dec, labels=present_dec)
print("\nConfusion Matrix (rows = True, columns = Predicted)")
print("Classes:", names_dec)
print(cm_dec)

from sklearn.metrics import accuracy_score, cohen_kappa_score
kappa_dec = cohen_kappa_score(y_val_dec, y_val_pred_dec)
print(f"\nOverall Accuracy: {accuracy_score(y_val_dec, y_val_pred_dec):.4f}")
print(f"Kappa Coefficient: {kappa_dec:.4f}")

# Final model trained on all December data
rf_dec_final = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
rf_dec_final.fit(X_train_dec, y_train_dec)

# ===============================
# Classify December Scene
# ===============================

dec_composite = all_composites[dec_label]
n_bands_dec, height_dec, width_dec = dec_composite.shape
valid_mask_dec = get_valid_mask(dec_composite)

X_full_dec = dec_composite.reshape(n_bands_dec, -1).T
y_pred_dec = rf_dec_final.predict(X_full_dec)
classified_dec = y_pred_dec.reshape(height_dec, width_dec).astype(np.int8)
classified_dec[~valid_mask_dec] = -1

print(f"\n--- {dec_label} Classification (own training scene) ---")
valid_pixels_dec = classified_dec[classified_dec != -1]
total_dec = len(valid_pixels_dec)
unique_dec, counts_dec = np.unique(valid_pixels_dec, return_counts=True)
for cls, count in zip(unique_dec, counts_dec):
    pct = (count/total_dec)*100
    print(f"  {class_names_map[int(cls)]}: {pct:.2f}%")

export_classified_geotiff(classified_dec, dec_b_path, "classified_Dec2023_b.tif", root_path)

# Get spectral values from the specific over-predicted region

region_pixels = dec_composite[:, 1000:1200, 2500:2700]
region_pixels_flat = region_pixels.reshape(7, -1).T
valid_region = region_pixels_flat[~np.all(region_pixels_flat == 0, axis=1)]

print("Mean spectral values in over-predicted region:")
print(valid_region.mean(axis=0))
print("\nCompare to Dec Other average:")
print(X_train_dec[y_train_dec == 3].mean(axis=0))
print("\nCompare to Dec Built-up average:")
print(X_train_dec[y_train_dec == 1].mean(axis=0))

# ===============================
# Visualise — December: Reference vs Classified
# ===============================

colors_dec = ['black', 'blue', 'red', 'green', 'yellow', 'white']
cmap_dec = ListedColormap(colors_dec)
labels_dec = {-1: 'No Data', 0: 'Water', 1: 'Built-up', 2: 'Vegetation', 3: 'Other', 4: 'Clouds'}

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

rgb_dec = np.stack([dec_composite[2], dec_composite[1], dec_composite[0]], axis=-1).astype(np.float32)
rgb_dec_norm = np.zeros_like(rgb_dec)
for i in range(3):
    band = rgb_dec[:,:,i]
    valid = band[band != 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
        rgb_dec_norm[:,:,i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)

axes[0].imshow(rgb_dec_norm)
axes[0].set_title(f'Reference — {dec_label} (NIR-Red-Green)', fontsize=12, fontweight='bold')
axes[0].axis('off')

display_dec = classified_dec.astype(np.int16) + 1
axes[1].imshow(display_dec, cmap=cmap_dec, vmin=0, vmax=5, interpolation='nearest')
axes[1].set_title(f'LULC Classification — {dec_label} (Winter)', fontsize=12, fontweight='bold')
axes[1].axis('off')

patches_dec = [mpatches.Patch(color=colors_dec[i], label=labels_dec[i-1]) for i in range(6)]
axes[1].legend(handles=patches_dec, loc='lower right', fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.savefig(os.path.join(root_path, f'classified_{dec_label}_v2_clean.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved classified_{dec_label}_v2_clean.png")

# ===============================
# CROSS-SEASON GENERALIZATION CHECK
# Apply BOTH January and December models to May and Sept
# ===============================

def apply_and_report(scene_label, model, model_name, all_composites, get_valid_mask, class_names_map):
    composite = all_composites[scene_label]
    n_bands, height, width = composite.shape
    valid_mask = get_valid_mask(composite)
    
    X_full = composite.reshape(n_bands, -1).T
    y_pred = model.predict(X_full)
    classified = y_pred.reshape(height, width).astype(np.int8)
    classified[~valid_mask] = -1
    
    valid_pixels = classified[classified != -1]
    total = len(valid_pixels)
    unique, counts = np.unique(valid_pixels, return_counts=True)
    
    print(f"\n--- {scene_label} using {model_name}-trained model ---")
    for cls, count in zip(unique, counts):
        pct = (count/total)*100
        print(f"  {class_names_map[int(cls)]}: {pct:.2f}%")
    
    return classified

# December model applied to May and Sept (January's results already have this from before)
classified_may_dec = apply_and_report("May2023", rf_dec_final, "December", all_composites, get_valid_mask, class_names_map)
classified_sept_dec = apply_and_report("Sept2023", rf_dec_final, "December", all_composites, get_valid_mask, class_names_map)

print("\n" + "="*60)
print("SUMMARY — CROSS-SEASON COMPARISON")
print("="*60)
print("\nMay2023:")
print("  January-model → Built-up 15.46%, Vegetation 0.06%, Other 76.34%, Clouds 8.14%")
print("  December-model → [check output above]")
print("\nSept2023:")
print("  January-model → Built-up 50.41%, Vegetation 0.00%, Other 10.74%, Clouds 38.85%")
print("  December-model → [check output above]")

# ===============================
# VISUALISE — May & Sept Under December Model (and January model too)
# ===============================

def visualize_cross_scene(scene_label, classified, all_composites, save_suffix, model_name):
    composite = all_composites[scene_label]
    
    colors = ['black', 'blue', 'red', 'green', 'yellow', 'white']
    cmap = ListedColormap(colors)
    labels = {-1: 'No Data', 0: 'Water', 1: 'Built-up', 2: 'Vegetation', 3: 'Other', 4: 'Clouds'}
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    rgb = np.stack([composite[2], composite[1], composite[0]], axis=-1).astype(np.float32)
    rgb_norm = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[:,:,i]
        valid = band[band != 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
            rgb_norm[:,:,i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    axes[0].imshow(rgb_norm)
    axes[0].set_title(f'Reference — {scene_label} (NIR-Red-Green)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    display = classified.astype(np.int16) + 1
    axes[1].imshow(display, cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
    axes[1].set_title(f'LULC Classification — {scene_label} ({model_name}-trained model)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    patches = [mpatches.Patch(color=colors[i], label=labels[i-1]) for i in range(6)]
    axes[1].legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.8)
    
    plt.tight_layout()
    out_path = os.path.join(root_path, f'classified_{scene_label}_{save_suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

# December-model results (already have these classified arrays from your last run)
visualize_cross_scene("May2023", classified_may_dec, all_composites, "dec_model", "December")
visualize_cross_scene("Sept2023", classified_sept_dec, all_composites, "dec_model", "December")

# January-model results — recreate these since you only printed percentages before, didn't save the arrays
classified_sept_jan = apply_and_report("Sept2023", rf_final, "January", all_composites, get_valid_mask, class_names_map)
classified_may_jan = apply_and_report("May2023", rf_final, "January", all_composites, get_valid_mask, class_names_map)

visualize_cross_scene("May2023", classified_may_jan, all_composites, "jan_model", "January")
visualize_cross_scene("Sept2023", classified_sept_jan, all_composites, "jan_model", "January")

export_classified_geotiff(classified_may_dec, jan_b_path, "classified_May2023_using_Dec_model.tif", root_path)
export_classified_geotiff(classified_sept_dec, jan_b_path, "classified_Sept2023_using_Dec_model.tif", root_path)
export_classified_geotiff(classified_may_jan, jan_b_path, "classified_May2023_using_Jan_model.tif", root_path)
export_classified_geotiff(classified_sept_jan, jan_b_path, "classified_Sept2023_using_Jan_model.tif", root_path)