# Deforestation-Detection
A quick and easy guide to creating your first ML project detecting deforestation in one region

# 🛰️ Near-Real-Time Deforestation Monitoring System

An end-to-end computer vision and satellite remote sensing pipeline designed to detect canopy loss in tropical rainforests. This project leverages European Space Agency (ESA) Sentinel-2 multispectral imagery, a 4-channel deep learning U-Net architecture, post-classification comparison (PCC), and an interactive Streamlit web dashboard.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Core Geospatial & Remote Sensing Concepts](#-core-geospatial--remote-sensing-concepts)
3. [End-to-End Architecture](#-end-to-end-architecture)
4. [Step-by-Step Implementation](#-step-by-step-implementation)
   - [Phase 1: Satellite Data Acquisition & Cloud Filtering (GEE)](#phase-1-satellite-data-acquisition--cloud-filtering-gee)
     - [1.1 Conceptual Background & Why GEE Matters](#11-conceptual-background--why-gee-matters)
     - [1.2 Deep-Dive: Cloud Masking via the QA60 Bitmask](#12-deep-dive-cloud-masking-via-the-qa60-bitmask)
     - [1.3 Multi-Temporal Dry-Season Median Compositing](#13-multi-temporal-dry-season-median-compositing)
     - [1.4 Tutorial: Setting Up & Running Phase 1](#14-tutorial-setting-up--running-phase-1)
   - [Phase 2: Local Verification & Band Combinations](#phase-2-local-verification--band-combinations)
   - [Phase 3: Tiling & Pseudo-Label Generation](#phase-3-tiling--pseudo-label-generation)
   - [Phase 4: 4-Channel U-Net Segmentation](#phase-4-4-channel-u-net-segmentation)
   - [Phase 5: Change Detection & Vectorization](#phase-5-change-detection--vectorization)
   - [Phase 6: Interactive Streamlit Dashboard](#phase-6-interactive-streamlit-dashboard)
5. [Repository Structure](#-repository-structure)
6. [Local Setup & How to Run](#-local-setup--how-to-run)
7. [Key Insights & Lessons Learned](#-key-insights--lessons-learned)

---

## 🌍 Project Overview

Deforestation in critical biomes like the Amazon Basin occurs at massive scales and often follows distinct geometric patterns (such as "fishbone" roads cut through dense canopy). Ground patrols cannot cover millions of hectares in real time. 

This project implements an automated, open-source pipeline using 100% free cloud tooling (Google Earth Engine and Google Colab T4 GPU) to:
- Ingest multi-temporal satellite data two years apart ($T_1 = 2022$ and $T_2 = 2024$).
- Strip clouds and atmospheric noise.
- Train a deep convolutional network to classify forest cover at 10-meter resolution.
- Isolate newly cleared forest parcels and compute cleared land area in hectares.
- Present the findings in an interactive map dashboard.

---

## 🔬 Core Geospatial & Remote Sensing Concepts

### 1. How Satellites "See" (Beyond RGB)
Standard smartphone cameras only capture visible light across three bands: **Red, Green, and Blue**. Satellite sensors like Sentinel-2 carry multispectral radiometers that split light into multiple discrete wavelengths:

| Band Name | Sentinel-2 Band | Center Wavelength | Spatial Resolution | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Blue** | B2 | 490 nm | 10 meters | Atmospheric scattering, water bodies |
| **Green** | B3 | 560 nm | 10 meters | Vegetation discrimination |
| **Red** | B4 | 665 nm | 10 meters | Chlorophyll absorption |
| **NIR (Near-Infrared)** | B8 | 842 nm | 10 meters | **Cellular plant reflectance** |

> **Why NIR is Essential:** Healthy green leaves absorb blue and red light for photosynthesis, but the internal spongy mesophyll cells of leaf tissue strongly reflect **Near-Infrared (NIR)** radiation. To the human eye, healthy forest and green artificial turf look identical; in NIR, the forest glows intensely while soil, concrete, and dead grass absorb or scatter it weakly.

### 2. GeoTIFF vs. Standard Image Formats
Satellite images are not saved as standard `.jpg` or `.png` files. They are saved as **GeoTIFFs (`.tif`)**. A GeoTIFF embeds spatial metadata directly into the raster file:
- **CRS (Coordinate Reference System):** Standard geographic coordinate systems (e.g., `EPSG:4326` for WGS84 latitude/longitude).
- **Affine Transform:** Maps pixel row/column numbers $(y, x)$ to real-world latitude/longitude coordinates on Earth.
- **Multi-band Stacking:** Stores raw sensor measurement values across 4 or more channels per pixel rather than simple 8-bit color channels.

---

## 🏗️ End-to-End Architecture

```text
  ┌────────────────────────────────────────────────────────┐
  │ Google Earth Engine (Cloud)                           │
  │  - Filter Sentinel-2 L2A Surface Reflectance           │
  │  - Mask clouds using QA60 bitmask                      │
  │  - Median dry-season compositing (2022 vs 2024)        │
  │  - Export 4-band GeoTIFFs (B2, B3, B4, B8) to Drive    │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Google Colab Data Prep (NumPy & Rasterio)              │
  │  - Validate pixel coordinate alignment & dimensions    │
  │  - Windowed tiling into 256x256 non-overlapping tiles  │
  │  - Generate binary masks (NDVI > 0.6 thresholding)     │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Model Training & Inference (PyTorch & SMP)            │
  │  - 4-Channel U-Net (ResNet-34 ImageNet Backbone)       │
  │  - Combined BCEWithLogitsLoss + DiceLoss               │
  │  - Sliding-window raster inference on full scenes      │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Change Detection & Vectorization (SciPy & GeoPandas)   │
  │  - Boolean change: Deforestation = (M_2022 & ~M_2024)   │
  │  - Morphological 3x3 binary opening (noise removal)    │
  │  - Convert raster shapes to GeoJSON polygon boundaries │
  │  - Calculate cleared hectares per parcel               │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ Presentation (Streamlit + Folium)                      │
  │  - Dual-layer Sentinel-2 image toggles                 │
  │  - Deforestation alert boundary overlay                │
  │  - Summary metric KPI cards & parcel tables            │
  └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Step-by-Step Implementation

### Phase 1: Satellite Data Acquisition & Cloud Filtering (GEE)

#### 1.1 Conceptual Background & Why GEE Matters

In traditional remote sensing workflows, gathering satellite imagery was manual, slow, and resource-intensive:
1. Search government portals (e.g., Copernicus Open Access Hub or USGS EarthExplorer).
2. Download 15–25 GB of raw `.SAFE` or `.zip` archives containing full $100\text{ km} \times 100\text{ km}$ orbital tiles.
3. Locally execute atmospheric correction processors (like ESA's Sen2Cor) to convert Top-Of-Atmosphere (TOA) Level-1C products into Bottom-Of-Atmosphere (BOA) Level-2A surface reflectance.
4. Reproject and manually clip tiles to the target study area using desktop GIS tools.
5. Suffer system crashes due to high RAM usage when loading and stacking rasters for compositing.

**Google Earth Engine (GEE)** bypasses this entirely:
- **Server-Side Computation:** Processing queries are parallelized across Google's distributed data centers.
- **Harmonized Surface Reflectance:** The `COPERNICUS/S2_SR_HARMONIZED` collection provides Level-2A data where sensor offsets and atmospheric scattering corrections are already calculated.
- **On-the-Fly Filtering:** Only the requested bounds ($10\text{ km} \times 10\text{ km}$), spectral bands (B2, B3, B4, B8), and dates are retrieved, transferring small, optimized outputs directly to Google Drive.

---

#### 1.2 Deep-Dive: Cloud Masking via the QA60 Bitmask

Tropical rainforests are frequently shrouded in cloud cover. Optical satellite sensors cannot penetrate dense clouds; a bright, opaque cloud over a forest parcel reflects high visible light and low NIR, mimicking exposed barren soil and triggering false change alerts.

Sentinel-2 includes an onboard 60-meter Quality Assessment band (`QA60`). Every pixel in this band contains a 16-bit unsigned integer that stores quality flags:

$$\text{Bit 10} \implies \text{Opaque Clouds} \quad (0 = \text{Clear}, \, 1 = \text{Cloudy})$$
$$\text{Bit 11} \implies \text{Cirrus Clouds} \quad (0 = \text{Clear}, \, 1 = \text{Cirrus})$$

##### Bitwise Operations Explained
We inspect bits 10 and 11 using bitwise operations:

1. **Shift Left (`<<`):**
   - `1 << 10` evaluates to binary `0000010000000000` (decimal `1024`).
   - `1 << 11` evaluates to binary `0000100000000000` (decimal `2048`).
2. **Bitwise AND (`bitwiseAnd`):**
   Comparing the pixel's `QA60` integer value to `1 << 10` isolates bit 10, setting all other 14 bits to 0.
3. **Equality Evaluation (`.eq(0)`):**
   If bit 10 is 0, the sky is clear of opaque clouds.
4. **Boolean Conjunction (`.And(...)`):**
   Requiring both bit 10 and bit 11 to equal 0 ensures the pixel is free of both dense clouds and high-altitude cirrus.
5. **Mask Application (`updateMask`):**
   Pixels failing this check are assigned `null` (`NoData`). Rather than painting clouds white or black, the mask punches transparent holes in the image at cloud locations.

```python
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    # Pixel is clean if both bits are 0
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    
    # Strip clouds to null, preserve 10m bands
    return image.updateMask(mask).select(['B2', 'B3', 'B4', 'B8'])
```

---

#### 1.3 Multi-Temporal Dry-Season Median Compositing

Sentinel-2 has a 5-day revisit rate at the equator. Over a two-month window (e.g., July 1 to August 31), the satellite passes over the exact same coordinates roughly 10–12 times.

##### Why July–August (Dry Season)?
In the Amazon Basin (Rondônia, Brazil), the rainy season produces near-constant cloud coverage. Restricting the acquisition window to the dry season provides more cloud-free passes. It also prevents false detections caused by seasonal ground drying or rainfall-induced pasture greening.

##### Why the Statistical Median?
After applying `mask_s2_clouds`, each individual pass contains empty `NoData` holes where clouds used to be. GEE stacks these passes vertically across time for each spatial coordinate $(x, y)$.

Running `.median()` computes the median value among valid, non-null observations:

$$\text{Pixel}(x, y) = \text{median}\Big(\big[\text{Pass}_1(x, y),\, \text{Pass}_2(x, y),\, \dots,\, \text{Pass}_n(x, y)\big]\Big)$$

```text
Stack of passes for pixel (x, y):
Pass 1: [ CLOUD (Dropped to null) ]
Pass 2: [ 0.22 (Surface Reflectance) ]
Pass 3: [ 0.21 (Surface Reflectance) ]
Pass 4: [ CLOUD (Dropped to null) ]
Pass 5: [ 0.24 (Surface Reflectance) ]
-------------------------------------------------------------
Resulting Median: 0.22 (Cloud-free surface measurement)
```

Transient clouds, moving aircraft, and passing shadows drop out, yielding a seamless, cloud-free composite over the entire $10\text{ km} \times 10\text{ km}$ area.

---

#### 1.4 Tutorial: Setting Up & Running Phase 1

Follow these steps to reproduce Phase 1 and export the clean 2022 and 2024 GeoTIFFs to Google Drive.

##### Step 1: Initialize Google Earth Engine
1. Register for free academic/non-commercial access at [earthengine.google.com](https://earthengine.google.com/).
2. Select **Contributor access** and create an unpaid Google Cloud Project (e.g., `deforestation-detection-gee`).
3. Note your **Google Cloud Project ID** from the upper-right corner of the GEE Code Editor.

##### Step 2: Open Google Colab & Install Requirements
Create a new notebook in Google Colab and install the Earth Engine client:

```python
!pip install -q earthengine-api geemap
```

##### Step 3: Run the Acquisition and Export Script
Paste and execute the script below. Replace `'your-project-id'` with your Google Cloud Project ID:

```python
import ee
import geemap

# 1. Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='your-project-id')

# 2. Define Area of Interest (AOI) - Rondônia, Brazil (~10km x 10km)
# Coordinates format: [min_lon, min_lat, max_lon, max_lat]
aoi = ee.Geometry.Rectangle([-62.85, -10.55, -62.75, -10.45])

# 3. Define cloud-masking function using QA60 bitmask
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).select(['B2', 'B3', 'B4', 'B8'])

# 4. Filter, mask, and composite Baseline Scene T1 (July-August 2022)
s2_2022 = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2022-07-01', '2022-08-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
    .median()
    .clip(aoi)
)

# 5. Filter, mask, and composite Post-Clearing Scene T2 (July-August 2024)
s2_2024 = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2024-07-01', '2024-08-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
    .median()
    .clip(aoi)
)

# 6. Queue Export for 2022 to Google Drive
task_2022 = ee.batch.Export.image.toDrive(
    image=s2_2022,
    description='Sentinel2_Rondonia_2022',
    folder='Deforestation_Project',
    fileNamePrefix='s2_2022_b2_b3_b4_b8',
    region=aoi,
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e9
)
task_2022.start()

# 7. Queue Export for 2024 to Google Drive
task_2024 = ee.batch.Export.image.toDrive(
    image=s2_2024,
    description='Sentinel2_Rondonia_2024',
    folder='Deforestation_Project',
    fileNamePrefix='s2_2024_b2_b3_b4_b8',
    region=aoi,
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e9
)
task_2024.start()

print("Export tasks initiated successfully. Monitor progress in the GEE Code Editor Tasks tab.")
```

##### Step 4: Verify the Export in Google Drive
1. Open [code.earthengine.google.com](https://code.earthengine.google.com/) and check the **Tasks** tab in the right-hand panel. The tasks will transition from ready to running to completed (indicated by a blue checkmark).
2. Within 2–5 minutes, a folder named `Deforestation_Project` will appear in your Google Drive root containing:
   - `s2_2022_b2_b3_b4_b8.tif` (~20–40 MB)
   - `s2_2024_b2_b3_b4_b8.tif` (~20–40 MB)

---

### Phase 2: Local Verification & Band Combinations

Once exported to Google Drive, verify that both rasters align down to the pixel level and visually confirm forest clearing:

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt

path_2022 = '/content/drive/MyDrive/Deforestation_Project/s2_2022_b2_b3_b4_b8.tif'
path_2024 = '/content/drive/MyDrive/Deforestation_Project/s2_2024_b2_b3_b4_b8.tif'

with rasterio.open(path_2022) as src_22, rasterio.open(path_2024) as src_24:
    assert src_22.crs == src_24.crs, "CRS mismatch!"
    assert src_22.shape == src_24.shape, "Dimensions mismatch!"
    print(f"Alignment Verified: {src_22.width}x{src_22.height} pixels at 10m resolution.")
```

Calculate the Normalized Difference Vegetation Index (NDVI) to verify real change before modeling:

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{B_8 - B_4}{B_8 + B_4}$$

$$\Delta\text{NDVI} = \text{NDVI}_{2022} - \text{NDVI}_{2024}$$

Positive $\Delta\text{NDVI}$ values indicate vegetation loss, revealing logging tracks and pastures that were cleared between 2022 and 2024.

---

### Phase 3: Tiling & Pseudo-Label Generation

Feeding an entire $1100 \times 1100 \times 4$ floating-point image into a deep network can cause Out-of-Memory (OOM) errors on consumer or free cloud GPUs. We divide the full raster into regular, non-overlapping $256 \times 256 \times 4$ patches using NumPy sliding-window operations.

Because manual pixel annotation is resource-intensive, we bootstrap ground truth training masks using an automated NDVI threshold:

$$\text{Label}(x, y) = \begin{cases} 1 \quad (\text{Forest}), & \text{if } \text{NDVI}(x, y) > 0.6 \\ 0 \quad (\text{Non-Forest / Bare Ground}), & \text{otherwise} \end{cases}$$

---

### Phase 4: 4-Channel U-Net Segmentation

Standard computer vision backbones only support 3-channel (RGB) images. We use `segmentation-models-pytorch` to configure a U-Net architecture with an ImageNet-pretrained ResNet-34 encoder, setting `in_channels=4` to ingest Blue, Green, Red, and Near-Infrared simultaneously:

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=4,        # B2, B3, B4, B8
    classes=1,            # Binary forest mask
    activation=None       # Output raw logits for BCEWithLogitsLoss
)
```

The model is trained using a composite objective function that combines pixel-level classification with structural overlap:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCEWithLogits}} + \mathcal{L}_{\text{Dice}}$$

---

### Phase 5: Change Detection & Vectorization

1. **Full-Scene Sliding-Window Inference:** The trained U-Net generates full binary canopy prediction masks: $M_{2022}$ and $M_{2024}$.
2. **Post-Classification Comparison (PCC):** Deforestation is identified using Boolean logic:

$$\text{Deforestation Mask} = (M_{2022} == 1) \land (M_{2024} == 0)$$

3. **Morphological Noise Removal:** Minor coregistration offsets can produce single-pixel noise. A $3 \times 3$ `scipy.ndimage.binary_opening` filter strips isolated pixel specks while preserving contiguous clearing boundaries.
4. **Polygonization:** `rasterio.features.shapes` converts connected raster change pixels into geospatial polygons, calculating the deforested area in hectares:

$$\text{Area (ha)} = \frac{\text{Pixel Count} \times 100\text{ m}^2}{10,000\text{ m}^2/\text{ha}}$$

Results are exported as `deforestation_alerts.geojson`.

---

### Phase 6: Interactive Streamlit Dashboard

To provide an accessible interface for viewing results, a Streamlit dashboard integrates Folium maps to:
- Render dual-layer Sentinel-2 basemap overlays (2022 baseline vs. 2024 cleared).
- Display red vectorized boundary outlines over detected deforestation patches.
- Compute KPI cards highlighting total hectares cleared and individual parcel sizes.

---

## 📂 Repository Structure

```text
├── notebooks/
│   └── deforestation_pipeline.ipynb   # Complete Colab pipeline (GEE -> U-Net -> GeoJSON)
├── display_2022.png                   # Georeferenced RGB background for 2022
├── display_2024.png                   # Georeferenced RGB background for 2024
├── deforestation_alerts.geojson       # Vectorized deforestation polygon alerts
├── app.py                             # Streamlit dashboard application
├── requirements.txt                   # Local application dependencies
├── .gitignore                         # Excludes heavy GeoTIFFs, .npy arrays, and checkpoints
└── README.md                          # Project documentation and tutorial
```

---

## 🚀 Local Setup & How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/aditiprashant07/Deforestation-Detection.git
cd Deforestation-Detection
```

### 2. Set Up a Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the running dashboard.

---

## 💡 Key Insights & Lessons Learned

- **Near-Infrared is Essential:** RGB alone struggles to separate dry pasture from cleared forest. NIR reflectance provides an unmistakable spectral signature for living canopy.
- **Median Composites Over Single Scenes:** Single scenes in the tropics are almost always contaminated by clouds or shadows. Aggregating multi-pass median composites removes these artifacts without manual clean-up.
- **Morphological Filtering Cleans Boundaries:** Direct raster subtraction produces noisy, single-pixel errors along parcel edges. A simple binary opening filter removes registration artifacts and isolates real clear-cuts.
- **Decouple Heavy Compute from UI:** Deep neural networks and multi-band GeoTIFFs belong in cloud GPU environments (Colab/GEE). Dashboards run fastest when serving lightweight vector files (GeoJSON) and optimized display PNGs.
