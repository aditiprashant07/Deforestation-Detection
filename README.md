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
* Ingest multi-temporal satellite data two years apart ($T_1 = 2022$ and $T_2 = 2024$).
* Strip clouds and atmospheric noise.
* Train a deep convolutional network to classify forest cover at 10-meter resolution.
* Isolate newly cleared forest parcels and compute cleared land area in hectares.
* Present the findings in an interactive map dashboard[cite: 1, 2].

---

## 🔬 Core Geospatial & Remote Sensing Concepts

### 1. How Satellites "See" (Beyond RGB)
Standard smartphone cameras only capture visible light across three bands: **Red, Green, and Blue**. Satellite sensors like Sentinel-2 carry multispectral radiometers that split light into multiple discrete wavelengths:

| Band Name | Sentinel-2 Band | Center Wavelength | Spatial Resolution | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Blue** | B2 | 490 nm | 10 meters | Atmospheric scattering, water bodies[cite: 1, 2] |
| **Green** | B3 | 560 nm | 10 meters | Vegetation discrimination[cite: 1, 2] |
| **Red** | B4 | 665 nm | 10 meters | Chlorophyll absorption[cite: 1, 2] |
| **NIR (Near-Infrared)** | B8 | 842 nm | 10 meters | **Cellular plant reflectance**[cite: 1, 2] |

> **Why NIR is Essential:** Healthy green leaves absorb blue and red light for photosynthesis, but the internal spongy mesophyll cells of leaf tissue strongly reflect **Near-Infrared (NIR)** radiation. To the human eye, healthy forest and green artificial turf look identical; in NIR, the forest glows intensely while soil, concrete, and dead grass absorb or scatter it weakly.

### 2. GeoTIFF vs. Standard Image Formats
Satellite images are not saved as standard `.jpg` or `.png` files. They are saved as **GeoTIFFs (`.tif`)**. A GeoTIFF embeds spatial metadata directly into the raster file:
* **CRS (Coordinate Reference System):** Standard geographic coordinate systems (e.g., `EPSG:4326` for WGS84 latitude/longitude).
* **Affine Transform:** Maps pixel row/column numbers $(y, x)$ to real-world latitude/longitude coordinates on Earth.
* **Multi-band Stacking:** Stores raw sensor measurement values across 4 or more channels per pixel rather than simple 8-bit color channels[cite: 1, 2].

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
