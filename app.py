import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import json

st.set_page_config(page_title="Forest Change Monitor", layout="wide")

st.title("🛰️ Sentinel-2 Deforestation Monitoring Dashboard")
st.markdown("Automated canopy loss detection via 4-band U-Net segmentation and post-classification comparison.")

# Load GeoJSON alerts
geojson_path = "deforestation_alerts.geojson"
gdf = gpd.read_file(geojson_path)

# Top metric summary cards
total_cleared_ha = gdf["area_ha"].sum()
total_parcels = len(gdf)

col1, col2, col3 = st.columns(3)
col1.metric("Total Forest Loss", f"{total_cleared_ha:.2f} ha")
col2.metric("Detected Parcels", f"{total_parcels}")
col3.metric("Analysis Period", "2022 – 2024")

# Set up interactive Folium map centered on Rondônia
center_lat, center_lon = -10.50, -62.80
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

# Image geographic bounds: [[south, west], [north, east]]
image_bounds = [[-10.55, -62.85], [-10.45, -62.75]]

# Add Sentinel-2 overlays
folium.raster_layers.ImageOverlay(
    name="Sentinel-2 2022",
    image="display_2022.png",
    bounds=image_bounds,
    opacity=1.0,
    show=True
).add_to(m)

folium.raster_layers.ImageOverlay(
    name="Sentinel-2 2024",
    image="display_2024.png",
    bounds=image_bounds,
    opacity=1.0,
    show=False
).add_to(m)

# Add Deforestation Alert Polygons
def style_fn(feature):
    return {
        "fillColor": "red",
        "color": "darkred",
        "weight": 1.5,
        "fillOpacity": 0.5,
    }

folium.GeoJson(
    gdf,
    name="Deforestation Alerts",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(fields=["area_ha", "detection_date"], aliases=["Area (ha):", "Date:"])
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Render map in Streamlit
st_folium(m, width="100%", height=650)

# Display tabular data below
st.subheader("Detected Deforestation Clusters")
st.dataframe(gdf[["area_ha", "pixel_count", "detection_date"]].sort_values(by="area_ha", ascending=False))