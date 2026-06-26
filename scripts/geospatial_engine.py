import os
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster

def generate_synthetic_geo_data(file_path, num_points=500):
    """Generates synthetic geospatial AQI data for the engine."""
    np.random.seed(42)
    
    # Base coordinates (Using a central urban approximation, e.g., Jaipur coords)
    base_lat = 26.9124
    base_lon = 75.7873
    
    # Generate random points within a ~20km radius
    lats = base_lat + np.random.uniform(-0.15, 0.15, num_points)
    lons = base_lon + np.random.uniform(-0.15, 0.15, num_points)
    
    # Generate AQI values with a heavy tail for hotspots
    aqi_values = np.random.gamma(shape=2.5, scale=60, size=num_points).astype(int)
    aqi_values = np.clip(aqi_values, 10, 500) 
    
    sources = ['Traffic', 'Construction', 'Industry', 'Mixed']
    pollution_sources = np.random.choice(sources, num_points, p=[0.4, 0.2, 0.3, 0.1])
    
    df = pd.DataFrame({
        'Latitude': lats,
        'Longitude': lons,
        'AQI': aqi_values,
        'Pollution_Source': pollution_sources
    })
    
    df.to_csv(file_path, index=False)
    print(f"[*] Generated synthetic geospatial data with {num_points} points at {file_path}")

def get_aqi_color(aqi):
    """Returns color codes based on standard AQI health thresholds."""
    if aqi <= 50:
        return 'green'
    elif aqi <= 100:
        return 'lightgray' # Yellow is hard to see on maps, folium uses standard bootstrap colors. We will use hex for custom circles.
    elif aqi <= 200:
        return 'orange'
    elif aqi <= 300:
        return 'red'
    else:
        return 'purple'

def get_hex_color(aqi):
    """Returns exact hex colors for accurate UI rendering."""
    if aqi <= 50: return '#00E400'       # Green
    elif aqi <= 100: return '#FFFF00'    # Yellow
    elif aqi <= 200: return '#FF7E00'    # Orange
    elif aqi <= 300: return '#FF0000'    # Red
    else: return '#8F3F97'               # Purple

def main():
    # 1. Directory Setup
    data_dir = 'data'
    dashboard_dir = 'dashboard'
    data_path = os.path.join(data_dir, 'geo_data.csv')
    map_path = os.path.join(dashboard_dir, 'aqi_heatmap.html')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # 2. Data Ingestion
    if not os.path.exists(data_path):
        generate_synthetic_geo_data(data_path)
        
    print("[*] Loading geospatial data...")
    df = pd.read_csv(data_path)
    
    # 3. Map Initialization
    print("[*] Initializing base map...")
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    # Create base map with a dark theme to make heatmap pop
    base_map = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=11, 
        tiles='CartoDB dark_matter'
    )
    
    # 4. Generate Heatmap Layer
    print("[*] Generating thermal density heatmap...")
    heat_data = [[row['Latitude'], row['Longitude'], row['AQI']] for index, row in df.iterrows()]
    HeatMap(
        heat_data, 
        radius=15, 
        max_zoom=13, 
        blur=10, 
        gradient={0.2: 'lime', 0.4: 'yellow', 0.6: 'orange', 0.8: 'red', 1.0: 'purple'}
    ).add_to(base_map)
    
    # 5. Generate Hotspot Clusters with Interactive Popups
    print("[*] Generating hotspot clusters...")
    marker_cluster = MarkerCluster(name="Pollution Hotspots").add_to(base_map)
    
    for index, row in df.iterrows():
        aqi = row['AQI']
        color_hex = get_hex_color(aqi)
        
        # HTML template for the popup
        popup_html = f"""
        <div style="font-family: Arial; min-width: 150px;">
            <h4 style="margin-bottom: 5px; color: {color_hex};">AQI: {aqi}</h4>
            <b>Source:</b> {row['Pollution_Source']}<br>
            <b>Lat:</b> {row['Latitude']:.4f}<br>
            <b>Lon:</b> {row['Longitude']:.4f}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            popup=folium.Popup(popup_html, max_width=250),
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.7,
            weight=1
        ).add_to(marker_cluster)
        
    # 6. Add Layer Control and Save
    folium.LayerControl().add_to(base_map)
    
    print(f"[*] Exporting interactive map to {map_path}...")
    base_map.save(map_path)
    print("[*] Geospatial engine execution complete.")

if __name__ == "__main__":
    main()