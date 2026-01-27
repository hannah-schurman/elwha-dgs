import pandas as pd
import folium
import branca.colormap as cm
import os

from tkinter import filedialog
from tkinter import *

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "./", title = "Select percentile stats", filetypes=[("Excel files","*.xlsx")])
stats_path = root.filename
print(stats_path)
root.withdraw()

df = pd.read_excel(stats_path)

# latitude, longitude, and ratio are numeric
year = df["Datetime"].dt.year[0]
print(year)
df["Latitude"] = pd.to_numeric(df["Latitude"])
df["Longitude"] = pd.to_numeric(df["Longitude"])
df["sand_to_coarse_ratio"] = pd.to_numeric(df["sand_to_coarse_ratio"])

# Create colormap
min_val = df["sand_to_coarse_ratio"].min()
max_val = df["sand_to_coarse_ratio"].max()
colormap = cm.linear.Reds_05.scale(min_val, max_val)
colormap.caption = "Sand-Coarse Ratio"

# Initialize map
m = folium.Map(
    location=[df["Latitude"].mean(), df["Longitude"].mean()],
    zoom_start=16,
)

# Add points
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=10, 
        color=colormap(row["sand_to_coarse_ratio"]),
        fill=True,
        fill_opacity=0.7,
        popup=f"Dataset: Topobathy<br>Year: {year}<br>Filename: {row['Filename']}<br>Sand-to-Coarse Ratio: {row['sand_to_coarse_ratio']:.2f}"
    ).add_to(m)

# Add colormap legend
colormap.add_to(m)

# Save map
outdir = os.path.join(os.path.dirname(stats_path), "charts")
os.makedirs(outdir, exist_ok=True)

m.save(os.path.join(outdir, "sand_to_coarse_map.html"))
print("Map saved as sand_to_coarse_map.html")
