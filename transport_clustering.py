 # transport_clustering.py

# --- Imports ---
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# --- Configuration ---
# Make sure 'nairobi_taxi_trips.csv' is in the same folder as this script.
DATASET_FILENAME = 'nairobi_taxi_trips.csv' 

# --- Step 4: Data Collection & Initial Cleaning ---
print(f"--- Step 4: Loading Data from {DATASET_FILENAME} ---")
try:
    # Load the dataset. Assumes it's a CSV with common taxi trip columns.
    data = pd.read_csv(DATASET_FILENAME)
    print(f"Successfully loaded data from '{DATASET_FILENAME}'.")
    print("\nOriginal columns in the dataset:")
    print(data.columns.tolist()) 
    print("\nFirst 5 rows of the loaded dataset:")
    print(data.head())

    # --- Identify Latitude and Longitude Columns for Clustering ---
    # Based on the common structure of taxi trip data, 'pickup_latitude' and 'pickup_longitude'
    # are excellent choices for representing origins (where trips start from).
    # If your CSV has different names (e.g., 'start_lat', 'start_lon', 'lat', 'lon'), 
    # you MUST CHANGE THESE two variable names below to match your actual CSV columns.
    lat_column = 'pickup_lat'
    lon_column = 'pickup_lon'

    # Check if the assumed columns exist in the loaded data
    if lat_column not in data.columns or lon_column not in data.columns:
        print(f"\nError: Expected columns '{lat_column}' or '{lon_column}' not found in '{DATASET_FILENAME}'.")
        print("Please inspect your CSV file and update 'lat_column' and 'lon_column' variables in this script.")
        print("\nAvailable columns are:", data.columns.tolist())
        # Exit the script gracefully if critical columns are missing
        exit() 

    # --- Data Cleaning for Clustering ---
    # Select only the relevant latitude and longitude columns for clustering.
    # Using .copy() to ensure we're working on a distinct DataFrame.
    df_for_clustering = data[[lat_column, lon_column]].copy()

    print(f"\nData points before cleaning (NaN, 0,0 filter): {len(df_for_clustering)}")
    
    # Drop rows where Latitude or Longitude values are missing (NaN - Not a Number).
    # This ensures our clustering algorithm only works with complete data points.
    df_for_clustering.dropna(subset=[lat_column, lon_column], inplace=True)
    
    # Filter out common errors like (0,0) coordinates. These usually represent invalid data
    # or points located in the ocean (e.g., off the coast of West Africa).
    # We keep rows where EITHER latitude OR longitude is not zero.
    df_for_clustering = df_for_clustering[(df_for_clustering[lat_column] != 0) | (df_for_clustering[lon_column] != 0)]
    
    # Optional: Filter for points within a realistic geographical bounding box for Nairobi.
    # This helps remove outliers that might be far from the city.
    # These values are approximate for Nairobi, Kenya. You can adjust them if needed.
    min_lat, max_lat = -1.4, -1.1 
    min_lon, max_lon = 36.6, 37.2
    
    df_for_clustering = df_for_clustering[
        (df_for_clustering[lat_column] >= min_lat) & (df_for_clustering[lat_column] <= max_lat) &
        (df_for_clustering[lon_column] >= min_lon) & (df_for_clustering[lon_column] <= max_lon)
    ]

    print(f"Data points after cleaning and geographical filtering: {len(df_for_clustering)}")
    print("\nFirst 5 rows of cleaned data used for clustering:")
    print(df_for_clustering.head())

except FileNotFoundError:
    print(f"Error: The file '{DATASET_FILENAME}' was not found.")
    print("Please make sure the CSV file is in the same folder as this Python script.")
    exit() # Stop the script if the file isn't found
except Exception as e:
    print(f"An unexpected error occurred during data loading or initial processing: {e}")
    exit() # Stop the script for any other unexpected errors

# --- Step 5: Data Preprocessing - Scaling Features ---
print("\n--- Step 5: Scaling Features ---")
# Create a StandardScaler object.
scaler = StandardScaler()

# 'Fit' the scaler to our data and then 'transform' it.
# This makes sure our latitude and longitude values are on a similar scale,
# which helps K-Means perform better by preventing one coordinate from
# dominating the distance calculations just because its values might be larger.
scaled_features = scaler.fit_transform(df_for_clustering[[lat_column, lon_column]])

print("First 5 rows of scaled features:")
print(scaled_features[:5])

# --- Step 6: Training K-Means Clustering Model ---
print("\n--- Step 6: Training K-Means Clustering Model ---")

# --- Deciding on 'K' (Number of Clusters) ---
# 'K' is the number of groups you want K-Means to find.
# For public transport optimization, 'K' could represent the number of major
# transport hubs or distinct travel origin zones you want to identify.
# You can adjust this value to explore different numbers of clusters.
num_clusters = 8 # Starting with 8 clusters for Nairobi taxi data, feel free to experiment!

# Create a KMeans model.
# 'n_clusters' specifies the number of clusters to form.
# 'random_state' ensures that the results are reproducible (you get the same clusters
# every time you run the code with the same data and settings).
# 'n_init='auto'' automatically chooses the best method for initializing centroids,
# making the clustering more robust.
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')

# Train the KMeans model using our scaled data.
# This is the core machine learning step where the AI "learns" to find the clusters
# based on the spatial proximity of the taxi pickup locations.
kmeans.fit(scaled_features) 

# Get the cluster label for each data point and add it back to our DataFrame.
# Each row in the DataFrame now has a 'Cluster' column indicating its group (0, 1, 2, etc.).
df_for_clustering['Cluster'] = kmeans.labels_

# Get the coordinates of the center of each cluster.
# These centroid coordinates represent the "ideal" or average location within each cluster.
# In the context of transport optimization, these can be seen as potential
# optimal locations for new or improved public transport hubs.
# We use scaler.inverse_transform to convert the scaled cluster centers back to
# their original Latitude/Longitude values for easy interpretation on a real map.
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

print("\nDataFrame with assigned clusters (first 5 rows):")
print(df_for_clustering.head())

print(f"\nCluster Centers (Ideal Transport Hub Locations) for {num_clusters} clusters (Latitude, Longitude):")
print(cluster_centers)

# --- Step 7: Evaluate & Visualize Results ---
print("\n--- Step 7: Visualizing Clustering Results ---")

# Set the size of our map for better visibility, especially with potentially many points.
plt.figure(figsize=(12, 10))

# Plot all our cleaned data points (taxi pickup locations).
# 'x' is Longitude, 'y' is Latitude.
# 'hue='Cluster'' tells seaborn to color each point based on its assigned cluster.
# 'palette='viridis'' is a visually distinct color scheme.
# 's=5' makes the points relatively small, suitable for dense datasets like taxi trips.
# 'alpha=0.7' makes them slightly transparent so you can see areas of higher density.
sns.scatterplot(x=lon_column, y=lat_column, hue='Cluster', data=df_for_clustering, palette='viridis', s=5, alpha=0.7)

# Plot the cluster centers on top of the data points.
# 'marker='X'' makes them prominent red 'X's, clearly visible against the data points.
# 's=300' makes them large.
# 'color='red'' and 'edgecolor='black', 'linewidth=1.5'' make them stand out.
# 'label='Cluster Centers'' adds them to the plot legend.
# Note: cluster_centers[:, 1] is Longitude (X-axis), cluster_centers[:, 0] is Latitude (Y-axis)
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], marker='X', s=300, color='red', label='Cluster Centers', edgecolor='black', linewidth=1.5)

# Add a title and labels to our map for clarity.
plt.title(f'Public Transport Origin Clusters in Nairobi from {DATASET_FILENAME} (SDG 11)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Display the legend.

plt.legend(title='Cluster')

# Add a grid to the plot for easier coordinate reading.
plt.grid(True)

# Display the map! In VS Code, this will typically open the plot in a new window.
# In Jupyter/Colab, it will display directly below the cell.
plt.show()

print("\nClustering and visualization complete. The map shows potential public transport pickup hotspots.")
