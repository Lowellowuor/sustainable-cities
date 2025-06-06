 # transport_clustering.py

# Step 4: Data Collection - Loading Mock Data

# First, let's make sure we have pandas imported for data handling.
import pandas as pd

# This is the mock data we decided on in Phase 1.
# Each inner list represents a location with [Latitude, Longitude].
# These are example coordinates for clusters in a city like Nairobi for context.
mock_transport_data = [
    # Cluster 1 (e.g., CBD area)
    [-1.2833, 36.8167], # Nairobi CBD approx
    [-1.2860, 36.8200],
    [-1.2850, 36.8180],
    [-1.2900, 36.8100],
    [-1.2880, 36.8150],

    # Cluster 2 (e.g., Westlands area)
    [-1.2650, 36.7900], # Westlands approx
    [-1.2700, 36.7950],
    [-1.2680, 36.7930],
    [-1.2660, 36.7880],
    [-1.2710, 36.7970],

    # Cluster 3 (e.g., Karen/Langata area)
    [-1.3200, 36.7000], # Karen approx
    [-1.3150, 36.7050],
    [-1.3250, 36.6980],
    [-1.3180, 36.7020],
    [-1.3220, 36.6950],
    
    # Cluster 4 (e.g., Embakasi area)
    [-1.2900, 36.9000], # Embakasi approx
    [-1.2950, 36.9050],
    [-1.2920, 36.9020],
    [-1.2970, 36.9080],
    [-1.2890, 36.9100]
]
# Create a pandas DataFrame from our mock data.
# We give names to our columns: 'Latitude' and 'Longitude'.
df = pd.DataFrame(mock_transport_data, columns=['Latitude', 'Longitude'])

# Print the first few rows of our DataFrame to see what it looks like.
print("First 5 rows of our transport data:")
print(df.head())

# Print how many data points we have.
print(f"\nTotal data points: {len(df)}")
     # Step 5: Data Preprocessing - Scaling Features

    # Import StandardScaler from scikit-learn.
    # This helps make sure all our numbers (latitude, longitude) are on a similar scale,
    # which can help some machine learning algorithms work better.
from sklearn.preprocessing import StandardScaler
    
    # Create a StandardScaler object.
scaler = StandardScaler()
    
    # 'Fit' the scaler to our data and then 'transform' it.
    # We select only the 'Latitude' and 'Longitude' columns for scaling.
scaled_features = scaler.fit_transform(df[['Latitude', 'Longitude']])
    
    # Print the first few scaled features to see they are now different (usually around 0).
print("\nFirst 5 rows of scaled features:")
print(scaled_features[:5])
    # Step 6: Training K-Means Clustering Model

    # Import the KMeans clustering algorithm.
from sklearn.cluster import KMeans
    
    # --- Deciding on 'K' (Number of Clusters) ---
    # 'K' is the number of groups you want K-Means to find.
    # For transport, you might think about how many main "zones" or "hubs" a city has.
    # Let's start with 4 because we made 4 imaginary groups in our mock data.
    # In a real project, you might try different K values or use the "Elbow Method"
    # to help choose the best K.
num_clusters = 4 
    
    # Create a KMeans model.
    # 'n_clusters' is our 'K'.
    # 'random_state' makes sure you get the same results every time you run it.
    # 'n_init=10' is important! It runs the K-Means algorithm 10 times with different
    # starting points and picks the best result, making it more robust.
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto') # 'auto' is default for modern sklearn
    # Note: 'n_init' should be 'auto' for newer sklearn versions, or an integer like 10 for older ones.
    # 'auto' is usually fine and recommended.
    
    # Train the KMeans model using our scaled data.
    # This is where the AI "learns" to find the clusters.
kmeans.fit(scaled_features) 
    
    # Get the cluster label for each data point.
    # This tells us which group (0, 1, 2, or 3) each location belongs to.
df['Cluster'] = kmeans.labels_
    
    # Get the coordinates of the center of each cluster.
    # These are the "ideal" locations for a transport hub in each cluster.
    # We use scaler.inverse_transform to get the original Latitude/Longitude back
    # because we scaled the data. If you skipped scaling, just use kmeans.cluster_centers_.
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    

    
print(f"\nCluster Centers (Ideal Transport Hub Locations) for {num_clusters} clusters (Latitude, Longitude):")
print(cluster_centers)
    # Step 7: Evaluate & Visualize Results

    # Import libraries for plotting.
import matplotlib.pyplot as plt
import seaborn as sns
    
    # Set the size of our map so it's easy to see.
plt.figure(figsize=(10, 8))
    
    # Plot all our original data points.
    # 'x' is Longitude, 'y' is Latitude.
    # 'hue='Cluster'' tells it to color each point based on its assigned cluster.
    # 'palette='viridis'' is a nice color scheme.
    # 's=100' makes the points a good size.
    # 'alpha=0.8' makes them slightly transparent so you can see overlapping points.
sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.8)
    
    # Now, plot the cluster centers on top of the points.
    # 'marker='X'' makes them big red 'X's so they stand out.
    # 's=200' makes them even bigger.
    # 'label='Cluster Centers'' adds them to the legend.
    # Note: cluster_centers[:, 1] is Longitude, cluster_centers[:, 0] is Latitude
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], marker='X', s=200, color='red', label='Cluster Centers')
    
    # Add a title and labels to our map.
plt.title('Public Transport Activity Clusters for SDG 11')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
    
    # Show the legend (the box that explains the colors and markers).
plt.legend()
    
    # Add a grid to make it easier to read coordinates.
plt.grid(True)
    
    # Display the map!
    # In VS Code, this will typically open the plot in a new window.
plt.show()
    