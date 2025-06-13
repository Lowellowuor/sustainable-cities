import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
import sys

# Helper function to convert matplotlib plot to base64 image
def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded PNG image."""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory
    return img_b64

def generate_cluster_map(df, lat_col_name, lon_col_name, n_clusters, cluster_centers_df):
    """Generates and returns a base64 encoded cluster map."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x=lon_col_name,
        y=lat_col_name,
        hue='cluster_label',
        palette='viridis',
        s=10,
        alpha=0.6,
        legend='full',
        ax=ax
    )
    sns.scatterplot(
        data=cluster_centers_df,
        x='center_lon',
        y='center_lat',
        marker='X',
        s=300,
        color='red',
        edgecolor='black',
        label='Cluster Centers',
        zorder=5,
        ax=ax
    )
    ax.set_title(f'Mobility Hotspots (K-Means Clusters: {n_clusters}) based on {lat_col_name}, {lon_col_name}')
    ax.set_xlabel(f'{lon_col_name} (Longitude)')
    ax.set_ylabel(f'{lat_col_name} (Latitude)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Cluster')
    return fig_to_base64(fig)

def generate_boxplot(df, column_name, title, color):
    """Generates and returns a base64 encoded boxplot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df[column_name], color=color, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(column_name)
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig_to_base64(fig)

def generate_kde_plot(df, lat_col_name, lon_col_name, title):
    """Generates and returns a base64 encoded KDE (Density) plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.kdeplot(
        x=df[lon_col_name],
        y=df[lat_col_name],
        fill=True,
        cmap='coolwarm',
        cbar=True,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(f'{lon_col_name} (Longitude)')
    ax.set_ylabel(f'{lat_col_name} (Latitude)')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig_to_base64(fig)

def generate_pairplot(df, lat_col_name, lon_col_name):
    """Generates and returns a base64 encoded pairplot for lat/lon."""
    g = sns.pairplot(df[[lat_col_name, lon_col_name]])
    g.fig.suptitle('Pairplot of Specified Location Coordinates', y=1.02, fontsize=16)
    img_b64 = fig_to_base64(g.fig)
    return img_b64


def preprocess_and_cluster(df_path, n_clusters=5, lat_col_name='pickup_lat', lon_col_name='pickup_lon'):
    """
    Loads, preprocesses, and performs K-Means clustering on the data specified by user.
    Generates and returns all required plots as base64 images,
    along with raw textual output and performance metrics.

    Args:
        df_path (str): Path to the input CSV file.
        n_clusters (int): Number of clusters for K-Means.
        lat_col_name (str): Name of the latitude column in the CSV.
        lon_col_name (str): Name of the longitude column in the CSV.

    Returns:
        tuple: (cluster_centers_df, plot_images_dict, raw_text_output, metrics_dict, error_message)
               - cluster_centers_df (pd.DataFrame): DataFrame of cluster centroids.
               - plot_images_dict (dict): Dictionary with base64 encoded image strings for each plot.
               - raw_text_output (str): Captured console output from the clustering process.
               - metrics_dict (dict): Dictionary of K-Means performance metrics (inertia, silhouette_score).
               - error_message (str or None): Error message if any, otherwise None.
    """
    plot_images = {}
    cluster_centers_df = pd.DataFrame()
    metrics = {}
    error_message = None
    df_processed = pd.DataFrame() # To ensure df_processed is defined even if errors occur early

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()

    try:
        print(f"--- Step 1: Loading Data from {df_path} ---")
        df = pd.read_csv(df_path)
        print("Successfully loaded data.")
        # Store initial_rows here, right after df is loaded
        initial_rows = len(df) # FIX: Define initial_rows here
        print(f"Original columns in the dataset: {df.columns.tolist()}")
        print("First 5 rows of the loaded dataset:")
        print(df.head().to_string())

        print(f"\n--- Step 2: Validating and Cleaning '{lat_col_name}' and '{lon_col_name}' Columns ---")

        # Check if specified columns exist
        if lat_col_name not in df.columns or lon_col_name not in df.columns:
            raise ValueError(f"Specified columns '{lat_col_name}' or '{lon_col_name}' not found in the CSV. "
                             f"Please check your input. Available columns: {df.columns.tolist()}")

        # Convert specified columns to numeric, coercing errors (non-numeric values) to NaN
        df[lat_col_name] = pd.to_numeric(df[lat_col_name], errors='coerce')
        df[lon_col_name] = pd.to_numeric(df[lon_col_name], errors='coerce')
        print(f"Converted '{lat_col_name}' and '{lon_col_name}' to numeric, non-numeric values replaced with NaN.")

        # Drop rows with NaN values in the relevant columns after conversion
        initial_rows_after_col_check = len(df)
        df_processed = df.dropna(subset=[lat_col_name, lon_col_name]).copy()
        print(f"Data points after dropping NaN in '{lat_col_name}' and '{lon_col_name}': {len(df_processed)}")
        print(f"Dropped {initial_rows_after_col_check - len(df_processed)} rows due to missing/non-numeric values.")


        # Filter out invalid (0,0) coordinates (if present)
        initial_rows_after_dropna = len(df_processed)
        df_processed = df_processed[
            (df_processed[lat_col_name] != 0) | (df_processed[lon_col_name] != 0)
        ].copy()
        print(f"Data points after filtering (0,0) coordinates: {len(df_processed)}")
        print(f"Dropped {initial_rows_after_dropna - len(df_processed)} rows due to (0,0) coordinates.")

        # Approximate bounding box for Nairobi (adjust or remove if you want truly global data)
        min_lat, max_lat = -1.4, -1.1
        min_lon, max_lon = 36.6, 37.2

        initial_rows_after_zero_filter = len(df_processed)
        df_processed = df_processed[
            (df_processed[lat_col_name] >= min_lat) & (df_processed[lat_col_name] <= max_lat) &
            (df_processed[lon_col_name] >= min_lon) & (df_processed[lon_col_name] <= max_lon)
        ].copy()
        print(f"Data points after geographical filtering (Nairobi bounds: Lat {min_lat}-{max_lat}, Lon {min_lon}-{max_lon}): {len(df_processed)}")
        print(f"Dropped {initial_rows_after_zero_filter - len(df_processed)} rows due to being outside Nairobi bounds.")
        print(f"Total valid data points for clustering: {len(df_processed)} out of {initial_rows} original rows.")


        if df_processed.empty:
            raise ValueError(f"No valid data points found after cleaning and filtering. "
                             f"Please ensure your CSV has numeric data in '{lat_col_name}' and '{lon_col_name}', "
                             f"and that the coordinates are within the expected geographical range for Nairobi.")

        print(f"\nFirst 5 rows of cleaned data used for clustering ({lat_col_name}, {lon_col_name}):")
        print(df_processed[[lat_col_name, lon_col_name]].head().to_string())

        # Select features for clustering
        X = df_processed[[lat_col_name, lon_col_name]]

        print("\n--- Step 3: Scaling Features ---")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("First 5 rows of scaled features:")
        print(X_scaled[:5].tolist())

        print(f"\n--- Step 4: Training K-Means Clustering Model (K={n_clusters}) ---")
        # Ensure n_clusters is not greater than the number of samples
        if n_clusters > len(X_scaled):
            n_clusters = max(1, len(X_scaled)) # Ensure at least 1 cluster if data exists
            print(f"Warning: Number of clusters (K={n_clusters}) was adjusted to match the number of available data points.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df_clustered = df_processed.copy() # Use the processed DataFrame
        df_clustered['cluster_label'] = kmeans.fit_predict(X_scaled)
        print("DataFrame with assigned clusters (first 5 rows):")
        print(df_clustered.head().to_string())

        # Get cluster centers (in original scale for plotting)
        cluster_centers_scaled = kmeans.cluster_centers_
        cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=['center_lat', 'center_lon'])
        cluster_centers_df['cluster_id'] = range(n_clusters)
        print("\nCluster Centers (Ideal Transport Hub Locations):")
        print(cluster_centers_df.to_string())

        # --- Performance Metrics ---
        print(f"\n--- Step 5: K-Means Performance Metrics (K={n_clusters}) ---")
        metrics['Inertia'] = kmeans.inertia_
        print(f"Inertia (Within-cluster Sum of Squares): {metrics['Inertia']:.2f}")

        if n_clusters > 1 and len(X_scaled) >= n_clusters: # Silhouette score needs at least 2 clusters and samples >= clusters
            try:
                metrics['Silhouette_Score'] = silhouette_score(X_scaled, df_clustered['cluster_label'])
                print(f"Silhouette Score: {metrics['Silhouette_Score']:.2f}")
            except Exception as e:
                print(f"Could not calculate Silhouette Score: {e}")
                metrics['Silhouette_Score'] = "N/A"
        else:
            print("Silhouette Score requires more than one cluster (K > 1) and at least K samples.")
            metrics['Silhouette_Score'] = "N/A (K=1 or insufficient samples)"

        # --- Generate all plots ---
        print("\n--- Step 6: Generating Visualizations ---")
        plot_images['cluster_map'] = generate_cluster_map(df_clustered, lat_col_name, lon_col_name, n_clusters, cluster_centers_df)
        print("  - Cluster Map generated.")
        plot_images['boxplot_lat'] = generate_boxplot(df_processed, lat_col_name, f'Boxplot of {lat_col_name}', 'blue')
        print(f"  - Boxplot for {lat_col_name} generated.")
        plot_images['boxplot_lon'] = generate_boxplot(df_processed, lon_col_name, f'Boxplot of {lon_col_name}', 'green')
        print(f"  - Boxplot for {lon_col_name} generated.")
        plot_images['kde_plot'] = generate_kde_plot(df_processed, lat_col_name, lon_col_name, 'Density Map of Mobility Locations (KDE Plot)')
        print("  - KDE Plot generated.")
        plot_images['pairplot'] = generate_pairplot(df_processed, lat_col_name, lon_col_name)
        print("  - Pairplot generated.")

    except ValueError as e:
        error_message = f"Data Error: {e}"
        print(f"\nERROR: {error_message}")
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}. Please review your data and selected columns."
        print(f"\nUNEXPECTED ERROR: {error_message}")
    finally:
        sys.stdout = old_stdout
        raw_output_text = output_buffer.getvalue()

    return cluster_centers_df, plot_images, raw_output_text, metrics, error_message

if __name__ == '__main__':
    # Test with a dummy file that simulates good data
    dummy_good_data = {
        'my_latitude_column': [-1.28, -1.29, -1.27, -1.30, -1.18, -1.19, -1.20, -1.25, -1.26, -1.27,
                               -1.285, -1.295, -1.275, -1.305, -1.185, -1.195, -1.205, -1.255, -1.265, -1.275],
        'my_longitude_column': [36.82, 36.80, 36.83, 36.81, 36.88, 36.87, 36.89, 36.75, 36.76, 36.74,
                                36.825, 36.805, 36.835, 36.815, 36.885, 36.875, 36.895, 36.755, 36.765, 36.745],
        'some_other_data': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    }
    dummy_good_df = pd.DataFrame(dummy_good_data)
    dummy_good_csv_path = 'dummy_good_data.csv'
    dummy_good_df.to_csv(dummy_good_csv_path, index=False)
    print(f"Created dummy good CSV at: {dummy_good_csv_path}")

    try:
        print("\n--- Testing with GOOD data ---")
        centers, plot_imgs_dict, raw_out, metrics_dict, err = preprocess_and_cluster(
            dummy_good_csv_path,
            n_clusters=3,
            lat_col_name='my_latitude_column',
            lon_col_name='my_longitude_column'
        )
        if err:
            print(f"Error during good data test: {err}")
        else:
            print("\nGood Data Test Results Summary:")
            print(f"Number of plots generated: {len(plot_imgs_dict)}")
            print(f"Metrics: {metrics_dict}")
            # print(raw_out) # Uncomment to see full raw output in console
    except Exception as e:
        print(f"Unexpected error during good data test: {e}")
    finally:
        os.remove(dummy_good_csv_path)

    # Test with a dummy file that simulates bad/irrelevant data (like your fertility.csv)
    dummy_bad_data = {
        'Age': [25, 30, 45, 'abc', 35],
        'Childish_diseases': [0, 1, 0, 1, 'xyz'],
        'Other': ['A','B','C','D','E']
    }
    dummy_bad_df = pd.DataFrame(dummy_bad_data)
    dummy_bad_csv_path = 'dummy_bad_data.csv'
    dummy_bad_df.to_csv(dummy_bad_csv_path, index=False)
    print(f"\nCreated dummy bad CSV at: {dummy_bad_csv_path}")

    try:
        print("\n--- Testing with BAD data ---")
        centers, plot_imgs_dict, raw_out, metrics_dict, err = preprocess_and_cluster(
            dummy_bad_csv_path,
            n_clusters=3,
            lat_col_name='Age', # Intentionally using non-geographic columns
            lon_col_name='Childish_diseases'
        )
        if err:
            print(f"Error during bad data test: {err}")
            # print(raw_out) # Uncomment to see full raw output in console
        else:
            print("\nBad Data Test Results Summary: (Should not reach here if error caught)")
    except Exception as e:
        print(f"Unexpected error during bad data test: {e}")
    finally:
        os.remove(dummy_bad_csv_path)
