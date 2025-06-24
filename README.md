Optimizing Public Transport Routes Using K-Means Clustering
🚀 Live Demo: https://plp-clustering-web-app.onrender.com/

🚀 Project Overview
This project applies K-Means clustering to analyze Nairobi taxi trip data, identifying optimal transport pickup zones. The goal is to enhance public transport accessibility and support sustainable urban mobility (SDG 11).

📂 Dataset
Source: nairobi_taxi_trips.csv (Real-world taxi trip records)
Key Features:
pickup_lat – Latitude of pickup location
pickup_lon – Longitude of pickup location
🔍 Workflow & Methodology
1️⃣ Data Collection & Initial Cleaning
✅ Load dataset and extract pickup latitude & longitude.
✅ Remove missing values, filter out (0,0) erroneous entries, and keep data within Nairobi’s city boundaries.

Cleaned Dataset Preview:
Cleaned Dataset

2️⃣ Data Preprocessing & Feature Scaling
✅ Apply StandardScaler to normalize latitude & longitude for clustering.

Scaled Dataset Statistics:
Scaled Data

3️⃣ K-Means Clustering Implementation
✅ Set num_clusters = 6 to segment transport pickup zones.
✅ Train K-Means model on scaled data.
✅ Extract cluster centers, representing ideal transport hubs.

Scatter Plot with Clusters:
Cluster Visualization

4️⃣ Evaluation Metrics
✅ Boxplots: Detect anomalies in latitude & longitude distribution.

Boxplot of Latitude:
Latitude Boxplot

Boxplot of Longitude:
Longitude Boxplot

✅ Heatmap Analysis: Identify high-density taxi pickup areas.

Density Heatmap of Pickup Locations:
Density Heatmap

✅ Pairplot: Visualize feature relationships in location data.

Pairplot of Latitude and Longitude:
Pairplot

✅ Cluster Distribution Plot: Ensures transport demand is balanced across clusters.

Cluster Size Distribution:
Cluster Size Distribution

🔍 Ethical Reflection
How Might Bias in the Data Affect Outcomes?
✅ Geographical Bias: If the dataset overrepresents high-traffic urban areas, clustering may neglect underserved regions where transport accessibility is lower.
✅ Time-Based Bias: If the data collection mostly occurred during rush hours, it may misrepresent off-peak transport demand, leading to ineffective service distribution.
✅ User Demographic Bias: Since ride-hailing services are used predominantly by middle- and upper-income groups, the clustering model may not reflect the needs of lower-income commuters relying on informal transport.

How Does the Solution Promote Fairness and Sustainability?
✅ Equitable Transport Planning: By identifying high-demand pickup zones, urban planners can optimize routes to ensure fair transport access for all areas.
✅ Reduced Congestion & Emissions: Well-clustered transport hubs improve route efficiency, lowering traffic congestion and carbon footprint.
✅ Improved Public Mobility: The results can be used to enhance matatu and bus networks, benefiting commuters across income levels.

About
No description, website, or topics provided.
Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Report repository
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Contributors
4
@Brillywam
Brillywam Brilliant Mwendwa
@teddy502
teddy502 Teddy Brian
@254Manuell
254Manuell Emmanuel Ngunnzi
@Lowellowuor
Lowellowuor lowell owuor
Languages
Jupyter Notebook
100.0%
Footer
© 2025 G
