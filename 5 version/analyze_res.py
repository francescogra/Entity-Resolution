import pandas as pd

# Load the results
results_df = pd.read_csv('entity_resolution_result.csv')
print("First few rows of the results:")
print(results_df.head())
print(f"\nBasic statistics:")
print(f"Total number of records: {len(results_df)}")
print(f"Number of clusters: {len(results_df['cluster_id'].unique())}")
print(f"Records without cluster (unique): {(results_df['cluster_id'] == -1).sum()}")

# Count records per cluster
cluster_counts = results_df[results_df['cluster_id'] != -1].groupby('cluster_id').size()
print("\nCluster sizes:")
print(cluster_counts)

# Select the top 5 clusters with the most records
large_clusters = cluster_counts.nlargest(5).index
sample_clusters = results_df[results_df['cluster_id'].isin(large_clusters)]
print("\nSample of large clusters:")
print(sample_clusters.to_string())  # Print formatted for readability

# Export clusters to CSV files
for cluster_id in large_clusters:
    cluster_data = results_df[results_df['cluster_id'] == cluster_id]
    cluster_data.to_csv(f'cluster_{cluster_id}.csv', index=False)
    print(f"Exported cluster {cluster_id} to 'cluster_{cluster_id}.csv' with {len(cluster_data)} records")

# Load the original dataset
original_df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet")
print(f"\nOriginal dataset dimensions: {len(original_df)}")

# Add the 'id' column to the original dataset based on the index
original_df['id'] = original_df.index.astype(str)

# Convert the 'id' column in results_df to string to standardize types
results_df['id'] = results_df['id'].astype(str)

# Merge the clusters into the original dataset
merged_df = original_df.merge(results_df[['id', 'cluster_id']], on='id', how='left')
print(f"\nMerged dataset dimensions: {len(merged_df)}")
print("First few rows of the merged dataset:")
print(merged_df.head())

# Check how many matches were found
matched_records = merged_df['cluster_id'].notna().sum()
total_records_in_results = len(results_df)
print(f"\nNumber of original records matched with the results: {matched_records} out of {total_records_in_results}")
print(f"Match percentage: {(matched_records / total_records_in_results * 100):.2f}%")

# Optional: Export the merged dataset for manual analysis
merged_df.to_csv('merged_results_with_original.csv', index=False)
print("Merged dataset exported to 'merged_results_with_original.csv'")