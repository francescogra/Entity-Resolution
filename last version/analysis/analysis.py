import pandas as pd

df = pd.read_parquet('entity_resolution_result.parquet')
cluster_sizes = df[df['cluster_id'] >= 0].groupby('cluster_id').size()
print(cluster_sizes.describe())
print("Maximum cluster size:", cluster_sizes.max())


# Find the 5 largest clusters
largest_clusters = cluster_sizes.nlargest(5)
for cluster_id, size in largest_clusters.items():
    print(f"\nCluster {cluster_id} - {size} records:")
    cluster_records = df[df['cluster_id'] == cluster_id]
    # Show company names and URLs
    print(cluster_records[['company_name', 'website_url']].head())

# Example: verify URL consistency within clusters
def check_url_consistency(cluster_df):
    urls = cluster_df['normalized_url'].dropna().unique()
    return len(urls) == 1 if len(urls) > 0 else True
cluster_df = df[df['cluster_id'] >= 0]
url_consistency = cluster_df.groupby('cluster_id').apply(check_url_consistency)
print(f"Clusters with consistent URLs: {url_consistency.mean()*100:.2f}%")

# Load the file
df = pd.read_parquet('entity_resolution_result.parquet')  # or .csv

# View a specific cluster
def view_cluster(cluster_id):
    cluster_data = df[df['cluster_id'] == cluster_id]
    return cluster_data[['id', 'company_name', 'website_url', 'main_address_raw_text', 'primary_phone']]

# Example: view the largest cluster
print(view_cluster(2314))

# View all records without clusters (unique entities)
single_entities = df[df['cluster_id'] == -1]
print(f"Number of unique entities: {len(single_entities)}")
df = pd.read_parquet('entity_resolution_result.parquet')


# Create a summary report for clusters
def create_cluster_summary(df, min_size=2, max_clusters=20):
    # Exclude records without clusters
    clustered_df = df[df['cluster_id'] >= 0]
   
    # Count the size of each cluster
    cluster_sizes = clustered_df.groupby('cluster_id').size().sort_values(ascending=False)
   
    # Select clusters with sizes >= min_size
    large_clusters = cluster_sizes[cluster_sizes >= min_size]
   
    print(f"Top {min(max_clusters, len(large_clusters))} clusters with at least {min_size} records:\n")
   
    for idx, (cluster_id, size) in enumerate(large_clusters.head(max_clusters).items()):
        print(f"Cluster {cluster_id} ({size} records):")
       
        # Get cluster data
        cluster_data = df[df['cluster_id'] == cluster_id]
       
        # Show company name and URL
        names = cluster_data['company_name'].value_counts().head(3)
        urls = cluster_data['normalized_url'].value_counts().head(3)
       
        print("  Most common names:")
        for name, count in names.items():
            print(f"    - {name} ({count})")
       
        print("  Most common URLs:")
        for url, count in urls.items():
            if pd.notna(url):
                print(f"    - {url} ({count})")
            else:
                print(f"    - [No URL] ({count})")
       
        print()
        
# Run the report
create_cluster_summary(df, min_size=10, max_clusters=10)
def export_cluster(df, cluster_id, output_file=None):
    cluster_data = df[df['cluster_id'] == cluster_id]
   
    if output_file:
        cluster_data.to_csv(output_file, index=False)
        print(f"Cluster {cluster_id} exported to {output_file}")
   
    return cluster_data
# Example: export the largest cluster
export_cluster(df, 2314, "cluster_2314.csv")