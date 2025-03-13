import pandas as pd
from dedupe import Dedupe, variables, console_label

# Read the Parquet file
df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet")

# Check available columns (optional but recommended)
print(df.columns)

# Define fields for deduplication
fields = [
    variables.String('company_name'),           # Company name
    variables.String('main_address_raw_text'),  # Main address
    variables.String('main_country'),
    variables.String('main_postcode'),
    variables.String('website_url'),
    variables.String('primary_phone')
]

# Create Dedupe object
deduper = Dedupe(fields)

# Prepare data for training
data = df.to_dict('index')
deduper.prepare_training(data)

# Interactive labeling (requires manual input)
console_label(deduper)

# Train the model
deduper.train()

# Save training data (optional)
with open('training.json', 'w') as f:
    deduper.write_training(f)

# Find duplicate clusters
clusters = deduper.partition(data, threshold=0.5)

# Print found clusters
for cluster_id, (cluster, scores) in enumerate(clusters):
    print(f"Cluster {cluster_id}: {cluster}")