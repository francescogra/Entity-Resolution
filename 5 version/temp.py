import pandas as pd
import numpy as np
from dedupe import Dedupe, variables
import os
import json
import re
from tqdm import tqdm
import gc
from metaphone import doublemetaphone
import phonenumbers
import logging

# Configure logging
logging.basicConfig(
    filename='entity_resolution.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def normalize_url(url):
    """Normalize URLs by removing prefixes and standardizing the format."""
    if pd.isna(url) or not isinstance(url, str):
        return None
    url = re.sub(r'^https?://|^www\.', '', url.lower()).rstrip('/')
    return url if url else None

def clean_company_name(name):
    """Clean the company name by removing legal suffixes and irrelevant characters."""
    if pd.isna(name) or not isinstance(name, str):
        return None
    name = name.lower()
    legal_entities = [' inc', ' inc.', ' llc', ' llc.', ' ltd', ' ltd.', ' corp', ' corp.']
    for entity in legal_entities:
        name = name.replace(entity, '')
    name = re.sub(r'[^\w\s]', ' ', name)
    cleaned = re.sub(r'\s+', ' ', name).strip()
    return cleaned if cleaned else None

def normalize_phone(phone):
    """Normalize phone numbers to E164 format."""
    if not phone:
        return None
    try:
        parsed = phonenumbers.parse(phone, None)
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        return None

def create_blocking_key(row):
    """Create a more specific blocking key to reduce the number of comparisons."""
    name = row['clean_company_name'] or ""
    url = row['normalized_url'] or ""
    country = row['main_country'] if pd.notna(row['main_country']) else "ZZ"

    name_key = doublemetaphone(name)[0][:10] if name else "##########"
    url_part = re.sub(r'^www\.|_|-|\.', '', url.lower()).split('.')[0][:8] if url else "########"
    url_key = url_part if len(url_part) >= 3 else "########"

    return f"{name_key}_{url_key}_{country}"

def is_valid_record(record):
    """Check if a record has at least one non-empty field."""
    return bool(
        (record['clean_company_name'] is not None and record['clean_company_name'] != '') or 
        (record['normalized_url'] is not None and record['normalized_url'] != '') or
        (record['main_address_raw_text'] is not None and record['main_address_raw_text'] != '') or
        (record['normalized_phone'] is not None and record['normalized_phone'] != '')
    )

def process_block_batch(block_dict, deduper, batch_size=1000):
    """Process a block in batches to manage memory."""
    keys = list(block_dict.keys())
    all_clusters = []
    
    for i in range(0, len(keys), batch_size):
        batch = {k: block_dict[k] for k in keys[i:i + batch_size]}
        if len(batch) > 1:
            try:
                clustered_dupes = deduper.partition(batch, threshold=0.7)
                all_clusters.extend(clustered_dupes)
            except MemoryError:
                logging.error(f"MemoryError in batch {i} of block ({len(batch)} record)")
                print(f"⚠️ MemoryError in batch {i} of block ({len(batch)} record)")
            except Exception as e:
                logging.error(f"Error in batch {i}: {str(e)}")
                print(f"⚠️ Error in batch {i}: {str(e)}")
        gc.collect()
    
    return all_clusters

def main():
    logging.info("Initializing Entity Resolution...")
    print("Initializing Entity Resolution...")
    
    # Read the full dataset
    df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet")
    df['id'] = df.index.astype(str)
    print(f"Dataset dimensions: {df.shape}")
    logging.info(f"Dataset dimensions: {df.shape}")
    
    # Preprocessing
    print("Preprocessing...")
    logging.info("Preprocessing...")
    df['clean_company_name'] = df['company_name'].apply(clean_company_name)
    df['normalized_url'] = df['website_url'].apply(normalize_url)
    df['normalized_phone'] = df['primary_phone'].apply(normalize_phone)
    
    df['blocking_key'] = df.apply(create_blocking_key, axis=1)

    # Filter records with all empty fields
    df = df[
        df.apply(lambda row: is_valid_record({
            'clean_company_name': row['clean_company_name'],
            'normalized_url': row['normalized_url'],
            'main_address_raw_text': row['main_address_raw_text'],
            'normalized_phone': row['normalized_phone']
        }), axis=1)
    ].copy()
    
    print(f"Dataset dimensions after filtering: {df.shape}")
    logging.info(f"Dataset dimensions after filtering: {df.shape}")
    
    # Create blocks
    blocks = df.groupby('blocking_key')
    print(f"Blocks created: {len(blocks)}")
    logging.info(f"Blocks created: {len(blocks)}")
    
    # Analyze block sizes
    block_sizes = blocks.size()
    print(f"Average block size: {block_sizes.mean():.2f}")
    print(f"Maximum block size: {block_sizes.max()}")
    logging.info(f"Average block size: {block_sizes.mean():.2f}")
    logging.info(f"Maximum block size: {block_sizes.max()}")
    
    # Create a dictionary with a data sample
    data_sample = {}
    sample_size = min(1000, len(df))  # Take a reasonable sample
    sample_indices = np.random.choice(df.index, sample_size, replace=False)

    for idx in sample_indices:
        row = df.loc[idx]
        data_sample[row['id']] = {
            'clean_company_name': row['clean_company_name'],
            'normalized_url': row['normalized_url'],
            'main_address_raw_text': row['main_address_raw_text'],
            'normalized_phone': row['normalized_phone']
        }

    # Fields for dedupe
    fields = [
        variables.String('clean_company_name', has_missing=True),
        variables.String('normalized_url', has_missing=True),
        variables.String('main_address_raw_text', has_missing=True),
        variables.String('normalized_phone', has_missing=True),
    ]
    
    # Initialize deduper
    deduper = Dedupe(fields)

    # Load training data from file
    training_file = 'training_data_test.json'
    print(f"Loading training data from '{training_file}'...")
    if os.path.exists(training_file):
        with open(training_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
            
            # Convert the JSON structures to the formats expected by Dedupe
            distinct_pairs = []
            for pair in training_data.get('distinct', []):
                if isinstance(pair, dict) and '__class__' in pair and pair['__class__'] == 'tuple':
                    items = pair.get('__value__', [])
                    if len(items) == 2:
                        distinct_pairs.append((items[0], items[1]))
            
            match_pairs = []
            for pair in training_data.get('match', []):
                if isinstance(pair, dict) and '__class__' in pair and pair['__class__'] == 'tuple':
                    items = pair.get('__value__', [])
                    if len(items) == 2:
                        match_pairs.append((items[0], items[1]))
            
            formatted_training = {'match': match_pairs, 'distinct': distinct_pairs}
            
            print(f"Training data loaded with {len(match_pairs)} match examples and {len(distinct_pairs)} distinct examples.")
            logging.info(f"Training data loaded from '{training_file}'.")
    
    # IMPORTANT CHANGE: Gather ALL IDs from the training data
    training_ids = set()
    for pairs in [formatted_training.get('match', []), formatted_training.get('distinct', [])]:
        for record_1, record_2 in pairs:
            training_ids.add(record_1)
            training_ids.add(record_2)
    
    # Ensure all training records are included in the sample
    data_sample = {}
    
    # First add all records from training
    for record_id in training_ids:
        if record_id in df.index:
            row = df.loc[record_id]
            data_sample[row['id']] = {
                'clean_company_name': row['clean_company_name'],
                'normalized_url': row['normalized_url'],
                'main_address_raw_text': row['main_address_raw_text'],
                'normalized_phone': row['normalized_phone']
            }
    
    # Then add additional records to reach the desired sample size
    remaining_sample_size = min(1000, len(df)) - len(data_sample)
    if remaining_sample_size > 0:
        remaining_indices = np.random.choice(
            [idx for idx in df.index if df.loc[idx, 'id'] not in data_sample], 
            remaining_sample_size, 
            replace=False
        )
        for idx in remaining_indices:
            row = df.loc[idx]
            data_sample[row['id']] = {
                'clean_company_name': row['clean_company_name'],
                'normalized_url': row['normalized_url'],
                'main_address_raw_text': row['main_address_raw_text'],
                'normalized_phone': row['normalized_phone']
            }
    
    # Fields for dedupe (repeated for clarity)
    fields = [
        variables.String('clean_company_name', has_missing=True),
        variables.String('normalized_url', has_missing=True),
        variables.String('main_address_raw_text', has_missing=True),
        variables.String('normalized_phone', has_missing=True),
    ]
    
    # Reinitialize deduper
    deduper = Dedupe(fields)
    
    # Prepare training with the sample data that now includes all training records
    deduper.prepare_training(data_sample)
    
    # Apply the formatted training data
    deduper.mark_pairs(formatted_training)
    
    # Train the model 
    # IMPORTANT CHANGE: Ensure that index_predicates is set correctly; here we remove index_predicates=True
    deduper.train(recall=0.5)
    
    # Index the predicates separately after training
    deduper.index(data_sample)

    # Deduplication by blocks
    all_clusters = []
    print("Processing blocks...")
    logging.info("Processing blocks...")
    for key, block in tqdm(blocks, total=len(blocks)):
        if len(block) <= 1:
            continue
        block_dict = {}
        for i, row in block.iterrows():
            record = {
                'clean_company_name': row['clean_company_name'],
                'normalized_url': row['normalized_url'],
                'main_address_raw_text': row['main_address_raw_text'],
                'normalized_phone': row['normalized_phone']
            }
            if is_valid_record(record):
                block_dict[row['id']] = record
        
        if len(block_dict) > 1:
            clustered_dupes = process_block_batch(block_dict, deduper, batch_size=500)
            all_clusters.extend(clustered_dupes)
        
        del block_dict
        gc.collect()
    
    # Results
    cluster_membership = {record_id: cid for cid, (cluster, _) in enumerate(all_clusters) for record_id in cluster}
    df['cluster_id'] = df['id'].map(cluster_membership).fillna(-1).astype(int)
    
    print("\nResults:")
    print(f"Total records: {len(df)}")
    print(f"Clusters: {len(set(cluster_membership.values()))}")
    print(f"Non-duplicates: {(df['cluster_id'] == -1).sum()}")
    logging.info("Results:")
    logging.info(f"Total records: {len(df)}")
    logging.info(f"Clusters: {len(set(cluster_membership.values()))}")
    logging.info(f"Non-duplicates: {(df['cluster_id'] == -1).sum()}")
    
    # Save the complete dataset with clusters
    df.to_csv('entity_resolution_result.csv', index=False)
    print("Results saved in 'entity_resolution_result.csv'")
    logging.info("Results saved in 'entity_resolution_result.csv'")

    # Also save in parquet format
    df.to_parquet('entity_resolution_result.parquet', index=False)
    print("Results also saved in 'entity_resolution_result.parquet'")
    logging.info("Results also saved in 'entity_resolution_result.parquet'")

if __name__ == "__main__":
    main()
