import pandas as pd
import numpy as np
from dedupe import Dedupe, variables, console_label
import os
import json
import re
from tqdm import tqdm
import gc
from metaphone import doublemetaphone
import phonenumbers
import shutil
import logging
import datetime

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
    legal_entities = [' inc', ' inc.', ' llc', ' llc.', ' ltd', ' ltd.,' ' corp', ' corp.']
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
    """Create a blocking key based on name, URL, country, and city to reduce comparisons."""
    name = row['clean_company_name'] or ""
    url = row['normalized_url'] or ""
    country = row['main_country'] if pd.notna(row['main_country']) else "ZZ"

    # Use doublemetaphone for the name (8 characters for granularity)
    name_key = doublemetaphone(name)[0][:8] if name else "########"

    # Normalize the URL (take the first part of the domain)
    url_part = re.sub(r'^www\.|_|-|\.', '', url.lower()).split('.')[0][:6] if url else "######"
    url_key = url_part if len(url_part) >= 3 else "######"

    # Add a component based on the city
    city = row['main_city'] if pd.notna(row['main_city']) else "ZZ"
    city_key = city.lower()[:4] if city else "ZZZZ"

    return f"{name_key}_{url_key}_{country}_{city_key}"

def is_valid_record(record):
    """Check if a record has at least one non-empty field."""
    return bool(
        (record['clean_company_name'] is not None and record['clean_company_name'] != '') or 
        (record['normalized_url'] is not None and record['normalized_url'] != '') or
        (record['main_address_raw_text'] is not None and record['main_address_raw_text'] != '') or
        (record['latlong'] is not None and record['latlong'] != (None, None)) or
        (record['company_commercial_names'] is not None and record['company_commercial_names'] != '') or
        (record['normalized_phone'] is not None and record['normalized_phone'] != '')
    )

def process_block_batch(block_dict, deduper, batch_size=1000):
    """Process a block in batches to handle memory."""
    keys = list(block_dict.keys())
    all_clusters = []
    
    for i in range(0, len(keys), batch_size):
        batch = {k: block_dict[k] for k in keys[i:i + batch_size]}
        if len(batch) > 1:
            try:
                clustered_dupes = deduper.partition(batch, threshold=0.5)
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
    
    # Load the full dataset
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
    
    # Handle latitude and longitude
    df['main_latitude'] = df['main_latitude'].apply(lambda x: float(x) if pd.notna(x) else None)
    df['main_longitude'] = df['main_longitude'].apply(lambda x: float(x) if pd.notna(x) else None)
    
    df['blocking_key'] = df.apply(create_blocking_key, axis=1)

    # Filter records with all fields empty
    df = df[
        df.apply(lambda row: is_valid_record({
            'clean_company_name': row['clean_company_name'],
            'normalized_url': row['normalized_url'],
            'main_address_raw_text': row['main_address_raw_text'],
            'latlong': (row['main_latitude'], row['main_longitude']),
            'company_commercial_names': row['company_commercial_names'],
            'normalized_phone': row['normalized_phone']
        }), axis=1)
    ].copy()
    
    print(f"Dataset dimensions after filtering: {df.shape}")
    logging.info(f"Dataset dimensions after filtering: {df.shape}")
    
    # Print percentage of missing values per field
    print("Percentage of missing values per field:")
    missing_percentages = df[['clean_company_name', 'normalized_url', 'main_address_raw_text', 'main_latitude', 
                             'main_longitude', 'company_commercial_names', 'normalized_phone']].isna().mean() * 100
    print(missing_percentages)
    logging.info("Percentage of missing values per field:")
    logging.info(missing_percentages.to_string())
    
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
    
    # Fields for dedupe
    fields = [
        variables.String('clean_company_name', has_missing=True),
        variables.String('normalized_url', has_missing=True),
        variables.String('main_address_raw_text', has_missing=True),
        variables.LatLong('latlong', has_missing=True),
        variables.String('company_commercial_names', has_missing=True),
        variables.String('normalized_phone', has_missing=True),
    ]
    
    # Initialize deduper
    deduper = Dedupe(fields)

    # Load the existing training.json file (for information only, not used directly)
    existing_training = {}
    try:
        with open('training.json', 'r') as f:
            existing_training = json.load(f)
        print("Existing training data loaded from 'training.json' (not used directly).")
        logging.info("Existing training data loaded from 'training.json' (not used directly).")
    except FileNotFoundError:
        print("No 'training.json' file found. Starting from scratch.")
        logging.info("No 'training.json' file found. Starting from scratch.")
    except json.JSONDecodeError:
        print("Error decoding 'training.json'. Starting from scratch.")
        logging.info("Error decoding 'training.json'. Starting from scratch.")
        existing_training = {}

    # Training sample
    training_sample = {}
    print("Preparing training sample...")
    logging.info("Preparing training sample...")
    for key, block in blocks:
        if len(block) > 1:
            # Sample up to 5 records per block
            sample_size = min(len(block), 5)
            block_sample = block.sample(n=sample_size)
            for i, row in block_sample.iterrows():
                record = {
                    'clean_company_name': row['clean_company_name'],
                    'normalized_url': row['normalized_url'],
                    'main_address_raw_text': row['main_address_raw_text'],
                    'latlong': (row['main_latitude'], row['main_longitude']),
                    'company_commercial_names': row['company_commercial_names'],
                    'normalized_phone': row['normalized_phone']
                }
                if is_valid_record(record):
                    training_sample[row['id']] = record
    
    # Training
    if not training_sample:
        print("Error: No valid training samples found.")
        logging.error("No valid training samples found.")
        return
        
    deduper.prepare_training(training_sample)
    print("Interactive training (classify pairs, press 'f' to finish whenever you want)...")
    logging.info("Interactive training (classify pairs, press 'f' to finish whenever you want)...")
    console_label(deduper)  # This will use the existing data if present in training.json
    deduper.train(recall=0.8, index_predicates=True)
    
    # Save the updated training data
    try:
        # Create a backup of the existing training.json file
        if os.path.exists('training.json'):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            shutil.copy('training.json', f'training_backup_{timestamp}.json')
            print(f"Backup of training.json created: training_backup_{timestamp}.json")
            logging.info(f"Backup of training.json created: training_backup_{timestamp}.json")
        
        # Save the updated training data
        with open('training.json', 'w') as f:
            deduper.write_training(f)
        print("Training data saved in 'training.json'.")
        logging.info("Training data saved in 'training.json'.")
    except Exception as e:
        print(f"Error saving 'training.json': {str(e)}")
        logging.error(f"Error saving 'training.json': {str(e)}")
    
    del training_sample
    gc.collect()
    
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
                'latlong': (row['main_latitude'], row['main_longitude']),
                'company_commercial_names': row['company_commercial_names'],
                'normalized_phone': row['normalized_phone']
            }
            if is_valid_record(record):
                block_dict[row['id']] = record
        
        if len(block_dict) > 1:
            # Process the block in batches
            clustered_dupes = process_block_batch(block_dict, deduper, batch_size=1000)
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
    
    # Save the full dataset with clusters
    df.to_csv('entity_resolution_result.csv', index=False)
    print("Results saved in 'entity_resolution_result.csv'")
    logging.info("Results saved in 'entity_resolution_result.csv'")

    # Also save in parquet format
    df.to_parquet('entity_resolution_result.parquet', index=False)
    print("Results also saved in 'entity_resolution_result.parquet'")
    logging.info("Results also saved in 'entity_resolution_result.parquet'")

if __name__ == "__main__":
    main()
