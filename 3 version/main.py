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


def normalize_url(url):
    if pd.isna(url) or not isinstance(url, str):
        return None
    url = re.sub(r'^https?://|^www\.', '', url.lower()).rstrip('/')
    return url if url else None

def clean_company_name(name):
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
    if not phone:
        return None
    try:
        parsed = phonenumbers.parse(phone, None)
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        return None

def create_blocking_key(row):
    name = row['clean_company_name'] or ""
    url = row['normalized_url'] or ""
    country = row['main_country'] if pd.notna(row['main_country']) else "ZZ"

    # Normalize the name using doublemetaphone; use 8 characters for more granularity
    name_key = doublemetaphone(name)[0][:8] if name else "########"

    # Normalize the URL: remove "www" and take the first part after the domain
    url_part = re.sub(r'^www\.|_|-|\.', '', url.lower()).split('.')[0][:6] if url else "######"
    url_key = url_part if len(url_part) >= 3 else "######"  # Avoid keys that are too short

    return f"{name_key}_{url_key}_{country}"

def is_valid_record(record):
    """Check if at least one field is not None and not empty"""
    return bool(
        (record['clean_company_name'] is not None and record['clean_company_name'] != '') or 
        (record['normalized_url'] is not None and record['normalized_url'] != '') or
        (record['main_address_raw_text'] is not None and record['main_address_raw_text'] != '') or
        (record['latlong'] is not None and record['latlong'] != (None, None)) or
        (record['company_commercial_names'] is not None and record['company_commercial_names'] != '') or
        (record['normalized_phone'] is not None and record['normalized_phone'] != '')
    )

def main():
    print("Initializing Entity Resolution...")
    
    # Read dataset
    df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet")
    df = df.sample(frac=0.1)  # Use 9% of the dataset (approximately 3,010 records)
    df['id'] = df.index.astype(str)
    print(f"Dataset dimensions: {df.shape}")
    
    # Preprocessing
    print("Preprocessing...")
    df['clean_company_name'] = df['company_name'].apply(clean_company_name)
    df['normalized_url'] = df['website_url'].apply(normalize_url)
    df['normalized_phone'] = df['primary_phone'].apply(normalize_phone)
    
    # Ensure latitude and longitude are properly handled
    df['main_latitude'] = df['main_latitude'].apply(lambda x: float(x) if pd.notna(x) else None)
    df['main_longitude'] = df['main_longitude'].apply(lambda x: float(x) if pd.notna(x) else None)
    
    df['blocking_key'] = df.apply(create_blocking_key, axis=1)

    # Filter records with all empty fields
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
    
    # Print percentage of missing values for relevant fields
    print("Percentage of missing values per field:")
    missing_percentages = df[['clean_company_name', 'normalized_url', 'main_address_raw_text', 'main_latitude', 
                             'main_longitude', 'company_commercial_names', 'normalized_phone']].isna().mean() * 100
    print(missing_percentages)
    
    # Create blocks
    blocks = df.groupby('blocking_key')
    print(f"Blocks created: {len(blocks)}")
    
    # Fields for dedupe (optimized list)
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

    # Load the existing training.json file, if present
    try:
        with open('training.json', 'r') as f:
            existing_training = json.load(f)
        deduper.read_training(existing_training)  # Load the data into the deduper
        print("Existing training data loaded from 'training.json'.")
    except FileNotFoundError:
        print("No 'training.json' file found. Starting from scratch.")
    except json.JSONDecodeError:
        print("Error decoding 'training.json'. Starting from scratch.")
    
    # Training sample
    training_sample = {}
    print("Preparing training sample...")
    for key, block in blocks:
        if len(block) > 1:
            sample_size = min(len(block), max(5, int(len(block) * 0.1)))
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
                if is_valid_record(record):  # Add only valid records
                    training_sample[row['id']] = record
    
    # Training
    if not training_sample:
        print("Error: No valid training samples found. Check your data preprocessing.")
        return
        
    deduper.prepare_training(training_sample)
    print("Interactive training (respond to 30 pairs)...")
    console_label(deduper)
    deduper.train(recall=0.8, index_predicates=True)
    
    # Merge the new training data with the existing ones
    try:
        with open('training.json', 'r') as f:
            existing_training = json.load(f)
        combined_training = {**existing_training, **deduper.training_data}  # Union of the dictionaries
        print("Training data merged successfully.")
    except FileNotFoundError:
        combined_training = deduper.training_data  # If the file does not exist, use only the new data
    except json.JSONDecodeError:
        print("Error decoding 'training.json'. Using only the new data.")
        combined_training = deduper.training_data

    # Save the merged file
    with open('training.json', 'w') as f:
        json.dump(combined_training, f)
    print("Training data saved in 'training.json'.")
    
    del training_sample
    gc.collect()
    
    # Deduplication by blocks
    all_clusters = []
    print("Processing blocks...")
    for key, block in tqdm(blocks):
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
            
            # Make sure the record is valid before adding
            if is_valid_record(record):
                block_dict[row['id']] = record
        
        if len(block_dict) > 1:
            try:
                clustered_dupes = deduper.partition(block_dict, threshold=0.5)
                all_clusters.extend(clustered_dupes)
            except MemoryError:
                print(f"⚠️ MemoryError in block {key} ({len(block_dict)} records)")
                with open('skipped_blocks.log', 'a') as log:
                    log.write(f"Skipped {key} ({len(block_dict)} records)\n")
            except Exception as e:
                print(f"⚠️ Error in block {key}: {str(e)}")
                with open('error_blocks.log', 'a') as log:
                    log.write(f"Error in {key}: {str(e)}\n")
        
        del block_dict
        gc.collect()
    
    # Results
    cluster_membership = {record_id: cid for cid, (cluster, _) in enumerate(all_clusters) for record_id in cluster}
    df['cluster_id'] = df['id'].map(cluster_membership).fillna(-1).astype(int)
    
    print("\nResults:")
    print(f"Total records: {len(df)}")
    print(f"Clusters: {len(set(cluster_membership.values()))}")
    print(f"Non-duplicates: {(df['cluster_id'] == -1).sum()}")
    
    df.to_csv('entity_resolution_result.csv', index=False)
    print("Results saved in 'entity_resolution_result.csv'")

if __name__ == "__main__":
    main()