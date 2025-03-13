import pandas as pd
import numpy as np
from dedupe import Dedupe, variables, console_label
import os
import json
import re
from tqdm import tqdm
import gc

def normalize_url(url):
    if pd.isna(url) or not isinstance(url, str):
        return None  # Return None instead of empty string
    url = re.sub(r'^https?://|^www\.', '', url.lower()).rstrip('/')
    return url if url else None  # Return None if empty

def clean_company_name(name):
    if pd.isna(name) or not isinstance(name, str):
        return None  # Return None instead of empty string
    name = name.lower()
    legal_entities = [' inc', ' inc.', ' llc', ' llc.', ' ltd', ' ltd.', ' corp', ' corp.']
    for entity in legal_entities:
        name = name.replace(entity, '')
    name = re.sub(r'[^\w\s]', ' ', name)
    cleaned = re.sub(r'\s+', ' ', name).strip()
    return cleaned if cleaned else None  # Return None if empty

def create_blocking_key(row):
    key = ""
    name = row['clean_company_name']
    url = row['normalized_url']
    country = row['main_country']
    key += (name[0:3] if name else "###")
    key += "_" + (url.split('.')[0][0:3] if url else "###")
    key += "_" + (country if pd.notna(country) else "ZZ")
    return key

def is_valid_record(record):
    """Check if at least one field is not None and not empty"""
    return bool(
        (record['clean_company_name'] is not None and record['clean_company_name'] != '') or 
        (record['normalized_url'] is not None and record['normalized_url'] != '') or 
        (record['main_address_raw_text'] is not None and record['main_address_raw_text'] != '')
    )

def main():
    print("Initializing Entity Resolution...")
    
    # Read dataset
    df = pd.read_parquet("veridion_entity_resolution_challenge.snappy.parquet")
    df = df.sample(frac=0.1)  # Use only 10% of data (about 3,344 records)
    df['id'] = df.index.astype(str)
    print(f"Dataset dimensions: {df.shape}")
    
    # Preprocessing
    print("Preprocessing...")
    df['clean_company_name'] = df['company_name'].apply(clean_company_name)
    df['normalized_url'] = df['website_url'].apply(normalize_url)
    
    # Ensure main_address_raw_text is not an empty string
    df['main_address_raw_text'] = df['main_address_raw_text'].apply(
        lambda x: x if pd.notna(x) and x.strip() != '' else None
    )
    
    df['blocking_key'] = df.apply(create_blocking_key, axis=1)

    # Filter records with all empty fields
    df = df[
        df.apply(lambda row: is_valid_record({
            'clean_company_name': row['clean_company_name'],
            'normalized_url': row['normalized_url'],
            'main_address_raw_text': row['main_address_raw_text']
        }), axis=1)
    ].copy()
    
    print(f"Dataset dimensions after filtering: {df.shape}")
    
    # Create blocks
    blocks = df.groupby('blocking_key')
    print(f"Blocks created: {len(blocks)}")
    
    # Fields for dedupe
    fields = [
        variables.String('clean_company_name', has_missing=True),
        variables.String('normalized_url', has_missing=True),
        variables.String('main_address_raw_text', has_missing=True)
    ]
    
    # Initialize deduper
    deduper = Dedupe(fields)
    
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
                    'main_address_raw_text': row['main_address_raw_text']
                }
                if is_valid_record(record):  # Add only valid records
                    training_sample[row['id']] = record
    
    # Training
    if not training_sample:
        print("Error: No valid training samples found. Check your data preprocessing.")
        return
        
    deduper.prepare_training(training_sample)
    print("Interactive training (respond to 15-20 pairs)...")
    console_label(deduper)
    deduper.train(recall=0.8, index_predicates=True)
    
    with open('training.json', 'w') as f:
        deduper.write_training(f)
    
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
            clean_name = row['clean_company_name'] 
            norm_url = row['normalized_url']
            address = row['main_address_raw_text']
            
            record = {
                'clean_company_name': clean_name,
                'normalized_url': norm_url,
                'main_address_raw_text': address
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