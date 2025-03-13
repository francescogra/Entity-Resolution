import pandas as pd
import json
import logging
from dedupe import Dedupe, variables, console_label
import os
import re
import phonenumbers

# Configure logging
logging.basicConfig(
    filename='interactive_labeling.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_company_name(name):
    """Cleans the company name by removing legal suffixes and irrelevant characters."""
    if pd.isna(name) or not isinstance(name, str):
        return None
    name = name.lower()
    legal_entities = [' inc', ' inc.', ' llc', ' llc.', ' ltd', ' ltd.,', ' corp', ' corp.']
    for entity in legal_entities:
        name = name.replace(entity, '')
    name = re.sub(r'[^\w\s]', ' ', name)
    cleaned = re.sub(r'\s+', ' ', name).strip()
    return cleaned if cleaned else None

def normalize_url(url):
    """Normalizes URLs by removing prefixes and standardizing the format."""
    if pd.isna(url) or not isinstance(url, str):
        return None
    url = re.sub(r'^https?://|^www\.', '', url.lower()).rstrip('/')
    return url if url else None

def normalize_phone(phone):
    """Normalizes phone numbers to E164 format."""
    if not phone:
        return None
    try:
        parsed = phonenumbers.parse(phone, None)
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        return None

def main():
    logging.info("Initializing Interactive Training...")
    print("Initializing Interactive Training...")

    # Read the dataset
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

    # Prepare the data for Dedupe
    data_for_dedupe = {}
    for _, row in df.iterrows():
        record = {
            'clean_company_name': row['clean_company_name'],
            'normalized_url': row['normalized_url'],
            'main_address_raw_text': row['main_address_raw_text'],
            'normalized_phone': row['normalized_phone']
        }
        data_for_dedupe[row['id']] = record

    # Define the fields for Dedupe
    fields = [
        variables.String('clean_company_name', has_missing=True),
        variables.String('normalized_url', has_missing=True),
        variables.String('main_address_raw_text', has_missing=True),
        variables.String('normalized_phone', has_missing=True),
    ]

    # Initialize Dedupe
    deduper = Dedupe(fields)

    training_file = 'training_data_test.json'

    # Start interactive labeling
    print("Starting Interactive Labeling...")
    logging.info("Starting Interactive Labeling...")
    deduper.prepare_training(data_for_dedupe, sample_size=1500)
    
    # Run interactive labeling
    print("Type 'y' to indicate a match, 'n' to indicate a non-match, 'f' to finish.")
    console_label(deduper)
    
    # Save the updated training data
    with open(training_file, 'w', encoding='utf-8') as f:
        deduper.write_training(f)
    print(f"Training data saved in '{training_file}'.")
    logging.info(f"Training data saved in '{training_file}'.")

if __name__ == "__main__":
    main()