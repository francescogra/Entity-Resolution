# Entity-Resolution
Identify unique companies and group duplicate records accordingly.

The dataset contains company records imported from multiple systems, leading to duplicate entries with slight variations.


# Usage Instructions
Follow these steps to set up and run the entity resolution system:

## 1. Environment Setup
Create and activate a virtual environment:

python -m venv veridion_env

### Activate the virtual environment
### On Windows:
veridion_env\Scripts\activate

### On macOS/Linux:
source veridion_env/bin/activate


## 2. Install Dependencies
Create a requirements.txt file with the following content:

pandas
pyarrow
numpy
dedupe
tqdm
metaphone
phonenumbers
fuzzywuzzy
python-Levenshtein
scikit-learn

Install all dependencies:
pip install -r requirements.txt


## 3. Data Preparation
Place your data file veridion_entity_resolution_challenge.snappy.parquet in the same directory as the script.


## 4. Run the Entity Resolution Script
Execute the main script:
python main.py


## 5. Interactive Training
During execution, the script will enter an interactive training phase:

You'll be presented with pairs of records to classify as either duplicates or non-duplicates
Type y to mark as duplicate, n to mark as non-duplicate
Press f when you're done training
The training data will be saved to training.json for future use


## 6. Results
After processing, the script will:

Display summary statistics about the identified clusters
Save results to entity_resolution_result.csv and entity_resolution_result.parquet
Create a log file entity_resolution.log with detailed execution information


## 7. Analyzing Results
The output files will contain all original data plus a new cluster_id column:

Records with the same cluster_id are considered duplicates of the same company
Records with cluster_id = -1 are unique companies with no duplicates


## Troubleshooting
If you encounter memory issues:

Decrease the batch_size parameter in the process_block_batch function
Ensure you have sufficient RAM available (at least 8GB recommended)
Check the log file for detailed error messages


## If you're getting unexpected results:

Provide more training examples during the interactive phase
Adjust the threshold parameter in the deduper.partition method (default is 0.7)
Modify the create_blocking_key function to create more specific blocks
