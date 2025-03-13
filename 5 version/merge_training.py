import json

# List of JSON file paths
docs = [
    'training_backup_20250307_194627.json',
    'training_backup_20250308_162906.json',
    'training_backup_20250309_120900.json',
    'training.json'
]

# Initialize the output dictionary
training_data = {
    "distinct": [],
    "match": []
}

# Merge the data
for file_path in docs:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for pair in data["distinct"]:
            training_data["distinct"].append(pair["__value__"])
        for pair in data["match"]:
            training_data["match"].append(pair["__value__"])

# Save the result
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print("Training file saved as 'training_data.json'")
