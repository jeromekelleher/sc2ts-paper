import json
import csv

def create_pango_nextstrain_map(json_file, tsv_file):
    """
    Parses the Pango consensus summary JSON to create a mapping
    between Pango lineages and their corresponding Nextstrain clades.
    JSON is from
    https://github.com/corneliusroemer/pango-sequences/blob/main/data/pango-consensus-sequences_summary.json
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from '{json_file}'.")
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file}'.")
        return

    mapping = []
    # The JSON is a dictionary where each key is a Pango lineage name
    for lineage_name, lineage_data in data.items():
        # The lineage_data is another dictionary containing details
        nextstrain_clade = lineage_data.get('nextstrainClade')
        if nextstrain_clade:
            mapping.append([lineage_name, nextstrain_clade])

    if not mapping:
        print("No mappings between Pango lineage and Nextstrain clade were found in the file.")
        return

    mapping.sort()

    try:
        with open(tsv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['pango_lineage', 'nextstrain_clade'])
            writer.writerows(mapping)
        print(f"Successfully wrote {len(mapping)} mappings to '{tsv_file}'.")
    except IOError as e:
        print(f"Error writing to file '{tsv_file}': {e}")

if __name__ == "__main__":
    INPUT_JSON = 'pango-consensus-sequences_summary.json'
    OUTPUT_TSV = 'pango_to_nextstrain_map.tsv'
    create_pango_nextstrain_map(INPUT_JSON, OUTPUT_TSV)
