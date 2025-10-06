# -*- coding: utf-8 -*-
"""
This script parses a Nextstrain NEXUS tree file to find the oldest
node for each defined clade and extracts its metadata.
"""

import csv
import re
import io
from Bio import Phylo

def parse_nexus_comments(comment_str):
    if not comment_str:
        return {}
    pattern = re.compile(r'([^,=&]+)=({[^}]+}|[^,]+)')
    attributes = dict(pattern.findall(comment_str))
    return attributes

def find_oldest_clades(tree_file):
    try:
        with open(tree_file, 'r', encoding='utf-8') as f:
            nexus_content = f.read()

        # To handle "(", ")" inside metadata
        def sanitize_comments(match):
            return match.group(0).replace('(', '_').replace(')', '_')

        sanitized_content = re.sub(r'\[&[^\]]*\]', sanitize_comments, nexus_content)
        handle = io.StringIO(sanitized_content)
        tree = Phylo.read(handle, 'nexus')
        print("Successfully parsed the tree file.")

    except FileNotFoundError:
        print(f"Error: File '{tree_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading or parsing the tree file: {e}")
        return None

    oldest_clades = {}
    nodes_processed = 0

    for node in tree.find_clades():
        nodes_processed += 1
        if node.comment:
            attributes = parse_nexus_comments(node.comment)

            clade_name = attributes.get('clade_membership')
            date_str = attributes.get('num_date')
            node_id = node.name if node.name else 'NA'

            if clade_name and date_str:
                try:
                    date = float(date_str)
                    
                    if clade_name not in oldest_clades or date < oldest_clades[clade_name]['date']:
                        oldest_clades[clade_name] = {
                            'node_id': node_id,
                            'date': date,
                            'date_CI': attributes.get('num_date_CI', 'NA')
                        }
                except (ValueError, TypeError):
                    continue
    
    return oldest_clades

def write_output_tsv(data, output_file):
    if not data:
        print("No clade data found.")
        return
        
    sorted_clade_names = sorted(data.keys())

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
            header = ['clade_membership', 'oldest_node_id', 'date', 'date_CI']
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(header)
            
            for clade_name in sorted_clade_names:
                clade_info = data[clade_name]
                original_clade_name = clade_name.replace('_', ' ').strip()
                writer.writerow([
                    original_clade_name,
                    clade_info['node_id'],
                    clade_info['date'],
                    clade_info['date_CI']
                ])
                
        print(f"Successfully wrote data to '{output_file}'")
    except IOError as e:
        print(f"Error writing to file '{output_file}': {e}")

if __name__ == "__main__":
    NEXUS_FILE_PATH = 'nextstrain_ncov_gisaid_global_all-time_timetree-2023-01-21.nex'
    OUTPUT_TSV_PATH = 'oldest_clade_nodes.tsv'
    
    oldest_clade_data = find_oldest_clades(NEXUS_FILE_PATH)
    
    if oldest_clade_data:
        write_output_tsv(oldest_clade_data, OUTPUT_TSV_PATH)

