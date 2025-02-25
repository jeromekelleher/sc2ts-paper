import collections
from datetime import datetime, timedelta
import json
import os
import requests
import fileinput

import sc2ts
import tskit
import tszip
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex


NODE_REPORT_KEYS = ", ".join([
    "title",
    "metadata",
    "node_title",
    "parents",
    "edge_title",
    "edges",
    "copying_title",
    "copying_pattern",
    "lpath_title",
    "closest_lrecomb",
    "lft_path",
    "rpath_title",
    "closest_rrecomb",
    "rgt_path",
    "children_title",
    "children",
    "mutations_title",
    "mutations"]
)


def load(filename = "v1-beta1_2023-02-21.pp.md.ts.dated.il.tsz"):
    ts_dir = "../data"
    ts = tszip.decompress(os.path.join(ts_dir, filename))
    print(
        f"Loaded {ts.nbytes/1e6:0.1f} megabyte SARS-CoV2 genealogy of {ts.num_samples} strains",
        f"({ts.num_trees} trees, {ts.num_mutations} mutations over {ts.sequence_length} basepairs).",
        f"Last collection date is {ts.node(ts.samples()[-1]).metadata['date']}",
    )
    return ts

def date(ts, node_id):
    return (
        datetime.fromisoformat(ts.node(1).metadata["date"]) + 
        timedelta(days=int(ts.node(1).time) - ts.node(node_id).time)
    )

def remove_single_descendant_re_nodes(ts):
    re_nodes = np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)[0]
    single_sample_re_nodes = []
    for u in re_nodes:
        children = np.unique(ts.edges_child[ts.edges_parent==u])
        if len(children) == 1 and children[0] in ts.samples():
            single_sample_re_nodes.append(u)
    tables = ts.dump_tables()
    nodes_flags = tables.nodes.flags
    nodes_flags[single_sample_re_nodes] = 0
    tables.nodes.flags = nodes_flags
    tables.simplify(list(set(ts.samples()) - set(ts.edges_child[np.isin(ts.edges_parent, single_sample_re_nodes)])), filter_nodes=False, keep_unary=True)
    return tables.tree_sequence()

def oldest_imputed(ts):
    oldest_imputed = collections.defaultdict(lambda: tskit.Node(-1, 0, 0, 0, 0, b""))
    for nd in tqdm(ts.nodes(), desc="Find oldest node for imputed Pangos"):
        pango = nd.metadata["Imputed_Viridian_pangolin"]
        if nd.time > oldest_imputed[pango].time:
            oldest_imputed[pango] = nd
    return oldest_imputed

def fetch_genbank_comment(accession):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&rettype=gb&retmode=text"
    url += f"&id={accession}"
    response = requests.get(url)
    for line in response.text.split('\n'):
        if line.strip().startswith('COMMENT'):
            return line.strip()        
    return ""


def set_sc2ts_labels_and_styles(d3arg, ts, add_strain_names=True):
    # Set node labels to Pango lineage + strain, if it exists
    # A questin mark at the end of a pango lineage indicates that the lineage is imputed
    def label_lines(md):
        s = md.get('strain', '')
        if s == "Vestigial_ignore":
            return([""])
        imputed = md.get("Imputed_Viridian_pangolin", "")
        if not imputed.startswith("Unknown"):
            imputed += "?"  # show this label is imputed using a question mark
        if add_strain_names:
            return [md.get("Viridian_pangolin", imputed), f"({s})"]
        else:
            return [md.get("Viridian_pangolin", imputed)]

    nodes = set(d3arg.nodes.id)
    d3arg.set_node_labels({
        u: "\n".join([s for s in label_lines(ts.node(u).metadata) if s not in ("()", "?")])
        for u in tqdm(range(ts.num_nodes), desc="Setting all labels")
        if u in nodes
    })
    # Mark recombination nodes in white and samples as squares
    d3arg.nodes.loc[:, "size"] = 50
    d3arg.nodes.loc[:, "fill"] = "darkgrey"
    d3arg.nodes.loc[:, "stroke_width"] = 1
    is_sample = np.isin(d3arg.nodes["id"], ts.samples())
    d3arg.nodes.loc[is_sample, "symbol"] = 'd3.symbolSquare'
    d3arg.nodes.loc[is_sample, "size"] = 100
    d3arg.nodes.loc[is_sample, "fill"] = "lightgrey"
    d3arg.set_node_styles([{"id": u, "fill": "white", "size": 150} for u in np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)[0]])
    

def plot_sc2ts_subgraph(
    d3arg,
    nodes,
    parent_levels=20,
    child_levels=1,
    *,
    height=1000,
    width=800,
    title=None,
    cmap=plt.cm.tab10,
    y_axis_scale="rank",

    condense_mutations=False,
    return_included_nodes=None,
):
    """
    Display a subset of the sc2ts arg, with mutations that are recurrent or reverted
    in the subgraph coloured by position (reversions have a black outline).
    """
    
    # Find mutations with duplicate position values and the same alleles (could be parallel eg. A->T & A->T, or reversions, e.g. A->T, T->A)
    # Create a composite key for basic duplicates
    if colour_recurrent_mutations:
        df = d3arg.subset_graph(nodes, (parent_levels, child_levels)).mutations.copy()
        df['fill'] = "white"
        df['stroke'] = "grey" # default stroke color
        df['duplicate_key'] = df.apply(lambda row: f"{row['position']}_{sorted([row['inherited'], row['derived']])}", axis=1)
        # Create a polarization key to identify the polarization of the allele arrangements
        df['polarization_key'] = df.apply(lambda row: f"{row['position']}_{row['inherited']}_{row['derived']}", axis=1)
        
        # Identify which rows are duplicates
        duplicate_mask = df.duplicated(subset=['duplicate_key'], keep=False)
            
        # Get only the duplicate keys that appear multiple times
        duplicate_keys = df[duplicate_mask]['duplicate_key'].unique()
        colors = {key: rgb2hex(cmap(i))
            for i, key in enumerate(duplicate_keys)
        }
        
        # Process only the duplicate groups
        for duplicate_key in duplicate_keys:
            group = df[df['duplicate_key'] == duplicate_key]
            polarization_keys = group['polarization_key'].unique()
                
            # Assign fill color based on duplicate_key
            fill = colors[duplicate_key]
            # If there are multiple direction keys, assign different strokes
            has_multiple_directions = len(polarization_keys) > 1
            for idx in group.index:
                df.loc[idx, 'fill'] = fill
                df.loc[idx, 'stroke'] = (
                    'grey' if not has_multiple_directions else 
                    ('grey' if group.loc[idx, 'polarization_key'] == polarization_keys[0] else 'black')
                )
        
        d3arg.mutations.loc[df.index, 'fill'] = df['fill']
        d3arg.mutations.loc[df.index, 'stroke'] = df['stroke']

    shown_nodes = d3arg.draw_nodes(
        nodes,
        degree=(parent_levels, child_levels),
        height=height,
        width=width,
        show_mutations=True,
        y_axis_scale=y_axis_scale,
        include_mutation_labels=False,
        title=title,
        condense_mutations=condense_mutations,
        return_included_nodes=return_included_nodes
    )
    if return_included_nodes:
        return shown_nodes

def set_x_01_from_json(d3arg, json_file):
    default = 0.5
    with open(json_file) as f:
        j = json.load(f)
        x_pos = {nd["id"]: nd.get(("x"), nd.get("fx", default)) for nd in j["data"]["nodes"]}
        mx, mn = max(x_pos.values()), min(x_pos.values())
        x_pos_01 = {k: (v-mn)/(mx-mn) for k, v in x_pos.items()}
        d3arg.nodes["x_pos_01"] = d3arg.nodes.id.map(x_pos_01)
        # Unspecified nodes in the middle
        d3arg.nodes.fillna({"x_pos_01": default}, inplace=True)
        return x_pos  # Just for debugging
    
def clear_x_01(d3arg):
    d3arg.nodes.drop(columns=["x_pos_01"], errors="ignore", inplace=True)

## from run_lineage_imputation.py
class MutationContainer:
    def __init__(self):
        self.names = {}
        self.positions = []
        self.alts = []
        self.size = 0
        self.all_positions = {}

    def add_root(self, root_lineage_name):
        self.names[root_lineage_name] = self.size
        self.size += 1
        self.positions.append([])
        self.alts.append([])

    def add_item(self, item, position, alt):
        if item not in self.names:
            self.names[item] = self.size
            self.positions.append([position])
            self.alts.append([alt])
            self.size += 1
        else:
            index = self.names[item]
            self.positions[index].append(position)
            self.alts[index].append(alt)
        # map each position to a set of alt alleles
        if position in self.all_positions:
            self.all_positions[position].add(alt)
        else:
            self.all_positions[position] = {alt}
            
    def get_mutations(self, item):
        index = self.names[item]
        return self.positions[index], self.alts[index]

## from run_lineage_imputation.py
def read_in_mutations(
    json_filepath,
    verbose=False,
    exclude_positions=None,
):
    """
    Read in lineage-defining mutations from COVIDCG input json file.
    Assumes root lineage is B.
    """
    if exclude_positions is None:
        exclude_positions = set()
    else:
        exclude_positions = set(exclude_positions)
    with fileinput.hook_compressed(json_filepath, "r") as file:
        linmuts = json.load(file)

    # Read in lineage defining mutations
    linmuts_dict = MutationContainer()
    linmuts_dict.add_root("B")
    if verbose:
        check_multiallelic_sites = collections.defaultdict(
            set
        )  # will check how many multi-allelic sites there are

    excluded_pos = collections.defaultdict(int)
    excluded_del = collections.defaultdict(int)
    for item in linmuts:
        if item["pos"] in exclude_positions:
            excluded_pos[item["pos"]] += 1
        elif item["ref"] == "-" or item["alt"] == "-":
            excluded_del[item["pos"]] += 1
        else:
            linmuts_dict.add_item(item["name"], item["pos"], item["alt"])
            if verbose:
                check_multiallelic_sites[item["pos"]].add(item["ref"])
            if verbose:
                check_multiallelic_sites[item["pos"]].add(item["alt"])

    if verbose:
        multiallelic_sites_count = 0
        for value in check_multiallelic_sites.values():
            if len(value) > 2:
                multiallelic_sites_count += 1
        print(
            "Multiallelic sites:",
            multiallelic_sites_count,
            "out of",
            len(check_multiallelic_sites),
        )
        print("Number of lineages:", linmuts_dict.size)
        if len(excluded_pos) > 0:
            print(
                f"Excluded {len(excluded_pos)} positions not in ts:",
                f"{list(excluded_pos.keys())}"
            )
        if len(excluded_del) > 0:
            print("Excluded deletions at positions", list(excluded_del.keys()))

    return linmuts_dict