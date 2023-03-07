from dataclasses import dataclass
from datetime import datetime
import os

import numpy as np
import tskit
import tszip

def load_tsz_file(date, filename):
    ts = tszip.decompress(os.path.join("data", filename.format(date)))
    ts.day0 = datetime.fromisoformat(date)  # Hack: this won't be saved / transferred
    return ts

def node_spans_parent(ts, min_num_children=None):
    """
    Returns the spans over which each node is a parent of greater than num_children_greater_than
    children. 
    
    Note - to find the "general node spans", i.e. the spans over which nodes are present in all local
      trees, you may wish to include cases where nodes are children as well as parents. If all leaf
      nodes in your trees are sample nodes, as is usually the case, the general node spans can simply
      be found by setting sample node spans to the total sequence length (as by definition, samples
      are present in all trees in the tree sequence). This can be done as follows
      
          node_span = node_spans_parent(ts)
          node_span[ts.samples()] = ts.sequence_length
    
    
    Note - if you wish to ignore regions over which a node is a unary node, y
    
    Note - this also counts "dead" branches
    
    """
    if min_num_children is None:
        min_num_children = 1
    num_children = np.zeros(ts.num_nodes, dtype=np.int32)
    span_start = np.zeros(ts.num_nodes)
    node_span = np.zeros(ts.num_nodes)

    for interval, edges_out, edges_in in ts.edge_diffs(include_terminal=True):
        touched=set()
        for edge in edges_out:
            num_children[edge.parent] -= 1
            if num_children[edge.parent] == min_num_children - 1:
            # Should no longer count this node
                node_span[edge.parent] += interval.left - span_start[edge.parent]

        for edge in edges_in:
            num_children[edge.parent] += 1
            if num_children[edge.parent] == min_num_children:
                span_start[edge.parent] = interval.left
    
    return node_span

def node_arities(ts):
    span_sums = np.zeros(ts.num_nodes)
    # Find the edge indices where parents change
    i = np.insert(np.nonzero(np.diff(ts.edges_parent))[0] + 1, 0, 0)
    span_sums[ts.edges_parent[i]] = np.add.reduceat(ts.edges_right - ts.edges_left, i)
    node_spans = node_spans_parent(ts)
    node_spans[ts.samples()] = ts.sequence_length

    return span_sums / node_spans