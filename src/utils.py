import os
import datetime

import sc2ts
import tszip

def load_tsz(ts_dir, fn):
    "Return a tuple of tree seq and basetime"
    if not os.path.exists(os.path.join(ts_dir, fn)):
        raise FileNotFoundError(
            f"You need to obtain {fn} and place it in {ts_dir}")
    ts = tszip.decompress(os.path.join(ts_dir, fn))
    try:
        basetime = sc2ts.last_date(ts)
    except AssertionError:
        test_node = ts.node(ts.samples()[0])
        basetime = sc2ts.parse_date(test_node.metadata["date"])
        basetime += datetime.timedelta(**{ts.time_units: test_node.time})
    return ts, basetime

def snip_tsz_suffix(fn):
    if fn.endswith(".tsz"):
        fn = fn[:-4]
    if fn.endswith(".ts"):
        fn = fn[:-3]
    return fn
