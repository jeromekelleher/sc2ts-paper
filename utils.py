import os

import sc2ts
import tszip

def load_tsz(ts_dir, fn):
    "Return a tuple of tree seq and basetime"
    if not os.path.exists(os.path.join(ts_dir, fn)):
        raise FileNotFoundError(
            f"You need to obtain {fn} and place it in {ts_dir}")
    ts = tszip.decompress(os.path.join(ts_dir, fn))
    return ts, sc2ts.last_date(ts)

def snip_tsz_suffix(fn):
    if fn.endswith(".tsz"):
        fn = fn[:-4]
    if fn.endswith(".ts"):
        fn = fn[:-3]
    return fn
