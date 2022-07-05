# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
from itertools import product

import xarray as xr
import pandas as pd
import numpy as np
import larch.numba as lx
from larch import P, X

from numpy.random import default_rng


# %%
def one_based(n):
    return pd.RangeIndex(1, n + 1)

def from_numpy(
    numpy_data,
    name,
    index_names=("otaz", "dtaz"),
    indexes="one-based",
    renames=None,
):
    arrays = {name: numpy_data}    
    d = {
        "dims": index_names,
        "data_vars": {name: {"dims": index_names, "data": numpy_data}},
    }
    if indexes == "one-based":
        indexes = {
            index_names[0]: one_based(numpy_data.shape[0]),
            index_names[1]: one_based(numpy_data.shape[1]),
        }
    if indexes is not None:
        d["coords"] = {
            index_name: {"dims": index_name, "data": index}
            for index_name, index in indexes.items()
        }
    return xr.Dataset.from_dict(d)


# %%
num_destinations = 2
num_purposes = 3

# %% [markdown]
# # Create data

# %%
num_alternatives = num_destinations * num_purposes

# %%
choice_mapping = {(dest, purpose): idx + 1 for idx, (dest, purpose) in enumerate(product(range(1, num_destinations + 1), range(1, num_purposes + 1)))}
display(choice_mapping)

# %%
alternatives = list(choice_mapping.values())

# %% [markdown]
# ## attractions

# %%
attrs = pd.DataFrame(
    [(dest, purpose, altid) for (dest, purpose), altid in choice_mapping.items()], 
    columns=["destination", "purpose", "altid"]
)

attrs["attractions"] = np.random.randint(1, 100, size=len(attrs))

attrs = attrs.set_index("altid")

# %%
display(attrs)

# %% [markdown]
# ## skims

# %%
skims = np.zeros((num_destinations, num_destinations))

# %%
skims = from_numpy(skims, "distance")

# %%
display(skims)

# %% [markdown]
# ## observations

# %%
obs = pd.DataFrame({
    "caseid": [1, 2, 3],
    "origin": [1, 2, 2],
    "partial": [False, False, True],
    "destination": [1, 2, 1],
    "chosen": pd.Series([choice_mapping[(1, 2)], choice_mapping[(2, 1)], None], dtype="Int32"),
}).set_index("caseid")

# %%
display(obs)

# %% [markdown]
# # Larch stuff

# %%
tree = lx.DataTree(
    obs=lx.Dataset.construct(obs, caseid="caseid", alts=alternatives),
    attr=attrs,
    skims=skims,
    relationships=(
        "obs._altid_ @ attr.altid",
        "obs.origin @ skims.otaz",
        "attr.destination @ skims.dtaz",
    ),
)

# %%
m = lx.Model(datatree=tree)
m.title = "blah"

m.quantity_ca = P("zero") * X("attractions")
m.quantity_scale = P.Theta

m.utility_ca = P.distance * X.distance

m.choice_co_code = "chosen"

# m.availability_var = "attr.attractions > 0"

# %%
for destination in range(num_destinations):
    m.graph.new_node(parameter='MuDest', children=[choice_mapping[(destination + 1, purpose + 1)] for purpose in range(num_purposes)], name=f"dest_{destination}")

# %%
m.graph

# %%
m.lock_values(
    MuDest=1,
    zero=0,
    Theta=1.,
    #distance=-10.0
)
# m.set_cap(10)

# %%
m.loglike()

# %%
m.d_loglike()

# %%
m.work_arrays

# %%
m.maximize_loglike()

# %%
