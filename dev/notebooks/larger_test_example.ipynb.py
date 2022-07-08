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

import pandas as pd
import numpy as np
import larch.numba as lx
from larch import P, X

from numpy.random import default_rng

# %%
MAKE_PARTIAL = True

# %%
rng = default_rng(seed=234310)

# %%
hh, pp, tour, skims, emp = lx.example(200, ['hh', 'pp', 'tour', 'skims', 'emp'])

# %%
tour = tour.merge(hh[["HHID", "HOMETAZ"]], on="HHID")
tour = tour[["TOURID", "HOMETAZ"]]
display(tour.head())

# %%
num_zones = skims.shape[0]
num_purposes = 3

# %%
zones = list(range(1, num_zones + 1))
purposes = list(range(1, num_purposes + 1))

# %%
alt_mapping = pd.merge(
    pd.Series(zones, name="DTAZ"),
    pd.Series(purposes, name="purpose"),
    how="cross",
).assign(altid=lambda df: np.arange(1, len(df) + 1))
display(alt_mapping.head())

# %%
alternatives = alt_mapping.altid.tolist()

# %%
attrs = alt_mapping.assign(attractions=lambda df: rng.integers(1, 1000, len(df)))
display(attrs.head())

# %%
auto_dist = pd.DataFrame(np.array(skims["AUTO_DIST"]), columns=zones, index=zones).melt(var_name="DTAZ", value_name="distance", ignore_index=False).rename_axis("OTAZ").reset_index()
display(auto_dist.head())

# %%
all_combos = (
    tour[["TOURID", "HOMETAZ"]]
    .merge(alt_mapping, how="cross")
    .merge(auto_dist.rename(columns={"OTAZ": "HOMETAZ"}), on=["HOMETAZ", "DTAZ"], how="left")
    .merge(attrs[["altid", "attractions"]], on="altid", how="left")
)
display(all_combos)

# %%
all_combos["utility"] = -0.06 * all_combos.distance + np.log(all_combos.attractions) - np.log(-np.log(rng.uniform(size=len(all_combos))))

# %%
chosen = (
    all_combos
    .sort_values(["TOURID", "utility"])
    .drop_duplicates(subset=["TOURID"], keep="last")
    [["TOURID", "altid"]]
    .rename(columns={"altid": "chosen"})
)
display(chosen)

# %% [markdown]
# # MAKE PARTIAL

# %%
if MAKE_PARTIAL:
    num_alts = len(alternatives)

    nest_mapping = {zone: nest + 1 for zone, nest in zip(zones, range(num_alts, num_alts + num_zones))}

    mask = rng.choice(a=[True, False], size=len(chosen), p=[0.3, 0.7])

    chosen = (
        chosen
        .merge(alt_mapping, how="left", left_on="chosen", right_on="altid")
        .assign(chosen=lambda df: df.chosen.mask(mask, 0) + df.DTAZ.map(nest_mapping).mask(~mask, 0))
        [["TOURID", "chosen"]]
    )

# %%
tour = tour.merge(chosen, on="TOURID", how="left")
display(tour.head())

# %%
tree = lx.DataTree(
    obs=lx.Dataset.construct(tour.set_index("TOURID"), caseid="TOURID", alts=alternatives),
    attr=attrs.set_index("altid"),
    skims=lx.Dataset.construct.from_omx(skims),
    relationships=(
        "obs._altid_ @ attr.altid",
        "obs.HOMETAZ @ skims.otaz",
        "attr.DTAZ @ skims.dtaz",
    ),
)

# %%
m = lx.Model(datatree=tree)
m.title = "blah"

m.quantity_ca = P("zero") * X("attractions")
m.quantity_scale = P.Theta

m.utility_ca = P.distance * X.AUTO_DIST # + P.log_distance * X('np.log(AUTO_DIST)') #+ P.purpose_is_one * X("purpose == 1")

m.choice_co_code = "obs.chosen"

m.availability_var = "attr.attractions > 0"



if MAKE_PARTIAL:
    m.partial_obs = True
    for destination in zones:
        m.graph.new_node(
            code=nest_mapping[destination], 
            parameter='MuDest', 
            children=alt_mapping.loc[alt_mapping.DTAZ == destination, "altid"].tolist(), 
            name=f"dest_{destination}"
        )
        
    m.lock_values(
        MuDest=1.,
        zero=0,
        Theta=1.,
    ),
else: 
    m.lock_values(
        zero=0,
        Theta=1.,
    )

m.set_cap(10)

# %%
# %%time
print(f"init ll = {m.loglike()}")
m.maximize_loglike(maxiter=1000, tol=1e-12, method="slsqp")

# %%
m.pbounds

# %%
