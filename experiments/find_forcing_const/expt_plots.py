import pandas as pd
from pathlib import Path
import itertools

fname = "hyper_param_robust"
fname_csv = fname+".csv"

output_dir = fname+"_plots"
path = Path(output_dir)
path.mkdir(exist_ok=True)

df = pd.read_csv(fname_csv)
print(df.describe())

hyper_params = {
        "num_points": [4, 8, 16],
        "lr_init": [5e-2, 5e-3],
        "gcn_layers": [10, 9, 8],
        "gcn_neurons": [100, 80],
        "internal_data_points": [2, 1]
        }

param_iterator = itertools.product(*[
    hyper_params[key] for key in hyper_params
    ])

for params in param_iterator:
    param_dict = dict(zip(hyper_params.keys(), params))
    mask = True
    for col,val in param_dict.items():
        mask &= (df[col] == val)
    selected_rows = df[mask]
    print(type(selected_rows))
    print(selected_rows)
