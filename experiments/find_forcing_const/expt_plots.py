import pandas as pd
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

fname = "hyper_param_robust"
fname_csv = fname+".csv"

output_dir = fname+"_plots/"
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

group_id = 0
output_metrics = [""]
for params in param_iterator:
    param_dict = dict(zip(hyper_params.keys(), params))
    mask = True
    for col,val in param_dict.items():
        mask &= (df[col] == val)
    selected_rows = df[mask]
    if selected_rows.shape[0] == 0:
        continue

    with open(f"{output_dir}details_{group_id}.txt", "w") as f:
        f.write(str(param_dict))

    plt.title("Loss Value")
    plt.scatter(list(range(selected_rows.shape[0])),selected_rows["loss_mean1K"], label="mean 1K")
    plt.scatter(list(range(selected_rows.shape[0])),selected_rows["loss_mean2K"], label="mean 2K")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}loss_{group_id}")
    plt.close()

    plt.title("f Value")
    plt.scatter(list(range(selected_rows.shape[0])),selected_rows["f_val_fn_mean1K"], label="mean 1K")
    plt.scatter(list(range(selected_rows.shape[0])),selected_rows["f_val_fn_mean2K"], label="mean 2K")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}f_{group_id}")
    plt.close()

    plt.title("Error Value")
    plt.scatter(list(range(selected_rows.shape[0])),selected_rows["rel_l2_err_fn_mean1K"], label="mean 1K")
    plt.scatter(list(range(selected_rows.shape[0])),selected_rows["rel_l2_err_fn_mean2K"], label="mean 2K")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}error_{group_id}")
    plt.close()

    plt.title("Penalty Value")
    plt.scatter(list(range(selected_rows.shape[0])),selected_rows["penalty_fun_mean1K"], label="mean 1K")
    plt.scatter(list(range(selected_rows.shape[0])),selected_rows["penalty_fun_mean2K"], label="mean 2K")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}penalty_{group_id}")
    plt.close()

    group_id += 1
