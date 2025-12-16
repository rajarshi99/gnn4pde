import pandas as pd
from find_forcing_const import main
from pathlib import Path

# from core.make_html_table import make_html_table
import itertools

fname = "hyper_param_robust"
fname_txt = fname+".txt"
fname_csv = fname+".csv"
# fname_html = fname+".html"

def u(x,y):
    return 1.2*0.25*(1 - x*x - y*y)

def f_guess(x,y):
    return 1

# Create new csv log file if it already exists
run_id = 1
while Path(fname_csv).exists():
    fname_csv = f"{fname}_{run_id}.csv"
    run_id += 1

with open(fname_txt, "w") as file:
    print("Running script name:", Path(__file__), file=file)
    print("Log @", fname_csv, file=file)
    print(""" Part of main when run
def main(
            num_points,
            u = u,
            f_guess = f_guess,
            gcn_layers = [1, 10, 10, 1],
            num_iters = 10,
            num_steps = 1,
            lr_init = 0.001,
            lr_final = 0.0001,
            num_internal_data_points = 2,
            output_dir = "trial/",
            input_prng_key = 42
            ):
          """, file=file)

expt_id = 0
hyper_params = {
        "num_points": [4, 8, 16],
        "lr_init": [5e-2, 5e-3],
        "gcn_layers": [10, 9, 8],
        "gcn_neurons": [100, 80],
        "internal_data_points": [2, 1],
        "input_prng_key": [42, 69, 99, 1729, 2025]
        }

# Create an iterator over the hyper parameters
param_iterator = itertools.product(*[
    hyper_params[key] for key in hyper_params
    ])

expt_df = pd.DataFrame()
# expt_df = pd.DataFrame(columns=list(hyper_params.keys()) + ["loss_vals", "metric_vals"])

# Iterate over the hyper parameters
for params in param_iterator:
    # Create a dictionary for the current set of hyper parameters
    param_dict = dict(zip(hyper_params.keys(), params))
    print(expt_id, param_dict)
    expt_id += 1

    output_dir = f"{fname}_out/expt{expt_id:03d}/"
    path = Path(output_dir)
    if path.exists():
        print("Path already exists. Avoiding overwrite. SKIP", path)
        continue
    path.mkdir(parents=True)

    iter_ids, loss_vals, metric_vals, metric_col_names, init_time, train_time = main(
            num_points = param_dict["num_points"],
            u = u, f_guess = f_guess,
            gcn_layers =[1] + param_dict["gcn_layers"]*[param_dict["gcn_neurons"]] + [1],
            num_iters = 50_000, # Change later
            num_steps = 1,
            lr_init = param_dict["lr_init"],
            lr_final = 1e-4,
            num_internal_data_points = param_dict["internal_data_points"],
            output_dir = output_dir,
            input_prng_key = param_dict["input_prng_key"]
            )

    print(iter_ids.shape, loss_vals.shape, metric_vals.shape, metric_col_names)

    expt_details = {**param_dict,
                    "init_time": init_time,
                    "train_time": train_time,
                    "output_dir": output_dir
                    }

    expt_details["loss_mean1K"] = loss_vals[-1000:].mean()
    expt_details["loss_mean2K"] = loss_vals[-2000:].mean()
    expt_details["loss_stdk1K"] = loss_vals[-1000:].std()
    expt_details["loss_stdk2K"] = loss_vals[-2000:].std()
    for col_id,metric_col_name in enumerate(metric_col_names):
        print(metric_col_name, col_id, metric_vals.shape, metric_vals[0, -3:, col_id])
        expt_details[metric_col_name + "_mean1K"] = metric_vals[0, -1000:, col_id].mean()
        expt_details[metric_col_name + "_std1K"] = metric_vals[0, -1000:, col_id].std()
        expt_details[metric_col_name + "_mean2K"] = metric_vals[0, -2000:, col_id].mean()
        expt_details[metric_col_name + "_std2K"] = metric_vals[0, -2000:, col_id].std()

    expt_df = pd.concat([expt_df, pd.DataFrame([expt_details])], ignore_index=True)

    expt_df.to_csv(fname_csv)
