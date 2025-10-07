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

with open(fname_txt, "w") as file:
    print("Running script name:", Path(__file__), file=file)
    # print("Expt specific logs", fname_csv, fname_html, file=file)
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
            output_dir = "trial/"
            ):
          """, file=file)

with open(fname_csv, "a") as file:
    print("num_points,iters_per_fit,lr_init,rel_l2_err,path", file=file)

expt_id = 0
hyper_params = {
        "num_points": [4, 8, 16],
        "lr_init": [5e-2, 5e-3],
        "gcn_layers": [10, 9, 8],
        "gcn_neurons": [100, 80],
        "internal_data_points": [16, 8]
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
    path.mkdir(parents=True, exist_ok=True)

    iter_ids, loss_vals, metric_vals, metric_col_names = main(
            num_points = param_dict["num_points"],
            u = u, f_guess = f_guess,
            gcn_layers =[1] + param_dict["gcn_layers"]*[param_dict["gcn_neurons"]] + [1],
            num_iters = 5_000, # Change later
            num_steps = 1,
            lr_init = param_dict["lr_init"],
            lr_final = 1e-4,
            output_dir = output_dir
            )

    print(iter_ids.shape, loss_vals.shape, metric_vals.shape, metric_col_names)

    expt_details = {**param_dict,
                    "loss_vals": loss_vals,
                    "metric_vals": metric_vals,
                    }

    expt_details["loss_mean1K"] = loss_vals[-1000:].mean()
    expt_details["loss_mean2K"] = loss_vals[-2000:].mean()
    expt_details["loss_stdk1K"] = loss_vals[-1000:].std()
    expt_details["loss_stdk2K"] = loss_vals[-2000:].std()
    for col_id,metric_col_name in enumerate(metric_col_names):
        expt_details[metric_col_name + "_mean1K"] = metric_vals[col_id,-1000:].mean()
        expt_details[metric_col_name + "_std1K"] = metric_vals[col_id,-1000:].std()
        expt_details[metric_col_name + "_mean2K"] = metric_vals[col_id,-2000:].mean()
        expt_details[metric_col_name + "_std2K"] = metric_vals[col_id,-2000:].std()

    expt_df = pd.concat([expt_df, pd.DataFrame([expt_details])], ignore_index=True)

    expt_df.to_csv(fname_csv)
