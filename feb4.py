from gcn_poisson_2d import main
from pathlib import Path

from core.make_html_table import make_html_table

fname = "feb4"
fname_csv = fname+".csv"
fname_html = fname+".html"

with open(fname, "w") as file:
    print("Running script name:", Path(__file__), file=file)
    print("Expt specific logs", fname_csv, fname_html, file=file)
    print(""" Part of main when run
def u(x,y):
    return x*y

def f(x,y):
    return 0

def main(
            num_points, # [4, 8, 16]:
            gcn_layers = [1, 10, 10, 1],
            num_fits = 10,
            iters_per_fit = 100, # [50, 100, 200, 500]:
            learning_rate = 5e-2, # [5e-2, 5e-3, 1e-3, 5e-4]:
            ):
          """, file=file)

with open(fname_csv, "a") as file:
    print("num_points,iters_per_fit,learning_rate,path", file=file)

expt_id = 0
for num_points in [4, 8, 16]:
    for iters_per_fit in [50, 100, 200, 500]:
        for learning_rate in [5e-2, 5e-3, 1e-3, 5e-4]:
            output_dir = f"gcn_poisson_2d_expts/out{expt_id:03d}/"
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            main(num_points=num_points,
                 iters_per_fit=iters_per_fit,
                 learning_rate=learning_rate,
                 output_dir=output_dir)
            with open(fname_csv, "a") as file:
                print(f"{num_points}, {iters_per_fit}, {learning_rate}, {output_dir}", file=file)
            expt_id += 1

make_html_table(fname_csv, "path", fname_html)
