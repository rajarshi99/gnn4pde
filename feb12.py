from gcn_poisson_2d import main
from pathlib import Path

from core.make_html_table import make_html_table

fname = "feb12"
fname_csv = fname+".csv"
fname_html = fname+".html"

# MMS from PIGGN numerical expt 3.1.1
def u(x,y):
    return 0.25*(1 - x*x - y*y)

def f(x,y):
    return 1

with open(fname, "w") as file:
    print("Running script name:", Path(__file__), file=file)
    print("Expt specific logs", fname_csv, fname_html, file=file)
    print(""" Part of main when run
def main(
            num_points,
            u = u,
            f = f,
            gcn_layers = [1, 10, 10, 1],
            num_fits = 10,
            iters_per_fit = 100,
            learning_rate = 5e-2,
            output_dir = "trial/"
            ):
          """, file=file)

with open(fname_csv, "a") as file:
    print("num_points,iters_per_fit,learning_rate,rel_l2_err,path", file=file)

expt_id = 0
for num_points in [4, 8, 16]:
    for iters_per_fit in [50, 100, 200, 500]:
        for learning_rate in [5e-2, 5e-3, 1e-3, 5e-4]:
            output_dir = f"{fname}/out{expt_id:03d}/"
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            rel_l2_err = main(num_points=num_points,
                 u=u, f=f,
                 iters_per_fit=iters_per_fit,
                 learning_rate=learning_rate,
                 output_dir=output_dir)
            with open(fname_csv, "a") as file:
                print(f"{num_points}, {iters_per_fit}, {learning_rate}, {rel_l2_err:.2e}, {output_dir}", file=file)
            expt_id += 1

make_html_table(fname_csv, "path", fname_html)
