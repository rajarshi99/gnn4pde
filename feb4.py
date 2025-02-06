from gcn_poisson_2d import main
from pathlib import Path

fname = "feb4.csv"
with open(fname, "a") as file:
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
            with open(fname, "a") as file:
                print(f"{num_points}, {iters_per_fit}, {learning_rate}, {output_dir}", file=file)
            expt_id += 1
