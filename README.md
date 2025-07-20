# gnn4pde

I am trying to use Graph Neural Networks
to solve Boundary Value Problems.
For more details
refer to my [article](https://rajarshi99.github.io/research/gnn4pde.html).

## How to run the code?

I use [uv](https://docs.astral.sh/uv/),
a Python package and project manager.
If you have `uv` installed,
you can run
```sh
uv sync
```
to create a virtual environment.
The project specific details
will be picked up from the
`./pyproject.toml` file.

To run the examples provided,
you will need to activate the virtual environment.
```sh
source .venv/bin/activate
```
Note that most of the example python scripts
require an output folder.
You will have to create it manually for the examples to run.

