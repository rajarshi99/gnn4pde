import pandas as pd
from pathlib import Path

def make_clickable(val):
    return f'<a href="{val}"> {Path(val).name} </a>'

def make_html_table(csv_fname, row_to_convert, html_fname):
    df = pd.read_csv(csv_fname, skipinitialspace=True)
    df[row_to_convert] = df[row_to_convert].apply(make_clickable)
    df.to_html(html_fname, escape=False, index=False)



