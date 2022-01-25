# Required Imports
import os
from argparse import ArgumentParser

# Create argparser for env name
arg = ArgumentParser()
arg.add_argument(name="env_name", help="Conda Environment Name", required=True, type=str)
opt_map = arg.parse_args()

# Create env, activate and install dependencies
os.system(f"conda create -n {opt_map.env_name} python=3.8")
os.system(f"conda activate {opt_map.env_name}")
os.system("pip install -r requirements.txt")
