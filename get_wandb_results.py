import wandb
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="GNN Unlearning on Injection Attack Testing")
parser.add_argument(
    "--username",
    type=str,
    default="your-username",
    help="W&B username",
)
parser.add_argument(
    "--project",
    type=str,
    help="W&B project name",
)
args = parser.parse_args()

# Authenticate with your W&B API key
api = wandb.Api()

# Specify your W&B project and entity (user/team name)
entity = args.username 
project = args.project

# Fetch runs
runs = api.runs(f"{entity}/{project}")

# Extract run data
summary_list = []
config_list = []
name_list = []
for run in runs:
    # Run summary metrics
    summary_list.append(run.summary._json_dict)

    # Run config parameters
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    config_list.append(config)
    
    # Run name
    name_list.append(run.name)

# Create a DataFrame
summary_df = pd.DataFrame(summary_list)
config_df = pd.DataFrame(config_list)
name_df = pd.DataFrame(name_list, columns=["name"])

# Concatenate all data
runs_df = pd.concat([name_df, config_df, summary_df], axis=1)

# Export to CSV
runs_df.to_csv("wandb_runs.csv", index=False)
