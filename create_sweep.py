# create_sweep.py
import yaml
import wandb

with open("sweep_config.yaml") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project="ZNEUS_1")
print(f"\n✅ Sweep created successfully!")
print(f"👉 Run the agent with:\nwandb agent nikolaskekelak-fiit-stu/ZNEUS_p1/{sweep_id}\n")
