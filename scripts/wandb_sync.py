import wandb

api = wandb.Api()
old_run = api.run("tecchk-cyt-duke-university/amplify_ally/3ycivkmj")

config = old_run.config
history = old_run.history(samples=1000000)

cutoff_step = 40_000
truncated = history[history["_step"] <= cutoff_step]

new_run = wandb.init(project="amplify_ally", entity="tecchk-cyt-duke-university", config=config)

for _, row in truncated.iterrows():
    metrics = row.dropna().to_dict()
    metrics.pop("_step", None)
    new_run.log(metrics, step=int(row["_step"]))

new_run.finish()