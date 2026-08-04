# Cold-start experiment package

CLI:

```bash
python -m recommender.experiment.cold_start --dataset movielens --mode diagnostic
# MovieLens cold strata (dense ratings):
python -m recommender.experiment.cold_start --dataset movielens --mode controlled --split leave_k
# Ciao trust zero-shot:
python -m recommender.experiment.cold_start --dataset ciao --mode zero_shot_trust
```

Full command reference: `docs/experiments/cold_start_commands.md`.

Method proposal: `docs/experiments/cold_start_experiment_proposal.md`.
