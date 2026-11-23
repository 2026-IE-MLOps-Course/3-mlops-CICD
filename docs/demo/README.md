# Optional demo fallback screenshots

If you use screenshots as a fallback for the live demo, prepare these from the unweighted-baseline to balanced-enhancement sequence in docs/DEMO.md:

- `01-pr.png`: the balanced candidate's PR checks
- `02-gate-summary.png`: the actual quality-gate result
- `03-wandb-versions.png`: unweighted and balanced runs/artifacts, their validation comparison and prod
- `04-release.png`: the balanced enhancement's release form using an unused tag
- `05-deploy-run.png`: the release workflow jobs
- `06-publish-summary.png`: published image tags and concrete balanced model version
- `07-health.png`: live health showing the enhancement's commit and model version
- `08-rollback.png`: manual rollback to the baseline, followed by successful roll forward to the enhancement

Use your observed versions and results. These screenshots are optional presentation aids, not an additional student evidence task. The rehearsal ends with the balanced enhancement running and prod aligned with it.
