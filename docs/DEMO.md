# Class demo: from pull request to live, traceable prediction (10 minutes)

Goal: show that a person decides **what** ships (W&B alias `prod`) and **when** (a GitHub Release), and automation does the rest, with a rollback by tag.

## Before class (15 minutes earlier)

- Set the live URL once (README section 2): `export RENDER_URL=https://<new-render-service>.onrender.com`
- Wake Render: `python scripts/smoke_test.py --url "$RENDER_URL" --timeout 300`
- Open tabs: the repo (Pull requests, Actions, Releases), W&B project `opioid-risk-classification` (Artifacts > `opioid_classification_model`), the live `/docs`, DagsHub
- Have a small PR ready on `feature/*` into `dev` (for example a README typo) with CI already green
- Know the two tags you will use: the current release (for example `v3.0.1`) and the one to roll back to (`v3.0.0`)
- Fallback screenshots for every step are in `docs/demo/` (name them `01-pr.png` ... `08-rollback.png`)

## Script

| Min | Show | Say |
|---|---|---|
| 0-1 | Pull request into `dev`, checks `lint` and `test-and-build` green | "Nothing merges without the same checks: lint, the real pipeline with its quality gate, 90 % branch coverage, a Docker build." Open the run summary: the gate line `pr_auc=0.61 >= 0.40` |
| 1-3 | W&B > `opioid_classification_model` > versions | "Every version here passed the gate; a failed gate saves nothing." Open the newest version: metrics and `lineage.git_commit`, `lineage.raw_data_md5`. Move the alias `prod` to it. "Moving the label changes nothing in production yet." |
| 3-4 | GitHub > Releases > Draft > tag `v3.0.1` on `main` > Publish | "A person publishes. From here on, no human touches production." |
| 4-7 | Actions > "CD - Deploy to Render" run | Walk the jobs: **guard** (tag is on `main`), **ci** (full CI again on the tag), **publish-image** (`fetch_prod_model.py` resolves `prod` to `opioid_classification_model:vN`; build; the container passes the smoke test **before** push; tags `3.0.1`, `3.0`, `latest`, `sha-...`), **deploy** (Render pulls `:3.0.1`, live smoke test waits for the new commit) |
| 7-8 | Live `$RENDER_URL/health`, then `/docs` > POST `/predict` with `scripts/sample_request.json` | "`git_commit` and `model_version` prove what is running." |
| 8-9 | Trace back | `git_commit` -> the commit on GitHub; `model_version` -> the W&B version -> its run -> `lineage.raw_data_md5` = the md5 in `data/raw/opioid_raw_data.csv.dvc` -> the file on DagsHub |
| 9-10 | Actions > "CD - Deploy to Render" > Run workflow > tag `v3.0.0` | "Rollback redeploys the old image: code and model go back together. Then I move `prod` back in W&B so the next release does not re-ship the bad model." Show `/health` with the old commit and model version |

## If something fails

- Render asleep or slow: show screenshot `docs/demo/07-health.png`, keep talking about the trace, retry `/health` at the end
- Deploy run still running at minute 7: open the previous green run of the same workflow and walk through it
- W&B unreachable: show the version page screenshot; the release image already carries the model, the live API does not depend on W&B
- Rollback takes too long: show the previous successful manual run and the rollback log in README section 9

## Questions to ask the class

1. Which two decisions did a person make, and which steps did automation make?
2. Why is the smoke test run on the container **before** the image is pushed?
3. After a rollback, why do we also move `prod` back?
