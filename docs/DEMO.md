# Class demo: unweighted baseline to balanced enhancement

Goal: show how a measured model improvement becomes a tested release, then demonstrate rollback and roll forward without retraining. A person chooses the approved W&B model with prod and starts a release; automation builds, checks, publishes and deploys the selected package.

## The two models

| Role | Training setting | W&B run label | Example application release |
| --- | --- | --- | --- |
| Starting baseline | `class_weight: null` without quotes | `01-calibrated-logistic-regression-unweighted`, tag baseline | `v3.1.0` |
| Enhancement candidate | `class_weight: "balanced"` | `02-calibrated-logistic-regression-balanced`, tag candidate | `v3.1.1` |

Use an unused release pair. W&B model versions are independent of these application tags; use the actual version attached to each run. Keep both tested images for rollback.

## Before class

- Complete the instructor or student end-to-end guide through both training runs; do not retrain just for this short demonstration
- Start with the unweighted baseline image live and both runs available in W&B
- Have the balanced candidate's feature → dev → main PR checks complete before publishing its release
- Open the GitHub PR, Actions and Releases pages, W&B project, Render service, live /health and /docs
- Confirm the new service's hook and URL are configured, and the published baseline image remains available
- If using fallback screenshots, prepare them as described in docs/demo/README.md; this is optional

## Script

| Min | Show | Explain |
| --- | --- | --- |
| 0-1 | Baseline run's Config and live /health | The first model uses unweighted logistic regression, `class_weight: null`; the running image contains that specific approved model |
| 1-3 | First and candidate runs > Overview > Summary | Compare `metrics/val_after_calibration/pr_auc`, `roc_auc` and `ece`; balanced weighting is the candidate enhancement. Both passing the 0.40 gate is necessary, but approval also needs a favourable validation comparison. Test metrics are the final audit |
| 3-4 | Balanced run's Artifacts tab and its model version | Move prod to the measured, gated improvement. Refresh /health: approval alone has not changed the running baseline image |
| 4-5 | GitHub > Releases > Draft a new release > unused tag `v3.1.1` on main > Publish | Publishing starts the automation after the code has passed the PR and main checks |
| 5-7 | CD - Deploy to Render | Guard checks the release commit; CI checks it; publish-image downloads prod to a concrete model file, builds the image, smoke-tests it before GHCR upload; deploy requests that exact image and smoke-tests the live service |
| 7-8 | Live /health and /docs > POST /predict | The code commit and concrete model version identify the deployed balanced enhancement. Follow its W&B artifact to the training run, configuration and raw-data fingerprint |
| 8-9 | Actions > CD - Deploy to Render > Run workflow > branch main > tag `v3.1.0` | Briefly restore the baseline image to demonstrate rollback of code and model together; no training or build runs. Confirm /health and align prod with that model |
| 9-10 | Run workflow again with branch main and tag `v3.1.1` | Restore prod to the balanced model and roll forward to its existing image. Confirm /health; finish with the balanced enhancement live and approved |

## Interpret the comparison honestly

In the fixed reference comparison, the unweighted baseline had validation PR AUC 0.6078, ROC AUC 0.8335 and calibration error 0.0912. Balanced weighting gave 0.6096, 0.8446 and 0.0834 respectively. The PR AUC gain is small. Show the actual run values, not an exaggerated claim of a universal improvement. If the candidate does not improve the validation comparison, retain the baseline and stop promotion rather than lowering the gate or choosing from test results.

Rollback is a recovery capability demonstration; it does not mean the balanced model has failed. Finish by restoring the measured improvement. If W&B is unavailable, the already-published image can still answer predictions, although monitoring uploads may fail.

## Questions to ask the class

1. What changed between the unweighted baseline and balanced candidate?
2. Why does passing the minimum gate alone not prove an improvement?
3. Why does moving prod leave the live model unchanged until deployment?
4. Which file joins the model and code inside the image, and why test that image before publishing it?
5. Why can rollback and roll forward reuse existing images without training, and why align prod afterwards?
