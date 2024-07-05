# Sidekick SWE Bench harness

## Installation

```sh
# Clone this repo
git clone https://github.com/genflow-dev/sidekick-swe-bench

# Clone the SWE Bench docker repo into a subdir of this repo
cd sidekick-swe-bench
git clone https://github.com/aorwall/SWE-bench-docker

# Install python packages
pdm install
```

See the
[SWE Bench Docker docs](https://github.com/aorwall/SWE-bench-docker)
to ensure you have built or pulled all the SWE Bench testbed
docker images you'll need.

## Running the benchmark and computing results

The workflow for working with SWE Bench in general is 2 steps:

1. Run your agent on the problems to produce predictions, which are a series of json records that get bundled up into a jsonl file.
2. Evaluate the predictions jsonl file using the acceptance tests. This produces `.eval.log` files with logs of the testing procedure.

This repo is for running and evaluating Sidekick on SWE Bench. As described in the README, it consists of 2 scripts:

1. The `harness.py` script will run Sidekick on all the problems and produce predictions. It does not do any *acceptance* testing. It does run any pre-existing tests that were part of the problem's repo, but never runs any acceptance tests. This script produces a bunch of predictions as individual json files in `predictions/<DIRNAME>/<instance_id>.json`.

Run it like so:

```sh
pdm run harness.py 
```

2. The `report.py` script consumes all those predictions and turns them into `predictions/<DIRNAME>/all_preds.jsonl`. It then feeds that jsonl file through the SWE Bench evaluation and reporting scripts to produce `logs/<DIRNAME>/<instance_id>...eval.log` files as well as a summary report in `predictions/<DIRNAME>/results.json`.
