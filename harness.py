#!/usr/bin/env python

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import lox

from dump import dump
from sidekick_client import FlowType, SidekickClient, TaskRequest, WorkspaceRequest
import time
from utils import (
    get_lite_dataset,
    # get_devin_instance_ids,
    # get_full_dataset,
    get_plausible,
    load_predictions,
    pick_winner,
)

REPOS_DNAME = Path("repos")
CHAT_LOGS_DNAME = Path("chat-logs")
PREDS_DNAME = Path("predictions")


def diff_versus_commit(git_dname, commit):
    """
    Take a diff of `git_dname` current contents versus the `commit`.
    """

    diff_cmd = f"git -C {git_dname} diff {commit}"
    diff_output = subprocess.check_output(diff_cmd.split()).decode()
    return diff_output


def files_in_patch(patch):
    """
    Extract the list of modified files from a unified diff patch string.
    """
    files = []
    for line in patch.split("\n"):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            fname = line.split("/", 1)[1]
            if fname not in files:
                files.append(fname)
    return files


def checkout_repo(entry, dname=None):
    """
    Clone the SWE Bench entry's git `repo` into `dname` at the `base_commit`.
    Make a tempdir if no `dname` provided.
    """
    github_url = "https://github.com/"
    repo_url = github_url + entry["repo"]
    commit = entry["base_commit"]

    print(repo_url, commit)

    git_tempdir = checkout_repo_url_commit(repo_url, commit, dname)

    return git_tempdir


def checkout_repo_url_commit(url, commit, dname):
    """
    Clone the git `url` into `dname` at `commit`.
    Check a local cache of the bare repo to avoid pulling from github every time.
    """

    # Extract repo name from URL
    repo_name = url.split("/")[-1].split(".")[0]
    repo_name += ".git"

    # dump(repo_name)
    REPOS_DNAME.mkdir(exist_ok=True, parents=True)
    bare_repo = REPOS_DNAME / repo_name

    if not bare_repo.exists():
        cmd = f"git clone --bare {url} {bare_repo}"
        subprocess.run(cmd.split(), check=True)

    if dname:
        Path(dname).mkdir()
        repo_dname = dname
    else:
        repo_dname = tempfile.TemporaryDirectory().name

    cmd = f"git clone {bare_repo} {repo_dname}"
    subprocess.run(cmd.split(), check=True)

    cmd = f"git -c advice.detachedHead=false -C {repo_dname} checkout {commit}"
    subprocess.run(cmd.split(), check=True)

    # IGNORE = '*test*\n'
    # ignore = Path(repo_dname) / '.sidekickignore'
    # ignore.write_text(IGNORE)

    return repo_dname


def show_problems(dataset):
    """
    Print out all the instance_id and problem_descriptions.
    """
    for inst, entry in dataset.items():
        problem = entry["problem_statement"].splitlines()[0]
        print(f"{inst}: {problem}")

def configure_sidekick(git_tempdir, entry):
    # configure sidekick via genflow.coding.toml file + .genflowignore file + setup run_dev_tests.sh script
    # then commit that stuff too
    #
    # Test script contains :
    #
    #     cd {current_dir} # for pdm to work with the existing virtualenv
    #     SWEBENCH_DOCKER_FORK_DIR={current_dir}/SWE-bench-docker pdm run {current_dir}/tests.py run_dev_tests {entry["instance_id"]} {git_tempdir}
    #     RETURN=$?
    #     rm {git_tempdir}/princeton-nlp--SWE-bench*.json 
    #     exit $RETURN
    # 
    # .genflowignore contains:
    #
    #     run_dev_tests.sh
    #     genflow.coding.toml
    #     .genflowignore
    #
    # genflow.coding.toml contains:
    #
    #     disable_human_in_the_loop = true
    #
    #     [[test_commands]]
    #     command = "/usr/bin/env sh run_dev_tests.sh"
    #


    # Create run_dev_tests.sh file
    current_dir = Path(__file__).parent
    run_dev_tests_sh = f"""
cd {current_dir} # for pdm to work with the existing virtualenv
SWEBENCH_DOCKER_FORK_DIR={current_dir}/SWE-bench-docker pdm run {current_dir}/tests.py run_dev_tests {entry["instance_id"]} {git_tempdir}
RETURN=$?
rm {git_tempdir}/princeton-nlp--SWE-bench*.json 
exit $RETURN
    """
    run_dev_tests_sh_path = Path(git_tempdir) / "run_dev_tests.sh"
    run_dev_tests_sh_path.write_text(run_dev_tests_sh)

    # Configure Sidekick via genflow.coding.toml file
    sidekick_config = """
disable_human_in_the_loop = true

[[test_commands]]
command = "/usr/bin/env sh run_dev_tests.sh"
    """
    genflow_coding_toml_path = Path(git_tempdir) / "genflow.coding.toml"
    genflow_coding_toml_path.write_text(sidekick_config)

    # Create .genflowignore file that just ignores the above two files and itself
    genflowignore = """
run_dev_tests.sh
genflow.coding.toml
.genflowignore
    """
    genflowignore_path = Path(git_tempdir) / ".genflowignore"
    genflowignore_path.write_text(genflowignore)

    # Commit the genflow.coding.toml, .genflowignore and run_dev_tests.sh files
    commit_cmd = f"git -C {git_tempdir} add genflow.coding.toml .genflowignore run_dev_tests.sh"
    subprocess.run(commit_cmd.split(), check=True)
    commit_cmd = f"git -C {git_tempdir} commit -m 'configure_sidekick'"
    subprocess.run(commit_cmd.split(), check=True)

def process_one_instance(entry, num_tries, models, temperature, model_name_or_path, out_dname):
    """Process one `entry` from SWE Bench using the LLM `models` at the
    given `temperature`.  Set `model_name_or_path` in the result json.
    Store the result json and the chat log into `out_dname`.
    """

    instance_id = entry["instance_id"]
    base_commit = entry["base_commit"]

    print("=" * 60)
    dump(instance_id)
    print("=" * 60)
    problem_statement = entry["problem_statement"]
    print(problem_statement)
    gold_files = files_in_patch(entry["patch"])

    client = SidekickClient()

    results = []
    cost = 0
    winner = None

    # Do NUM_TRIES tries for each of the models, until we find a *plausible* solution
    for attempt in range(1, num_tries + 1):
        for model in models:
            dump(attempt, model)

            git_tempdir = checkout_repo(entry)
            dump(git_tempdir)

            dump(instance_id)
            dump(gold_files)

            # Configure and tell Sidekick to work on the `problem_statement` by creating a task.
            # This is the same as if you pasted it into a new task within the Sidekick app.
            configure_sidekick(git_tempdir, entry)
            all_workspaces = client.get_workspaces()

            # TODO set up only one git_tempdir and one workspace per repo
            # (instead of per instance_id), by relying on sidekick to use its
            # local worktree environment to edit the code
            workspace = next((w for w in all_workspaces if w.name == instance_id), None)
            upsertRequestData = WorkspaceRequest(
                name=instance_id,
                local_repo_dir=git_tempdir,
            )
            if workspace:
                workspace = client.update_workspace(workspace.id, upsertRequestData)
            else:
                workspace = client.create_workspace(upsertRequestData)

            if workspace.name == instance_id:
                # mark any existing tasks that are "to_do", "in_progress" or
                # "blocked" in the instance-specific workspace as failed
                existing_tasks = client.get_tasks(workspace.id, statuses="to_do,in_progress,blocked")
                for task in existing_tasks:
                    workspace.update_task(task.id, TaskRequest(
                        description=task.description,
                        flow_type=task.flow_type,
                        agent_type="none",
                        status="failed",
                    ))

            task = workspace.create_task(TaskRequest(
                description=f"""
A user has reported the following bug/issue in the {entry["repo"]} project,
which you are tasked with resolving. Before you try to fix the code, you must
add a test that reproduces the issue. Ensure this test fails with the expected
failure message. This test should pass after your fix is done. If requirements
are unclear, you must consider several alternatives, think through their pros
and cons, and then make an informed decision, which should ideally be recorded
into your plan. Choose the solution that is most likely to cover the most edge
cases and align best with developer expectations given the
project/language/framework.

The bug/issue report follows:

{problem_statement}""",
                flow_type=FlowType.PLANNED_DEV,
            ))

            # Wait for the task to be completed by polling the Sidekick API
            sleep_interval = 5  # seconds
            time_limit = 900 # seconds - set high due to slow tests via docker on mac
            start_time = time.time()
            while True:
                task = workspace.get_task(task.id)
                if task.status == "completed" or task.status == "failed":
                    print(f"Task finished with status: {task.status}")
                    break
                if time.time() - start_time >= time_limit:
                    print(f"Task timed out with status: {task.status}")
                    break
                time.sleep(sleep_interval)

            # try:
            #     # TODO
            #     1 / 0
            # except Exception as coder_err:
            #     # swallow any exceptions during benchmarking
            #     dump(coder_err)
            #     continue

            dump(instance_id)
            dump(gold_files)

            # TODO: Keep track of API costs
            # cost += coder.total_cost

            # Get the diff between the current state and the original commit
            # Note: we actually use the next commit after the base_commit as the basis for the diff,
            # because the test harness makes an extra git commit before sidekick starts
            command = f"""git-C {git_tempdir} rev-list --topo-order {base_commit}.."$*" | tail -1"""
            next_commit = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
            model_patch = diff_versus_commit(git_tempdir, next_commit)
            dump(model_patch)

            # Record the results for the logs
            result = dict(
                # Required args for running eval tests
                instance_id=instance_id,
                model_name_or_path=model_name_or_path,
                model_patch=model_patch,
                # For computing stats
                #model=model,
                #temperature=temperature,
                #cost=coder.total_cost,
                #added_files=added_files,
                #gold_files=gold_files,
                #edited_files=files_in_patch(model_patch),
                #edit_outcome=coder.edit_outcome,
                #lint_outcome=coder.lint_outcome,
                #test_outcome=coder.test_outcome,
            )
            result["try"] = attempt  # `try` is a python keyword
            results.append(result)

            dump(result)

            # Did we get a successful edit, lint and test? If so, we found a plausible solution!
            if model_patch: # and coder.edit_outcome and coder.lint_outcome and coder.test_outcome:
                winner = result
                break

        # also break out of the attempts loop
        if winner:
            break

    # If there's no clear winner, look for the most viable result we got...
    if not winner:
        winner = pick_winner(results)

    if not winner:
        result = dict(
            # Required args for running eval tests
            instance_id=instance_id,
            model_name_or_path=model_name_or_path,
            model_patch=None,
        )

    dump(winner)
    if not winner:
        return

    print("\n\nFinal diff:\n")
    print(winner["model_patch"])

    # Avoid circular reference when we save to json
    winner = dict(winner)

    winner.update(
        dict(
            tries=attempt,
            all_results=results,  # Record all the results for later analysis
            cost=cost,  # total cost across all results
        )
    )

    out_fname = out_dname / (instance_id + ".json")
    out_fname.write_text(json.dumps(winner, indent=4))


def main():
    #
    # Set the prefix to use in front of the predictions/ subdir name.
    #
    prefix = "full-"
    # prefix = "full025-"

    #
    # Configure 1 or more models to use to try and find plausible solutions
    #
    # models = ["openrouter/deepseek/deepseek-chat"]
    # models = ["gpt-4o", "openrouter/anthropic/claude-3-opus"]
    # models = ["openrouter/anthropic/claude-3-opus"]
    models = ["gpt-4o"]
    # models = ["gpt-4-1106-preview"]

    # How many attempts per model to try and find a plausible solutions?
    num_tries = 1

    # What temperature to use during chat completions
    temperature = 0

    # Load the SWE Bench dataset
    # dataset = get_full_dataset()
    dataset = get_lite_dataset()

    just_devin_570 = False
    # Filter it to the Devin 570 when using the full dataset
    # just_devin_570 = True
    # if just_devin_570:
    #     # Filter it to the Devin 570
    #     devin_insts = get_devin_instance_ids()
    #     dataset = dict((inst, entry) for inst, entry in dataset.items() if inst in devin_insts)

    # How many threads to use for attempting instances in parallel
    threads = 1

    # Any predictions/ dirs provided on the command line are treated
    # as earlier, higher priority runs.  If a plausible solution was
    # found for an instance already, we don't need to keep looking in
    # this run.
    prior_dnames = sys.argv[1:]

    process_instances(
        prefix, dataset, models, num_tries, temperature, threads, prior_dnames, just_devin_570
    )


def process_instances(
    prefix, dataset, models, num_tries, temperature, threads, prior_dnames, just_devin_570
):
    """
    prefix - Prefix used in front of the dirname in predictions/.
    dataset - The subset of the SWE Bench dataset to process.
    models - List of models to use to try and find plausible solutions.
    num_tries - Number of attempts to make using each model.
    temperature - Temp to use during chat completions.
    threads - How many problems to attempt concurrently.
    prior_dnames - Names of predictions/ dirnames from previous runs.
                   If they contain a plausible solution for an instance,
                   don't continue looking.
    """
    models_slug = "--".join(model.replace("/", "-") for model in models)
    model_name_or_path = "sidekick--" + models_slug
    models_slug = prefix + "--" + models_slug

    dump(models)
    dump(temperature)

    out_dname = PREDS_DNAME / models_slug
    if not out_dname.exists():
        out_dname.mkdir(parents=True)

    dump(out_dname)

    # If we are restarting this run, figure out which instances are already done.
    done_preds = load_predictions([out_dname], just_devin_570)
    done_instances = set(done_preds.keys())
    dump(len(done_instances))

    dump(prior_dnames)
    prior_preds = load_predictions(prior_dnames, just_devin_570)
    dump(len(prior_preds))

    plausible_instances = get_plausible(prior_preds)
    dump(len(plausible_instances))

    if prior_preds:
        # Just keep trying to solve instances that exist in the previous runs
        all_instances = set(prior_preds.keys())
    else:
        all_instances = set(dataset.keys())

    remaining_instances = set(all_instances)
    remaining_instances -= done_instances
    remaining_instances -= plausible_instances

    remaining_instances = list(remaining_instances)
    random.shuffle(remaining_instances)

    dump(len(remaining_instances))
    dump(sorted(remaining_instances))

    print()
    print("press enter...")
    input()

    chat_history_dname = CHAT_LOGS_DNAME / models_slug
    chat_history_dname.mkdir(exist_ok=True, parents=True)

    threads = 1
    if threads > 1:
        process_one_instance_lox = lox.thread(threads)(process_one_instance)
        process_one_instance_func = process_one_instance_lox.scatter
        gather = process_one_instance_lox.gather
    else:
        process_one_instance_func = process_one_instance

    for instance_id in remaining_instances:
        entry = dataset[instance_id]

        # Only process instances from the mwaskom/seaborn repo for now
        # FIXME revert later on after testing seaborn is complete
        if entry["repo"] != "mwaskom/seaborn":
            continue
        if instance_id != "mwaskom__seaborn-3407":
            continue

        process_one_instance_func(
            dataset[instance_id],
            num_tries,
            models,
            temperature,
            model_name_or_path,
            out_dname,
        )

        print("#" * 60)
        # input()

    if threads > 1:
        gather()


if __name__ == "__main__":
    status = main()
    sys.exit(status)
