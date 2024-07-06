#!/usr/bin/env python

import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from dump import dump
from harness import diff_versus_commit
from swebench_docker.constants import MAP_VERSION_TO_INSTALL
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_directives
import re
from utils import get_dataset, get_devin_instance_ids, get_lite_dataset, load_predictions  # noqa: F401
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import json

PY_LANGUAGE = Language(tspython.language())


# clipped from `run_docker_evaluation()`
def get_docker_image(task_instance: dict, namespace: str = "aorwall"):
    repo_name = task_instance["repo"].replace("/", "_")

    specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][task_instance["version"]]
    image_prefix = "swe-bench"

    if specifications.get("instance_image", False):
        docker_image = (
            f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
        )
    else:
        docker_image = f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"

    return docker_image


# A no-op patch which creates an empty file is used to stand in for
# the `model_patch` and/or `test_patch` when running SWE Bench tests
# without one or both of those patches.
NOOP_PATCH = (
    "diff --git a/empty.file.{nonce}.ignore b/empty.file.{nonce}.ignore\n"
    "new file mode 100644\n"
    "index 0000000..e69de29\n"
)


def remove_patches_to_tests(model_patch):
    """
    Remove any changes to the tests directory from the provided patch.
    This is to ensure that the model_patch does not disturb the repo's
    tests when doing acceptance testing with the `test_patch`.
    """
    lines = model_patch.splitlines(keepends=True)
    filtered_lines = []
    is_tests = False

    for line in lines:
        if line.startswith("diff --git a/"):
            pieces = line.split()
            to = pieces[-1]
            if to.startswith("b/") and (
                "/test/" in to
                or "/tests/" in to
                or "/testing/" in to
                or "/test_" in to
                or "/tox.ini" in to
            ):
                is_tests = True
            else:
                is_tests = False

        if not is_tests:
            filtered_lines.append(line)

    return "".join(filtered_lines)


TEST_PYTEST = "pytest --no-header -rfE --disable-warnings --tb=no -p no:cacheprovider"
TEST_PYTEST_SKIP_NO_HEADER = "pytest -rfE --disable-warnings --tb=no -p no:cacheprovider"
CUSTOM_MAP_REPO_TO_TEST_FRAMEWORK = {
    "astropy/astropy": TEST_PYTEST,
    "django/django": "./tests/runtests.py --verbosity 2",
    "marshmallow-code/marshmallow": TEST_PYTEST,
    "matplotlib/matplotlib": TEST_PYTEST,
    "mwaskom/seaborn": "pytest --no-header -rfE --disable-warnings",
    "pallets/flask": TEST_PYTEST,
    "psf/requests": TEST_PYTEST,
    "pvlib/pvlib-python": TEST_PYTEST,
    "pydata/xarray": TEST_PYTEST,
    "pydicom/pydicom": TEST_PYTEST_SKIP_NO_HEADER,
    "pylint-dev/astroid": TEST_PYTEST,
    "pylint-dev/pylint": TEST_PYTEST,
    "pytest-dev/pytest": "pytest -rfE --disable-warnings",
    "pyvista/pyvista": TEST_PYTEST,
    "scikit-learn/scikit-learn": TEST_PYTEST_SKIP_NO_HEADER,
    "sphinx-doc/sphinx": "tox -epy39 -v --",
    "sqlfluff/sqlfluff": TEST_PYTEST,
    "swe-bench/humaneval": "python",
    "sympy/sympy": "bin/test -C --verbose",
}

def run_tests(entry, model_patch=None, use_test_patch=False, model_name_or_path="none", test_directives=None):
    """
    Run tests for the SWE Bench `entry`, optionally applying a `model_patch` first.

    If `use_test_patch` is True, then also apply the `test_patch` to bring in
    the tests which determine if the issue is resolved. So False means
    only run the tests that existed at the `base_commit` and any new/changed
    tests contained in the `model_patch`.

    Optionally specify a `model_name_or_path`, which isn't really used since
    the log_dir for the tests is a temp dir which is discarded.
    """
    instance_id = entry["instance_id"]
    # print(f"Running tests for {instance_id}...")

    test_type = CUSTOM_MAP_REPO_TO_TEST_FRAMEWORK[entry["repo"]]
    if test_directives is None:
        test_directives = get_test_directives(entry)
    test_cmd = f"{test_type} {' '.join(test_directives)}"
    # print(f"Test command: {test_cmd}")

    # Use a no-op patch if no model_patch is provided
    if not model_patch:
        model_patch = NOOP_PATCH.format(nonce="model_patch")

    # Use a no-op patch if use_test_patch is False
    if use_test_patch:
        test_patch = entry["test_patch"]
    else:
        test_patch = NOOP_PATCH.format(nonce="test_patch")

    if model_patch and use_test_patch:
        # Make sure the model_patch does not disturb the repo's tests
        # when doing acceptance testing with the `test_patch`.
        print("=" * 30)
        print(model_patch)
        model_patch = remove_patches_to_tests(model_patch)
        print("=" * 30)
        print(model_patch)
        print("=" * 30)

    entry_instance = {
        "repo": entry["repo"],
        "version": entry["version"],
        "base_commit": entry["base_commit"],
        "instance_id": entry["instance_id"],
        "model_name_or_path": model_name_or_path,
        "model_patch": model_patch,
        "test_patch": test_patch,
        "test_directives": test_directives,
        "test_cmd": test_cmd,
    }

    namespace = "aorwall"
    log_dir = tempfile.TemporaryDirectory(dir="/tmp").name
    timeout = 200
    log_suffix = ""

    asyncio.run(run_docker_evaluation(entry_instance, namespace, log_dir, timeout, log_suffix))

    log_fname = Path(log_dir) / f"{instance_id}.{model_name_or_path}.eval.log"
    if not log_fname.exists():
        return None, ""

    log_text = log_fname.read_text()
    log_lines = log_text.splitlines()
    log_lines = [line for line in log_lines if line.startswith(">>>>")]
    #print("\n".join(log_lines))

    passed = ">>>>> All Tests Passed" in log_text

    return passed, log_text

def run_dev_tests(entry, git_dname):
    # during development, we don't have a test patch, so we just run existing
    # tests or tests added in the model patch
    return diff_and_run_tests(entry, git_dname, use_test_patch=False)

def run_eval_tests(entry, git_dname):
    # like run_dev_tests, but now we have the test patch used for evaluation.
    # also, we skip evaluating any additional tests added in the model patch,
    # in case of git patch apply conflicts and for a cleaner evaluation.
    return diff_and_run_tests(entry, git_dname, use_test_patch=True)

def diff_and_run_tests(entry, git_dname, use_test_patch=False):
    """Given the current contents of the `git_dname`, run the tests that were
    present in the entry's `repo` at the time of the `base_commit` and are
    specified as required to pass (i.e. PASS_TO_PASS).
    
    If use_test_patch is True, we also evaluate new tests which have been added
    into the repo since during development.

    If use_test_patch is False, we do NOT attempt to run the tests in the
    `test_patch`, which are used to evaluate whether the `model_patch` has
    resolved the `problem_statement` (i.e. FAIL_TO_PASS).

    Returns tuple of whether tests passed and output.
    """

    # get the next commit after the base_commit as the basis for the diff,
    # because the test harness makes an extra git commit before sidekick starts.
    command = f"""git -C {git_dname} rev-list --topo-order {entry["base_commit"]}.."$*" | tail -1"""
    next_commit = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
    model_patch = diff_versus_commit(git_dname, next_commit)

    # switch working directory to this directory
    current_dir = Path(__file__).parent
    os.chdir(current_dir)

    #print(f"Base commit: {entry['base_commit']}")
    #print(f"Next commit after base_commit: {next_commit}")
    #print("Model Patch:")
    #print(model_patch)

    existing_test_directives = json.loads(entry["PASS_TO_PASS"])
    if use_test_patch:
        additional_test_directives = json.loads(entry["FAIL_TO_PASS"])
    else: 
        newly_added_test_directives = extract_added_test_directives_from_patch(entry, model_patch, git_dname)
        additional_test_directives = newly_added_test_directives

        # the following test directives are actually not present in the repo but in
        # the test patch. this appears to be an error in the swe-bench dataset. we
        # need to filter them out as trying to run non-existent tests will cause the
        # test runner to fail.
        not_actually_existing_test_directives = extract_added_test_directives_from_patch(entry, entry["test_patch"], None)
        existing_test_directives = [d for d in existing_test_directives if d not in not_actually_existing_test_directives]

    # print(f"Existing Test directives: {existing_test_directives}, Newly Added Test directives: {newly_added_test_directives}")
    test_directives = existing_test_directives + additional_test_directives

    passed, output = run_tests(
        entry,
        model_patch=model_patch,
        use_test_patch=use_test_patch,
        test_directives=test_directives,
    )

    # We were UNABLE to run tests
    if passed is None:
        return passed, output

    # Just keep the actual output from the tests that were run, throwing away the harness output
    if "Std. Output:" not in output:
        raise ValueError("Output does not contain the expected 'Std. Output:' string, did the SWE-bench-docker test runner change?")
    output = output.split("Std. Output:")[-1]

    return passed, output

def extract_added_test_directives_from_patch(entry, git_patch, git_dname):
    """Given a `git_patch`, extract the test directives for any new tests
    that have been added to the repo since the `base_commit`.
    """
    if git_dname == None:
        # we can't extract test directives using tree-sitter without the git
        # directory, but we do the hacky way using just the patch instead
        return extract_added_test_directives_from_patch_old(entry, git_patch)

    test_directives = []

    file_rows = extract_file_rows_from_patch(git_patch)
    print("Extracted File rows:")
    print(file_rows)

    print("Extracted Test directives:")
    print(test_directives)
    exit(1)

def extract_file_rows_from_patch(git_patch):
    file_rows = {}

    # Regular expressions for matching file headers and chunk headers
    file_pattern = re.compile(r'^diff --git a/.* b/(.*)$')
    chunk_pattern = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@')

    current_file = None
    current_line = 0
    last_added_line = {}
    last_removed_line = {}
    last_added_line_number = {}
    last_removed_line_number = {}

    # Process each line of the git diff
    for line in git_patch.split('\n'):
        # Check if this line is a file header
        file_match = file_pattern.match(line)
        if file_match:
            current_file = file_match.group(1)
            file_rows[current_file] = []
            continue

        # Check if this line is a chunk header
        chunk_match = chunk_pattern.match(line)
        if chunk_match:
            current_line = int(chunk_match.group(1)) - 1
            continue

        # Process added or removed lines
        if line.startswith('+') and not line.startswith('+++'):
            file_rows[current_file].append(current_line)
            last_added_line[current_file] = line[1:]
            last_added_line_number[current_file] = current_line
            current_line += 1
        elif line.startswith('-') and not line.startswith('---'):
            file_rows[current_file].append(current_line)
            last_removed_line[current_file] = line[1:]
            last_removed_line_number[current_file] = current_line
        elif line.startswith(' '):
            current_line += 1

    # Remove newline-only changes at the end of files
    for file in file_rows:
        if file in last_added_line and file in last_removed_line and last_added_line[file] == last_removed_line[file] and last_removed_line_number[file] == last_added_line_number[file]:
            file_rows[file] = file_rows[file][:-2]

    return file_rows

def extract_added_test_directives_from_patch_old(entry, git_patch):
    """Given a `git_patch`, extract the test directives for any new tests
    that have been added to the repo since the `base_commit`.
    """
    # extract each diff hunk
    hunks = git_patch.split("\ndiff --git")

    # for each hunk, extract the file path and the tests added
    test_directives = []

    for hunk in hunks:
        if len(hunk) == 0:
            continue
        lines = hunk.split("\n")
        file_path = re.search(r' a/(.*?) b/', lines[0]).group(1)
        class_name = None
        for line in lines:
            # NOTE: a class name may be in the hunk header, or in the diff itself for a newly added class
            if re.search(r'^(@@.*@@|\+).*class\s*(\w+)', line):
                class_name = re.search(r'class\s*(\w+)', line).group(1).strip()

            # filter to lines that start with '+' then any whitespace, then 'def test'
            if re.search(r'^\+\s*def\s+test', line):
                test_name = re.search(r'^\+\s*def\s+(test(\w*))\(', line).group(1).strip()
                if re.search(r"pytest|tox", CUSTOM_MAP_REPO_TO_TEST_FRAMEWORK[entry["repo"]]):
                    if class_name is not None:
                        test_directive = f"{file_path}::{class_name}::{test_name}"
                    else:
                        test_directive = f"{file_path}::{test_name}"
                else:
                    # FIXME make this work for django and all other non-pytest/non-tox frameworks
                    test_directive = test_name
                test_directives.append(test_directive)

    print("Extracted Test directives (old):")
    print(test_directives)
    exit(1)

    return test_directives


def check_docker_images():
    dataset = get_dataset()

    # instances = get_devin_instance_ids()
    instances = list(dataset.keys())
    random.shuffle(instances)

    cache_fname = Path("tmp.dockerimages.json")
    if cache_fname.exists():
        data = json.loads(cache_fname.read_text())
        good_dockers = defaultdict(int, data["good"])
        bad_dockers = defaultdict(int, data["bad"])
        seen_instances = set(data["instances"])
    else:
        good_dockers = defaultdict(int)
        bad_dockers = defaultdict(int)
        seen_instances = set()

    for instance_id in instances:
        entry = dataset[instance_id]

        if instance_id in seen_instances:
            continue

        seen_instances.add(instance_id)

        docker_image = get_docker_image(entry)
        if docker_image in bad_dockers:
            bad_dockers[docker_image] += 1
            continue

        if docker_image in good_dockers:
            good_dockers[docker_image] += 1
            continue

        dump(instance_id)
        dump(docker_image)

        passed, test_text = run_tests(
            entry,
            model_patch=None,
            use_test_patch=False,
        )
        if passed is None:
            bad_dockers[docker_image] += 1
        else:
            good_dockers[docker_image] += 1

        update_cache(cache_fname, seen_instances, good_dockers, bad_dockers)

    update_cache(cache_fname, seen_instances, good_dockers, bad_dockers)

    dump(bad_dockers)


def update_cache(cache_fname, instances, good_dockers, bad_dockers):
    save_dict = dict(
        instances=list(instances),
        good=dict(good_dockers),
        bad=dict(bad_dockers),
    )
    cache_fname.write_text(json.dumps(save_dict, indent=4, sort_keys=True))

    total_instances = sum(good_dockers.values()) + sum(bad_dockers.values())
    dump(total_instances)
    bad_instances = sum(bad_dockers.values())
    dump(bad_instances)
    if total_instances:
        pct_bad_instances = bad_instances / total_instances * 100
        dump(pct_bad_instances)
    dump(len(bad_dockers))


def preds():
    dataset = get_dataset()

    dnames = sys.argv[2:]
    preds = load_predictions(dnames)

    num = 0
    num_passed = 0
    for instance_id, pred in preds.items():
        entry = dataset[instance_id]

        passed, test_text = run_tests(
            entry,
            model_patch=pred["model_patch"],
            use_test_patch=True,
        )

        num += 1
        if passed:
            num_passed += 1

        dump(num_passed, num)


def main(argv):
    if len(argv) < 2:
        print("Usage: python tests.py [command]")
        return 1

    command = argv[1]
    if command == "check_docker_images":
        status = check_docker_images()

    elif command == "run_dev_tests" or command == "run_eval_tests":
        instance_id = sys.argv[2]
        repo_dir = sys.argv[3]

        dataset = get_lite_dataset()
        entry = dataset[instance_id]
        #print(entry)
        if command == "run_dev_tests":
            passed, output = run_dev_tests(entry, repo_dir)
        elif command == "run_eval_tests":
            passed, output = run_eval_tests(entry, repo_dir)
        else:
            raise ValueError(f"Invalid command: {command}")
        print(output)
        print(f"Tests Passed: {passed}")

        if passed:
            status = 0
        elif passed is None:
            status = 2
        elif not passed:
            status = 1

    elif command == "preds":
        status = preds()

    else:
        print("Invalid command")
        return 1

    return status

if __name__ == "__main__":
    status = main(sys.argv)
    sys.exit(status)
