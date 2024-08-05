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
import time
from typing import Dict, List, Optional
import requests
from dump import dump
from harness import diff_versus_commit
from swebench_docker.constants import MAP_VERSION_TO_INSTALL, TESTS_PASSED
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_directives
import re
from utils import (
    get_dataset,
    get_devin_instance_ids,
    get_lite_dataset,
    load_predictions,
)  # noqa: F401
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import json
import re

PY_LANGUAGE = Language(tspython.language())


# clipped from `run_docker_evaluation()`
def get_docker_image(task_instance: dict, namespace: str = "aorwall"):
    repo_name = task_instance["repo"].replace("/", "_")
    specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][
        task_instance["version"]
    ]
    image_prefix = "swe-bench"
    if specifications.get("instance_image", False):
        docker_image = f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
    else:
        docker_image = (
            f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"
        )
    return docker_image


# A no-op patch which creates an empty file is used to stand in for
# the `model_patch` and/or `test_patch` when running SWE Bench tests
# without one or both of those patches.
NOOP_PATCH = (
    "diff --git a/empty.file.{nonce}.ignore b/empty.file.{nonce}.ignore\n"
    "new file mode 100644\n"
    "index 0000000..e69de29\n"
)


class TestDirective:
    def __init__(
        self,
        file_path,
        method_name=None,
        class_name=None,
        module_based=False,
        docstring=None,
        method_only=False,
    ):
        self.file_path = file_path
        self.class_name = class_name
        self.method_name = method_name
        self.module_based = module_based
        self.method_only = method_only
        self.docstring = docstring

    def __str__(self):
        if self.module_based:
            file_segments = self.file_path.split("/")
            file_segments[-1] = file_segments[-1].replace(".py", "")
            file_segments = file_segments[1:]  # remove "tests" directory
            module_name = ".".join(file_segments)
            if self.class_name is not None:
                if self.method_name is not None:
                    return f"{module_name}.{self.class_name}.{self.method_name}"
                else:
                    return f"{module_name}.{self.class_name}"
            elif self.method_name is not None:
                return f"{module_name}.{self.method_name}"
            else:
                return module_name
        elif self.method_only:
            return self.method_name
        else:
            if self.class_name is not None:
                if self.method_name is not None:
                    return f"{self.file_path}::{self.class_name}::{self.method_name}"
                else:
                    return f"{self.file_path}::{self.class_name}"
            elif self.method_name is not None:
                return f"{self.file_path}::{self.method_name}"
            elif self.file_path is None and self.docstring is not None:
                return self.docstring
            else:
                return self.file_path

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)

    # module-based test directives are initially formatted like this: "test_empty_plan (migrations.test_executor.ExecutorTests)"
    @staticmethod
    def parse_module_in_parens(directive_with_module_in_parens):
        method_name, module_name = directive_with_module_in_parens.split(" (")
        method_name = method_name.strip()
        module_name = module_name[:-1]  # remove closing parens
        # infer class name based on case of last segment of module name
        class_name = module_name.split(".")[-1]
        if class_name[0].isupper():
            module_name = ".".join(module_name.split(".")[:-1])  # remove last
        else:
            class_name = None
        file_path = "tests/" + module_name.replace(".", "/") + ".py"
        return TestDirective(file_path, method_name, class_name, module_based=True)

    # a pytest test directive is formatted like this: "tests/migrations/test_executor.py::ExecutorTests::test_empty_plan"
    # there may or may not be a class name in there, we infer based number of "::" separators/segments
    @staticmethod
    def parse_pytest(pytest_directive):
        if "::" in pytest_directive:
            segments = pytest_directive.split("::")
            if len(segments) == 3:
                file_path, class_name, method_name = segments
                return TestDirective(file_path, method_name, class_name)
            elif len(segments) == 2:
                file_path, method_name = segments
                return TestDirective(file_path, method_name)
        else:
            file_path = pytest_directive
            return TestDirective(file_path)

    # parse a test directive from a string, infer the right format based on the string
    @staticmethod
    def parse(directive):
        if re.match(r"^test_.*\([^)]*\)$", directive):
            return TestDirective.parse_module_in_parens(directive)
        elif "::" in directive:
            return TestDirective.parse_pytest(directive)
        elif re.match(r"^test_\w+$", directive):
            return TestDirective(None, method_name=directive, method_only=True)
        else:
            return TestDirective(None, docstring=directive)


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


TEST_PYTEST = (
    "pytest --no-header -rfE --disable-warnings --tb=short -p no:cacheprovider"
)
TEST_PYTEST_SKIP_NO_HEADER = (
    "pytest -rfE --disable-warnings --tb=short -p no:cacheprovider"
)
CUSTOM_MAP_REPO_TO_TEST_FRAMEWORK = {
    "astropy/astropy": TEST_PYTEST,
    "django/django": "./tests/runtests.py --verbosity 0 --noinput --parallel=1",
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
    "sympy/sympy": "bin/test -C",
}


def run_tests_via_server(
    entry,
    model_patch=None,
    use_test_patch=False,
    model_name_or_path="none",
    test_directives=None,
    test_server_host=None,
):
    headers = {"Content-Type": "application/json"}
    body = {
        "entry": entry,
        "model_patch": model_patch,
        "use_test_patch": use_test_patch,
        "model_name_or_path": model_name_or_path,
        "test_directives": test_directives,
    }
    # print("Sending POST request...")
    url = f"http://{test_server_host}/run_tests"
    response = requests.post(url, json=body, headers=headers)
    if response.status_code == 200:
        try:
            response_data = response.json()
            passed = response_data.get("passed")
            log_text = response_data.get("log_text")
            # print(f"Passed: {passed}")
            # print(f"Log text: {log_text}")
            return passed, log_text
        except json.JSONDecodeError:
            message = f"Error: Running tests via server failed: Response is not valid JSON: {response.text}"
            return False, message
    else:
        message = f"Error: Running tests via server failed: Received status code {response.status_code}\n{response.text}"
        return False, message


def run_tests(
    entry,
    model_patch=None,
    use_test_patch=False,
    model_name_or_path="none",
    test_directives=None,
    test_server_host=None,
):
    """
    Run tests for the SWE Bench `entry`, optionally applying a `model_patch` first.
    If `use_test_patch` is True, then also apply the `test_patch` to bring in
    the tests which determine if the issue is resolved. So False means
    only run the tests that existed at the `base_commit` and any new/changed
    tests contained in the `model_patch`.
    Optionally specify a `model_name_or_path`, which isn't really used since
    the log_dir for the tests is a temp dir which is discarded.
    """
    if test_server_host is not None:
        return run_tests_via_server(
            entry,
            model_patch,
            use_test_patch,
            model_name_or_path,
            test_directives,
            test_server_host,
        )
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
    # log_dir = tempfile.TemporaryDirectory(dir="/tmp").name
    current_dir = Path(__file__).parent
    log_dir = f"{current_dir}/logs"
    timeout = 300  # TODO check how long the longest tests take
    log_suffix = ""
    asyncio.run(
        run_docker_evaluation(entry_instance, namespace, log_dir, timeout, log_suffix)
    )
    log_fname = Path(log_dir) / f"{instance_id}.{model_name_or_path}.eval.log"
    if not log_fname.exists():
        return (
            False,
            f"Something went wrong, no log file found fo the test run at {log_fname}",
        )
    log_text = log_fname.read_text()
    # log_lines = log_text.splitlines()
    # log_lines = [line for line in log_lines if line.startswith(">>>>")]
    # print("\n".join(log_lines))
    passed = TESTS_PASSED in log_text
    return passed, log_text


def run_dev_tests(entry, git_dname, test_server_host=None):
    # during development, we don't have a test patch, so we just run existing
    # tests or tests added in the model patch
    return diff_and_run_tests(
        entry, git_dname, use_test_patch=False, test_server_host=test_server_host
    )


def run_eval_tests(entry, git_dname, test_server_host=None):
    # like run_dev_tests, but now we have the test patch used for evaluation.
    # also, we skip evaluating any additional tests added in the model patch,
    # in case of git patch apply conflicts and for a cleaner evaluation.
    return diff_and_run_tests(
        entry, git_dname, use_test_patch=True, test_server_host=test_server_host
    )


def diff_and_run_tests(entry, git_dname, use_test_patch=False, test_server_host=None):
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
    next_commit = subprocess.run(
        command, shell=True, capture_output=True, text=True
    ).stdout
    model_patch = diff_versus_commit(git_dname, next_commit)
    # switch working directory to this directory
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    # print(f"Base commit: {entry['base_commit']}")
    # print(f"Next commit after base_commit: {next_commit}")
    # print("Model Patch:")
    # print(model_patch)
    # some of the "potentially-existing" test directives are actually not
    # present in the repo but in the test patch. we need to filter them out as
    # trying to run non-existent tests will cause the test runner to fail.
    potentially_existing_test_directives: List[str] = [
        # normalize
        str(TestDirective.parse(td))
        for td in json.loads(entry["PASS_TO_PASS"])
    ]
    module_based = entry["repo"] == "django/django"
    method_only = entry["repo"] == "sympy/sympy"
    # print(f"Existing Test directives: {str(existing_test_directives)}")
    if use_test_patch:
        additional_test_directives: List[str] = [
            # normalize
            str(TestDirective.parse(td))
            for td in json.loads(entry["FAIL_TO_PASS"])
        ]
        existing_test_directives: List[str] = potentially_existing_test_directives
    else:
        newly_added_test_directives: List[TestDirective] = (
            extract_added_test_directives_from_patch(
                model_patch, git_dname, module_based=module_based, method_only=method_only
            )
        )
        additional_test_directives: List[str] = [
            str(td) for td in newly_added_test_directives
        ]
        parsed_potentially_existing: List[TestDirective] = [
            TestDirective.parse(td) for td in json.loads(entry["PASS_TO_PASS"])
        ]

        # then use the file path to extract the test directives from the file (first check if the file exists)
        existing_test_directives: List[str] = []
        parser = Parser(PY_LANGUAGE)

        # key: potential test directive, value: actual test directive that exists that can be backup
        backup_test_directives: Dict[str, TestDirective] = {}

        # we don't know if these test directives actually exist in the repo, so
        # we need to check before we count on that
        for pd in parsed_potentially_existing:
            if pd.file_path is None:
                # this is a docstring test directive OR a test directive that excludes file path, we can't check if it exists
                # but we'll search for the file that contains this using the ripgrep command
                if pd.docstring is not None:
                    search_term = pd.docstring.replace('"', '\\"')
                elif pd.method_only:
                    search_term = pd.method_name
                else:
                    print(f"Bad test directive missing filepath, method name and docstring: {pd}")
                    continue
                command = f"""rg -l -tpy -F "{search_term}" {git_dname}"""
                paths = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                ).stdout
                # print(f"Searching for docstring: {pd.docstring}")
                # print(f"Found paths: {paths}")
                found = False
                for path in paths.split("\n"):
                    if found:
                        break
                    if path == "":
                        continue
                    with open(path) as f:
                        source_code = bytes(f.read(), "utf8")
                        relative_path = path[len(git_dname) + 1 :]
                        file_test_directives = extract_test_directives_from_source_code(
                            source_code, parser, relative_path, whitelist_rows=None
                        )
                        for ftd in file_test_directives:
                            ftd.module_based = module_based
                            ftd.method_only = method_only
                            if ftd.module_based:
                                if ftd.docstring is None:
                                    continue
                                # compare ignoring all whitespace
                                if " ".join(pd.docstring.split()) == " ".join(
                                    ftd.docstring.split()
                                ):
                                    existing_test_directives.append(str(ftd))
                                    found = True
                                    break
                            else:
                                # method_only is True, so we only need to check the method name
                                if pd.method_name == ftd.method_name:
                                    existing_test_directives.append(str(ftd))
                                    found = True
                                    break
                if not found:
                    # print(f"Skipping not actually existing test directive 0: {pd}")
                    continue
            elif not os.path.exists(f"{git_dname}/{pd.file_path}"):
                # print(f"Skipping not actually existing test directive 1: {pd}")
                # try backup with the parent directory of the file path
                # do this in a while loop until we find a parent directory that exists or we reach the root
                # if the file doesn't exist, then we can use the higher-level
                # module name (or test directory name) as the test directive, by
                # finding the first parent directory that exists. if none exist,
                # then we just run all tests (ie. no test directives) since we
                # won't have existing_test_directives at the end
                parent_dir = pd.file_path
                while True:
                    parent_dir = "/".join(parent_dir.split("/")[:-1])
                    if parent_dir == "":
                        break
                    if os.path.exists(f"{git_dname}/{parent_dir}"):
                        backup_test_directives[str(pd)] = str(
                            TestDirective(parent_dir, module_based=module_based)
                        )
                        break
            else:
                with open(f"{git_dname}/{pd.file_path}") as f:
                    source_code = bytes(f.read(), "utf8")
                    file_test_directives = extract_test_directives_from_source_code(
                        source_code, parser, pd.file_path, whitelist_rows=None
                    )
                    for ftd in file_test_directives:
                        ftd.module_based = module_based
                    if str(pd) in [str(ftd) for ftd in file_test_directives]:
                        existing_test_directives.append(str(pd))
                    else:
                        # print(f"Skipping not actually existing test directive 2: {pd}")
                        # find the first test directive in the file that matches the same class name
                        # and add to backup_test_directives
                        found = False
                        for td in file_test_directives:
                            if pd.class_name != None and pd.class_name == td.class_name:
                                td.method_name = None
                                backup_test_directives[str(pd)] = td
                                found = True
                                break
                        if not found and len(file_test_directives) > 0:
                            td = file_test_directives[0]
                            td.method_name = None
                            td.class_name = None
                            backup_test_directives[str(pd)] = td
                            found = True
        # TODO? should we add the backup test directives in the non-empty case too, eg
        # one skipped but others found?
        if len(existing_test_directives) == 0:
            # print(f"All tests skipped, using backup test directives instead: {backup_test_directives}")
            for pd, bd in backup_test_directives.items():
                if bd is not None:
                    existing_test_directives.append(bd)
            # make unique
            existing_test_directives = list(set(existing_test_directives))
        else:
            # print(f"Not using backup test directives: {backup_test_directives}")
            existing_test_directives = existing_test_directives
    test_directives = existing_test_directives + additional_test_directives
    # print(f"Existing test directives: { existing_test_directives }")
    # print(f"Additional test directives: { additional_test_directives }")
    # print(f"Test directives: {test_directives}")
    passed, output = run_tests(
        entry,
        model_patch=model_patch,
        use_test_patch=use_test_patch,
        test_directives=test_directives,
        test_server_host=test_server_host,
    )
    # We were UNABLE to run tests
    if passed is None:
        return passed, output
    # Just keep the actual output from the tests that were run, throwing away the harness output
    if "Std. Output:" not in output:
        raise ValueError(
            "Output does not contain the expected 'Std. Output:' string, did the SWE-bench-docker test runner change?"
        )
    output = output.split("Std. Output:")[-1]
    # throw away very wordy test directives list
    if len(test_directives) > 0:
        output = output.replace(
            " ".join(test_directives), "[... omitted test directives]"
        )
    # throw away additional INFO/DEBUG lines from SWE-bench-docker
    output = "\n".join(
        [
            line
            for line in output.split("\n")
            if not line.startswith("[INFO]") and not line.startswith("[DEBUG]")
        ]
    )
    return passed, output


def extract_added_test_directives_from_patch(git_patch, git_dname, module_based, method_only):
    """Given a `git_patch`, extract the test directives for any new tests
    that have been added to the repo since the `base_commit`.
    """
    if git_dname == None:
        raise ValueError("git_dname is None")
    parser = Parser(PY_LANGUAGE)
    test_directives = []
    file_rows = extract_file_rows_from_patch(git_patch)
    # print("Extracted File rows:")
    # print(file_rows)
    for file_path, rows in file_rows.items():
        # TODO check if the file is in the testpaths:
        if file_path.endswith(".py"):
            with open(f"{git_dname}/{file_path}") as f:
                source_code = bytes(f.read(), "utf8")
                file_test_directives = extract_test_directives_from_source_code(
                    source_code, parser, file_path, rows
                )
                for ftd in file_test_directives:
                    ftd.module_based = module_based
                    ftd.method_only = method_only
                test_directives.extend([str(ftd) for ftd in file_test_directives])
    return test_directives


def extract_test_directives_from_source_code(
    source_code: bytes,
    parser: Parser,
    relative_path: str,
    whitelist_rows: Optional[List[int]],
) -> List[TestDirective]:
    test_directives = []
    tree = parser.parse(source_code)
    query = PY_LANGUAGE.query(
        """
(class_definition
  name: (identifier) @class.name
  body: (_
    (function_definition
      name: (identifier) @class.method.name
      body: (_
        . (expression_statement
          . (string (string_content) @docstring_content)
        )?
      )
    ) @overlap
  )
)
(class_definition
  name: (identifier) @class.name
  body: (_
    (decorated_definition
      definition: (function_definition
        name: (identifier) @class.method.name
        body: (_
          . (expression_statement
            . (string (string_content) @docstring_content)
          )?
        )
      )
    ) @overlap
  )
)
(module
  (function_definition
    name: (identifier) @function.name
    body: (_
      . (expression_statement
        . (string (string_content) @docstring_content)
      )?
    )
  ) @overlap
)
(module
  (decorated_definition
    definition: (function_definition
      name: (identifier) @function.name
      body: (_
        . (expression_statement
          . (string (string_content) @docstring_content)
        )?
      )
    )
  ) @overlap
)
"""
    )
    matches = query.matches(tree.root_node)
    for match in matches:
        if whitelist_rows != None:
            overlap_node = match[1]["overlap"]
            has_overlap = any(
                [
                    overlap_node.start_point.row <= row <= overlap_node.end_point.row
                    for row in whitelist_rows
                ]
            )
            if not (has_overlap):
                continue
        if "function.name" in match[1]:
            node = match[1]["function.name"]
            node.start_point.row
            function_name = source_code[node.start_byte : node.end_byte].decode("utf8")
            if function_name.startswith("test"):
                docstring = None
                if "docstring_content" in match[1]:
                    docstring_node = match[1]["docstring_content"]
                    docstring = source_code[
                        docstring_node.start_byte : docstring_node.end_byte
                    ].decode("utf8")
                directive = TestDirective(
                    relative_path, method_name=function_name, docstring=docstring
                )
                test_directives.append(directive)
        elif "class.method.name" in match[1]:
            node = match[1]["class.method.name"]
            method_name = source_code[node.start_byte : node.end_byte].decode("utf8")
            class_node = match[1]["class.name"]
            class_name = source_code[
                class_node.start_byte : class_node.end_byte
            ].decode("utf8")
            # NOTE checking class name is not sufficient as inheritence can also
            # play a part, and that's a lot more work, so we'll just leave the
            # class check out until it becomes a problem. for common conventions
            # see: https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#conventions-for-python-test-discovery
            # class_name.startswith("Test") and
            if method_name.startswith("test"):
                docstring = None
                if "docstring_content" in match[1]:
                    docstring_node = match[1]["docstring_content"]
                    docstring = source_code[
                        docstring_node.start_byte : docstring_node.end_byte
                    ].decode("utf8")
                directive = TestDirective(
                    relative_path,
                    method_name=method_name,
                    class_name=class_name,
                    docstring=docstring,
                )
                test_directives.append(directive)
        else:
            raise ValueError(f"Unexpected match: {match[1]}")
    return test_directives
    # print("Matches:")
    # print(matches)


def extract_file_rows_from_patch(git_patch):
    file_rows = {}
    # Regular expressions for matching file headers and chunk headers
    file_pattern = re.compile(r"^diff --git a/.* b/(.*)$")
    chunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    current_file = None
    current_line = 0
    last_added_line = {}
    last_removed_line = {}
    last_added_line_number = {}
    last_removed_line_number = {}
    # Process each line of the git diff
    for line in git_patch.split("\n"):
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
        if line.startswith("+") and not line.startswith("+++"):
            file_rows[current_file].append(current_line)
            last_added_line[current_file] = line[1:]
            last_added_line_number[current_file] = current_line
            current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            file_rows[current_file].append(current_line)
            last_removed_line[current_file] = line[1:]
            last_removed_line_number[current_file] = current_line
        elif line.startswith(" "):
            current_line += 1
    # Remove newline-only changes at the end of files
    for file in file_rows:
        if (
            file in last_added_line
            and file in last_removed_line
            and last_added_line[file] == last_removed_line[file]
            and last_removed_line_number[file] == last_added_line_number[file]
        ):
            file_rows[file] = file_rows[file][:-2]
    return file_rows


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
        test_server_host = None  # optional
        if len(argv) > 4:
            test_server_host = sys.argv[4]
        dataset = get_lite_dataset()
        entry = dataset[instance_id]
        # print(entry)
        if command == "run_dev_tests":
            passed, output = run_dev_tests(entry, repo_dir, test_server_host)
        elif command == "run_eval_tests":
            passed, output = run_eval_tests(entry, repo_dir, test_server_host)
        else:
            raise ValueError(f"Invalid command: {command}")
        print(output)
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
