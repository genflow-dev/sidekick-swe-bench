import sys
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_directives
NOOP_PATCH = (
    "diff --git a/empty.file.{nonce}.ignore b/empty.file.{nonce}.ignore\n"
    "new file mode 100644\n"
    "index 0000000..e69de29\n"
)
def run_tests(entry, model_patch=None, use_new_tests=False, model_name_or_path="none"):
        test_patch = entry["test_patch"]
        "test_cmd": test_cmd,
    asyncio.run(run_docker_evaluation(entry_instance, namespace, log_dir, timeout, log_suffix))
    log_fname = Path(log_dir) / f"{instance_id}.{model_name_or_path}.eval.log"
    log_lines = [line for line in log_lines if line.startswith(">>>>")]
    print("\n".join(log_lines))
    passed = ">>>>> All Tests Passed" in log_text
        if instance_ids and entry["instance_id"] not in instance_ids:
        dump(num_passed / num)
if __name__ == "__main__":