import tempfile
def run_tests(entry, model_patch = None):
    instance_id = entry["instance_id"]
    if not model_patch:
        model_patch = NOOP_PATCH.format(nonce="model_patch")

    model_name_or_path = "testing"

        "instance_id": entry["instance_id"],
        "model_name_or_path": model_name_or_path,
        "model_patch": model_patch,
    log_dir = tempfile.TemporaryDirectory(dir="/Users/gauthier/tmp").name
    dump(log_dir)

    log_fname = Path(log_dir) / f'{instance_id}.{model_name_or_path}.eval.log'
    return log_text


dataset = get_dataset()
num = 0
num_passed = 0
for entry in dataset.values():
    test_text = run_tests(entry)
    passed = '>>>>> All Tests Passed' in test_text
    num += 1
    if passed:
        num_passed += 1
    dump(num_passed/num)