def run_tests(entry, model_patch = None, use_new_tests=False, model_name_or_path="none"):
    if use_new_tests:
        test_patch = entry['test_patch']
    else:
        test_patch = NOOP_PATCH.format(nonce="test_patch")
        "test_patch": test_patch,