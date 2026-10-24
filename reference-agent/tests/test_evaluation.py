from reference_agent.evaluation.harness import evaluate, load_samples


def test_eval_runs_on_fixed_set():
    report = evaluate()
    assert report["total"] == 3
    assert report["passed"] == 3
    assert report["pass_rate"] == 1.0


def test_load_samples_shape():
    samples = load_samples()
    assert all("input" in s and "expect_contains" in s for s in samples)
