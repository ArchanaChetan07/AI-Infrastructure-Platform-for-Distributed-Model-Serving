"""pytest configuration — SmolLM3 vLLM Port"""


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests, no GPU/network needed")
    config.addinivalue_line(
        "markers", "accuracy: output accuracy vs HuggingFace (needs GPU+HF_TOKEN)"
    )
    config.addinivalue_line("markers", "integration: full vLLM pipeline tests (needs GPU+HF_TOKEN)")
    config.addinivalue_line("markers", "benchmark: throughput/perf tests (needs GPU+HF_TOKEN)")
