.PHONY: help install test test-unit lint format docker-build run benchmark clean
.PHONY: train gateway benchmark-scheduler build-cpp build-go docker-stack

HF_TOKEN ?= $(shell echo $$HF_TOKEN)
MODEL ?= HuggingFaceTB/SmolLM3-3B
VLLM_URL ?= http://localhost:8002
GATEWAY_URL ?= http://localhost:8080
PYTHONPATH := .:python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements-dev.txt

test:  ## Run all unit tests
	PYTHONPATH=$(PYTHONPATH) pytest tests/ -m unit -v --tb=short

test-unit: test

test-smollm3:  ## Run SmolLM3 model port tests
	PYTHONPATH=$(PYTHONPATH) pytest tests/test_smollm3.py -m unit -v

lint:  ## Lint Python code
	ruff check python/ tests/
	black --check python/ tests/

format:  ## Format Python code
	ruff check python/ tests/ --fix
	black python/ tests/

docker-build:  ## Build vLLM SmolLM3 image
	docker build -t vllm-smollm3:latest -f docker/Dockerfile .

run:  ## Run vLLM SmolLM3 backend
	docker run --runtime nvidia --gpus all \
		-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
		-p 8002:8000 vllm-smollm3:latest

train:  ## Train output-length predictor
	PYTHONPATH=$(PYTHONPATH) python python/training/train.py

gateway:  ## Start Go API gateway
	cd go && go run ./cmd/gateway

benchmark:  ## Run vLLM throughput benchmark
	python benchmarks/run_benchmark.py --base-url $(VLLM_URL) --model $(MODEL)

benchmark-scheduler:  ## Benchmark FCFS vs SJF schedulers
	PYTHONPATH=$(PYTHONPATH) python python/benchmark/compare.py

build-cpp:  ## Build C++ runtime
	cmake -S cpp -B cpp/build
	cmake --build cpp/build --config Release

build-go:  ## Build Go gateway binary
	cd go && go build -o ../bin/gateway ./cmd/gateway

test-cpp:  ## Run C++ tests
	ctest --test-dir cpp/build --output-on-failure

docker-stack:  ## Start full scheduler stack
	docker compose -f docker/docker-compose.yml up -d

smoke-test:  ## Smoke test gateway
	curl -sf $(GATEWAY_URL)/health
	curl -sf $(GATEWAY_URL)/scheduler/stats

clean:  ## Clean build artifacts
	rm -rf cpp/build bin/ .pytest_cache .ruff_cache python/**/__pycache__
