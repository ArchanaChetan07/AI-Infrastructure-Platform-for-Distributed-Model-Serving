# =============================================================================
# SmolLM3 vLLM Port — Production Dockerfile
# =============================================================================
# Base: official vLLM image (latest — requires transformers>=4.53 for SmolLM3)
# Adds: the smollm3.py native port, patched registry, and scripts.
#
# Build:
#   docker build -t vllm-smollm3:latest -f docker/Dockerfile .
#
# Run:
#   docker run --runtime nvidia --gpus all \
#     -e HF_TOKEN=$HF_TOKEN \
#     -e VLLM_ATTENTION_BACKEND=XFORMERS \
#     -p 8000:8000 vllm-smollm3:latest

FROM vllm/vllm-openai:latest

LABEL org.opencontainers.image.title="vLLM SmolLM3 Native Port"
LABEL org.opencontainers.image.description="Project 2A: SmolLM3ForCausalLM native vLLM model"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /workspace

# Copy the native model implementation into the installed vLLM package
COPY vllm_port/smollm3.py /workspace/smollm3.py
RUN python3 -c "import os, shutil, vllm; dest=os.path.join(vllm.__path__[0],'model_executor','models','smollm3.py'); shutil.copy('/workspace/smollm3.py', dest); print('installed', dest)"

# Apply registry patch (removes Transformers fallback, adds native entry)
COPY scripts/patch_registry.py /workspace/patch_registry.py
RUN python3 /workspace/patch_registry.py

# Copy __init__.py patch (adds SmolLM3ForCausalLM to exports)
COPY scripts/patch_init.py /workspace/patch_init.py
RUN python3 /workspace/patch_init.py

# Copy test and benchmark scripts
COPY tests/ /workspace/tests/
COPY benchmarks/ /workspace/benchmarks/
RUN pip install pytest pytest-asyncio ruff mypy -q

# Model and env defaults (T1000 8GB tuned)
ENV VLLM_MODEL=HuggingFaceTB/SmolLM3-3B
ENV VLLM_DTYPE=float16
ENV VLLM_MAX_MODEL_LEN=2048
ENV VLLM_GPU_MEM_UTIL=0.85
ENV VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

EXPOSE 8000

HEALTHCHECK \
    --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "HuggingFaceTB/SmolLM3-3B", \
     "--trust-remote-code", \
     "--dtype", "float16", \
     "--max-model-len", "512", \
     "--gpu-memory-utilization", "0.85", \
     "--enforce-eager", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
