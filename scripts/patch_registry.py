"""
patch_registry.py — Applied during Docker build to register SmolLM3 natively.
Supports vLLM 0.4.x (_GENERATION_MODELS in __init__.py) and newer registry.py.
"""
import os
import re
import sys

try:
    import vllm
except ImportError:
    print("vLLM not installed — skipping registry patch")
    sys.exit(0)

MODELS_DIR = os.path.join(vllm.__path__[0], "model_executor", "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.py")
INIT_PATH = os.path.join(MODELS_DIR, "__init__.py")

NATIVE_ENTRY = '"SmolLM3ForCausalLM": ("smollm3", "SmolLM3ForCausalLM"),'
TRANSFORMERS_ENTRY_RE = re.compile(
    r'\s*"SmolLM3ForCausalLM":\s*\(\s*"transformers",\s*"TransformersForCausalLM",\s*\),'
)
NATIVE_ENTRY_RE = re.compile(
    r'"SmolLM3ForCausalLM":\s*\(\s*"smollm3",\s*"SmolLM3ForCausalLM",\s*\)'
)

INSERT_CANDIDATES = [
    '"StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),',
    '"Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),',
]


def patch_content(content: str, label: str) -> str:
    content = TRANSFORMERS_ENTRY_RE.sub("", content)

    if NATIVE_ENTRY_RE.search(content):
        print(f"SmolLM3ForCausalLM already registered natively in {label} ✓")
        return content

    native_line = f"\n    {NATIVE_ENTRY}  # native SmolLM3 port"
    for insert_after in INSERT_CANDIDATES:
        if insert_after in content:
            content = content.replace(insert_after, insert_after + native_line)
            print(f"SmolLM3ForCausalLM registered natively in {label} ✓")
            return content

    print(f"WARNING: Could not find insertion point in {label} — manual patch needed")
    return content


if os.path.isfile(REGISTRY_PATH):
    with open(REGISTRY_PATH) as f:
        content = f.read()
    with open(REGISTRY_PATH, "w") as f:
        f.write(patch_content(content, "registry.py"))
elif os.path.isfile(INIT_PATH):
    with open(INIT_PATH) as f:
        content = f.read()
    with open(INIT_PATH, "w") as f:
        f.write(patch_content(content, "__init__.py"))
else:
    print(f"No registry found under {MODELS_DIR} — skipping patch")
    sys.exit(0)
