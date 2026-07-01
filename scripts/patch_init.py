"""
patch_init.py — Adds SmolLM3ForCausalLM to __init__.py exports (newer vLLM only).
vLLM 0.4.x registers models via _GENERATION_MODELS lazy loading — no-op there.
"""
import os
import re
import sys

try:
    import vllm
except ImportError:
    print("vLLM not installed — skipping __init__ patch")
    sys.exit(0)

INIT_PATH = os.path.join(vllm.__path__[0], "model_executor", "models", "__init__.py")

try:
    with open(INIT_PATH) as f:
        content = f.read()
except FileNotFoundError:
    print(f"__init__.py not found at {INIT_PATH} — skipping patch")
    sys.exit(0)

if "_GENERATION_MODELS" in content and '"SmolLM3ForCausalLM"' in content:
    print("SmolLM3ForCausalLM registered via _GENERATION_MODELS ✓")
    sys.exit(0)

if "SmolLM3ForCausalLM" in content:
    print("SmolLM3ForCausalLM already exported ✓")
    sys.exit(0)

import_line = "from .smollm3 import SmolLM3ForCausalLM  # native SmolLM3 port"
if "from .smollm3 import" not in content:
    match = re.search(r"(from \.\w+ import \w+ForCausalLM\n)", content)
    if match:
        content = content[: match.end()] + import_line + "\n" + content[match.end() :]
    else:
        content = import_line + "\n" + content

if "__all__" in content:
    content = re.sub(
        r"(__all__\s*=\s*\[[^\]]*)",
        r'\1\n    "SmolLM3ForCausalLM",',
        content,
        count=1,
    )

with open(INIT_PATH, "w") as f:
    f.write(content)

print("SmolLM3ForCausalLM added to __init__.py exports ✓")
