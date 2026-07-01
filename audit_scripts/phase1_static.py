#!/usr/bin/env python3
"""Phase 1 static integrity checks — writes to audit_logs/phase1_static_integrity.log"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT.parent / "audit_logs" / "phase1_static_integrity.log"


def log(msg: str) -> None:
    print(msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(
            f"=== PHASE 1 — Static Integrity Pass ===\n"
            f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n"
        )

    # git
    for title, cmd in [
        ("git status", ["git", "status"]),
        ("git diff --stat", ["git", "diff", "--stat"]),
    ]:
        log(f"\n=== {title} ===")
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log(r.stdout + r.stderr)

    # py_compile
    log("\n=== py_compile all .py ===")
    errors = []
    for py in ROOT.rglob("*.py"):
        if ".venv_audit" in str(py) or "__pycache__" in str(py):
            continue
        r = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            errors.append((py, r.stderr))
            log(f"FAIL: {py}\n{r.stderr}")
    log(f"COMPILE: {len(list(ROOT.rglob('*.py')))} files, {len(errors)} errors")

    # top-level imports
    log("\n=== import top-level packages ===")
    packages = ["vllm_port", "python.predictor", "python.scheduler"]
    for pkg in packages:
        r = subprocess.run(
            [sys.executable, "-c", f"import {pkg}; print('OK', {pkg!r})"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        log(f"{pkg}: exit={r.returncode}\n{r.stdout}{r.stderr}")

    # dangling refs in Docker/CI
    log("\n=== dangling file references ===")
    import re

    patterns = [
        (r"COPY\s+(\S+)", "docker"),
        (r"volumes:\s*\n\s*-\s*(\S+)", "compose"),
        (r"models_dir[:\s]+[\"']?([^\"'\s]+)", "config"),
    ]
    refs = set()
    for glob in ["docker/*", "docker-compose*.yml", "configs/*", ".github/workflows/*", "ci/*"]:
        for f in ROOT.glob(glob):
            if f.is_file():
                text = f.read_text(encoding="utf-8", errors="replace")
                for pat, _ in patterns:
                    for m in re.finditer(pat, text):
                        refs.add((str(f.relative_to(ROOT)), m.group(1)))

    dangling = []
    for src, ref in sorted(refs):
        if ref.startswith("$") or ref.startswith("${"):
            continue
        candidate = (ROOT / ref).resolve() if not ref.startswith("/") else Path(ref)
        if not candidate.exists() and not (ROOT / ref).exists():
            dangling.append((src, ref))
            log(f"DANGLING: {src} -> {ref}")
    log(f"Dangling refs found: {len(dangling)}")

    # ruff
    log("\n=== ruff check . --select=F,E9 ===")
    r = subprocess.run(
        ["ruff", "check", ".", "--select=F,E9"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    log(r.stdout + r.stderr)
    log(f"ruff exit code: {r.returncode}")

    # config parse
    log("\n=== config file parse validation ===")
    import tomllib

    try:
        import yaml
    except ImportError:
        yaml = None

    parse_errors = []
    for f in ROOT.rglob("*"):
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf not in (".yaml", ".yml", ".json", ".toml"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
            if suf == ".json":
                json.loads(text)
            elif suf in (".yaml", ".yml"):
                if yaml is None:
                    log(f"SKIP yaml (no pyyaml): {f}")
                else:
                    yaml.safe_load(text)
            elif suf == ".toml":
                tomllib.loads(text)
        except Exception as e:
            parse_errors.append((f, str(e)))
            log(f"PARSE FAIL: {f}: {e}")
    log(f"Config parse errors: {len(parse_errors)}")

    return 1 if errors or parse_errors else 0


if __name__ == "__main__":
    sys.exit(main())
