#!/usr/bin/env python3
"""Master audit runner — executes phases and writes logs under audit_logs/."""
import os
import re
import subprocess
import sys
import venv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parent / "audit_logs"
AUDIT.mkdir(parents=True, exist_ok=True)
VENV = ROOT / ".venv_audit"


def run(cmd, cwd=None, env=None, log_name=None, check=False):
    cwd = cwd or ROOT
    print(f">>> {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    out = f"=== CMD: {' '.join(cmd)} ===\nexit={r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}\n"
    if log_name:
        (AUDIT / log_name).write_text(out, encoding="utf-8")
        with open(AUDIT / log_name, "a", encoding="utf-8") as f:
            pass  # ensure flushed
    print(out[:2000])
    if check and r.returncode != 0:
        print(f"FAILED: {cmd}")
    return r


def phase0():
    lines = [f"=== PHASE 0 — Environment Fingerprint ===", f"Timestamp: {datetime.now().isoformat()}"]
    for title, cmd in [
        ("systeminfo", ["systeminfo"]),
        ("python", ["python", "--version"]),
        ("pip", ["pip", "--version"]),
        ("node", ["node", "--version"]),
        ("go", ["go", "version"]),
        ("cmake", ["cmake", "--version"]),
        ("nvidia-smi", ["nvidia-smi"]),
        ("pip list", ["pip", "list"]),
        ("pip freeze", ["pip", "freeze"]),
    ]:
        lines.append(f"\n=== {title} ===")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            lines.append(r.stdout + r.stderr)
            if title == "nvidia-smi" and r.returncode != 0:
                lines.append("NO GPU (nvidia-smi failed)")
        except FileNotFoundError:
            lines.append(f"NOT FOUND: {cmd[0]}")
    import shutil
    du = shutil.disk_usage("C:\\")
    lines.append(f"\n=== Disk C: ===\nfree={du.free} total={du.total}")
  # memory via wmic fallback
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory | Format-List | Out-String)"],
            capture_output=True, text=True,
        )
        lines.append(f"\n=== Memory ===\n{r.stdout}")
    except Exception as e:
        lines.append(f"memory query failed: {e}")
    (AUDIT / "phase0_environment.log").write_text("\n".join(lines), encoding="utf-8")


def phase1():
    log = AUDIT / "phase1_static_integrity.log"
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"=== PHASE 1 ===\n{datetime.now().isoformat()}\n")

    def L(msg):
        print(msg)
        with open(log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    for title, cmd in [("git status", ["git", "status"]), ("git diff --stat", ["git", "diff", "--stat"])]:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        L(f"\n=== {title} ===\n{r.stdout}{r.stderr}")

    L("\n=== py_compile ===")
    errors = []
    pyfiles = [p for p in ROOT.rglob("*.py") if ".venv_audit" not in str(p)]
    for py in pyfiles:
        r = subprocess.run([sys.executable, "-m", "py_compile", str(py)], capture_output=True, text=True)
        if r.returncode:
            errors.append(str(py))
            L(f"FAIL {py}: {r.stderr}")
    L(f"Total {len(pyfiles)} files, {len(errors)} compile errors")

    L("\n=== imports ===")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for mod in ["vllm_port", "vllm_port.smollm3"]:
        r = subprocess.run([sys.executable, "-c", f"import {mod}; print('ok')"], cwd=ROOT, env=env, capture_output=True, text=True)
        L(f"{mod}: exit={r.returncode} {r.stdout}{r.stderr}")

    L("\n=== ruff F,E9 ===")
    r = subprocess.run(["ruff", "check", ".", "--select=F,E9"], cwd=ROOT, capture_output=True, text=True)
    L(r.stdout + r.stderr + f"\nexit={r.returncode}")

    L("\n=== config parse ===")
    import json
    import tomllib
    import yaml
    parse_err = 0
    for f in ROOT.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".yaml", ".yml", ".json", ".toml"):
            continue
        if ".venv_audit" in str(f):
            continue
        try:
            t = f.read_text(encoding="utf-8")
            if f.suffix.lower() == ".json":
                json.loads(t)
            elif f.suffix.lower() == ".toml":
                tomllib.loads(t)
            else:
                yaml.safe_load(t)
        except Exception as e:
            parse_err += 1
            L(f"PARSE FAIL {f}: {e}")
    L(f"parse errors: {parse_err}")

    L("\n=== dangling refs (Docker/compose) ===")
    dangling = []
    for f in list((ROOT / "docker").glob("*")) + list((ROOT / ".github" / "workflows").glob("*")):
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"COPY\s+(\S+)", text):
            ref = m.group(1).rstrip("/")
            if not (ROOT / ref).exists() and not (ROOT / ref.split("/")[0]).exists():
                # check more carefully
                p = ROOT / ref
                if not p.exists():
                    dangling.append((f.name, ref))
                    L(f"DANGLING COPY {f.name}: {ref}")
    L(f"dangling: {len(dangling)}")


def phase2():
    log = AUDIT / "phase2_venv_install.log"
    if VENV.exists():
        import shutil
        shutil.rmtree(VENV, ignore_errors=True)
    venv.create(VENV, with_pip=True)
    pip = VENV / "Scripts" / "pip.exe"
    py = VENV / "Scripts" / "python.exe"
    lines = [f"=== PHASE 2 — Clean venv install ===", f"{datetime.now().isoformat()}"]
    for req in ["requirements.txt", "requirements-dev.txt"]:
        lines.append(f"\n=== pip install -r {req} ===")
        r = subprocess.run([str(pip), "install", "-r", req], cwd=ROOT, capture_output=True, text=True)
        lines.append(f"exit={r.returncode}\n{r.stdout}\n{r.stderr}")
        if r.returncode != 0:
            log.write_text("\n".join(lines), encoding="utf-8")
            return False
    log.write_text("\n".join(lines), encoding="utf-8")
    return True


def phase3():
    py = VENV / "Scripts" / "python.exe"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    log_path = AUDIT / "pytest_full.log"
    with open(log_path, "w", encoding="utf-8") as f:
        r = subprocess.run(
            [str(py), "-m", "pytest", "-vv", "--tb=long", "-rA", "--durations=0", "--capture=no", "tests/"],
            cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
        )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== pytest exit code: {r.returncode} ===\n")
    # parse junit-like summary
    text = log_path.read_text(encoding="utf-8", errors="replace")
  # collect warnings
    warns = {}
    for line in text.splitlines():
        if "Warning" in line or "warning" in line.lower():
            warns[line.strip()] = warns.get(line.strip(), 0) + 1
    (AUDIT / "phase3_warnings.txt").write_text(
        "\n".join(f"{c}x {w}" for w, c in sorted(warns.items(), key=lambda x: -x[1])),
        encoding="utf-8",
    )
    return r.returncode


if __name__ == "__main__":
    phase0()
    phase1()
    ok = phase2()
    if ok:
        phase3()
