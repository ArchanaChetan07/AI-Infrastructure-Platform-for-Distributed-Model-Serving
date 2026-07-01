# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a vulnerability

If you discover a security issue, please **do not** open a public issue with exploit details.

1. Open a [GitHub Security Advisory](https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving/security/advisories/new), or
2. Contact the repository owner privately.

Include steps to reproduce, impact assessment, and any suggested mitigation.

## Security practices in this project

- Secrets (`HF_TOKEN`, API keys) must be supplied via environment variables or CI secrets—never committed.
- Container images run as non-root where possible.
- Dependencies are scanned in CI with `pip-audit`.
- Gateway enforces request timeouts and connect timeouts to vLLM backends.
- YAML configs are loaded with safe parsers; no arbitrary code execution from config files.

## Token hygiene

Rotate credentials immediately if they are exposed in logs, chat, or version control.
