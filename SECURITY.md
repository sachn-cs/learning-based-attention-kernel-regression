# Security Policy

## Supported Versions

The following versions of LAKER are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in LAKER, please report it responsibly.

**Please do not disclose security issues publicly** (e.g., via GitHub issues) until
we have had a chance to address them.

Instead, send an email to the maintainers with:

- A description of the vulnerability
- Steps to reproduce (if applicable)
- The potential impact
- Any suggested fixes or mitigations

We will acknowledge receipt within 48 hours and aim to provide an initial
assessment within 5 business days. We ask that you allow 90 days after we
acknowledge the report before publicly disclosing the vulnerability, to give us
time to develop and release a fix.

## Security Best Practices

When using LAKER:

- Keep dependencies up to date (`pip install -U laker`)
- Only load model checkpoints from trusted sources
- Be cautious when running untrusted example scripts
