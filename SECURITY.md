# Security Policy

QuantFlow places real orders on a cryptocurrency exchange. A vulnerability here can cost
someone money directly, so security reports are taken seriously and answered.

## Reporting a vulnerability

**Please do not open a public issue, pull request or discussion for a security problem.**

Report privately through **[GitHub Private Vulnerability Reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)** —
the **Security** tab → **Report a vulnerability**.

<!-- MAINTAINER: if you prefer email, replace the line below with a real address you
     monitor, and delete this comment. Otherwise leave GitHub PVR as the only channel. -->

### What to include

- What the issue is and what an attacker gains.
- Steps to reproduce, or a minimal proof of concept.
- Affected version, commit SHA or release tag.
- Your configuration where it matters — mode (`backtest`/`paper`/`demo`/`live`),
  environment (`demo`/`testnet`/`mainnet`), deployment shape.
- **Never include real API keys, secrets or account identifiers in a report.** Redact them.
  If a key was exposed, rotate it at the exchange first, then report.

### What to expect

| | |
|---|---|
| Acknowledgement | Within **7 days** |
| Initial assessment | Within **14 days** |
| Fix or mitigation plan | Communicated once assessed; timing depends on severity |
| Disclosure | Coordinated with you; credit given unless you decline |

This is a volunteer-maintained project with no paid security team and **no bug-bounty
programme**. Responses are best-effort and there is no monetary reward.

## Scope

**In scope**

- Anything that could cause an unintended order, an order at an unintended size or price,
  or a bypass of the risk engine.
- Anything that defeats the live-trading interlocks (mode gate, confirmation token,
  mainnet refusal, demo/mainnet credential separation).
- Credential leakage — into logs, API responses, reports, error messages or the
  dashboard.
- Authentication or authorization flaws in the API.
- Injection, deserialization, path traversal or SSRF in the API, CLI or workers.
- Dependency vulnerabilities that are actually reachable from QuantFlow's code paths.

**Out of scope**

- Losing money on a trade. Market risk is not a vulnerability. See the trading-risk
  disclaimer in the README.
- Strategy performance, or a strategy that does not make money.
- Vulnerabilities in Bybit, OANDA, MetaTrader or any other third-party venue — report
  those to the venue.
- Findings that require an attacker to already have your `.env`, your shell or your
  exchange credentials.
- Anything caused by a configuration the documentation explicitly warns against —
  most notably exposing the API to a public network (see below).
- Automated scanner output with no demonstrated impact.
- Dev-only toolchain advisories with no production reachability.

## Known security-relevant design facts

These are **documented behaviours**, not undisclosed bugs. Reports about them are
welcome only if you can show impact beyond what is described here.

1. **The API is not authenticated by default.** `X-API-Key` enforcement (`require_api_key`)
   activates only in production-like environments, and several routers — `dashboard`,
   `account`, `portfolio`, `analytics`, `marketdata` — carry no auth dependency at all.
   They return balances, positions, orders and PnL. `api_host` defaults to `0.0.0.0`.
   **Do not expose the API to an untrusted network.** `docker-compose.yml` binds to
   `127.0.0.1`; use an SSH tunnel or an authenticating reverse proxy for remote access.
2. **Redis ships with no password.** The loopback binding is the boundary.
3. **`QF_TRADING__LIVE_CONFIRMATION=I_UNDERSTAND_THE_RISK` is a public constant, not a
   secret.** It is an intent gate against accidental live arming, and nothing more.
4. **Secrets are `SecretStr` and are redacted from every log line**, and blank values are
   coerced to absent so an empty key can never be used to sign a request. Treat log files
   as sensitive regardless.
5. **Live trading is disabled by default** and requires three independent, deliberate
   settings changes.

## Operator guidance

- Grant exchange API keys **trade permission only — never withdrawal** — and IP-restrict
  them.
- Use **separate keys** for demo and mainnet. QuantFlow stores them in separate variables
  precisely so one cannot be used in place of the other.
- Never commit `.env`. It is gitignored; verify with `git check-ignore .env`.
- Rotate any key that has ever appeared in a file, a log, a screenshot, a terminal
  recording or a chat message.
- Run demo first. Run it long enough to see a losing streak.
- Keep dependencies current: `uv pip install --upgrade`, `npm audit fix`.

## Supported versions

Pre-1.0. **Only the latest release receives security fixes.** No backports.

| Version | Supported |
|---|---|
| latest release | ✅ |
| everything older | ❌ |
