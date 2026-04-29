## Repository Purpose
This repo benchmarks LLM inference: utilization, memory, power, latency, throughput.
Forked from https://github.com/ml-energy/benchmark.
Main entry point: `mlenergy/llm/benchmark.py`.

## Environment
- Python version is pinned in `.python-version`.
- Dependencies are managed with `uv` (`pyproject.toml` + `uv.lock`).
- Secrets/keys live in `.env`; never read or echo their contents and never commit them.

## Guidelines
### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Safety Rules
- Never work on the main branch
- Don't run jobs that require a GPU without explicit approval.
- Ask before modifying dependency files (`requirements.txt`, `pyproject.toml`).
- Never commit secrets or example `.env` contents.
- Don't perform destructive git operations (force-push, history rewrite, branch
  deletion, `git clean -fdx`) without explicit approval.

## When in Doubt
- For questions about benchmark methodology, datasets, or runtime: read `./claude/run-bench.md`.
- If a doc and the code disagree, flag it rather than silently picking one.
- For code writing guidelines: read `./claude/coding-style.md`

## Specific excecutions
- For benchmarking a model follow `claude/run-bench.md`.
- For post-run interpretation of results follow `claude/analyze-results.md`.
- Ignore files in `claude/.claudeignore` to avoid redundant work
