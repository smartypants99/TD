# AI Time Dilation Runtime — Design Spec

## Overview

A general-purpose runtime that makes a local open-source LLM (Qwen family) do arbitrarily more cognitive work per task. The user sets a dilation factor and a wall-clock time budget. The dilation factor is always honored — it determines how many improvement cycles run. The budget is advisory — the system runs until the dilation factor is satisfied, reporting progress along the way. The user can interrupt at any time to take the current best result.

**Core invariant:** The base model is never modified, compressed, or degraded. Every cycle uses the full model at full quality.

**Scaling invariant:** The dilation factor has no coded upper limit. 2x, 1000x, 1,000,000,000x — the system will attempt any factor. More dilation = more cycles and branches = better output.

## Concepts

- **Wall-clock time**: Real-world seconds the user is willing to wait.
- **Dilation factor**: Multiplier on subjective effort. Factor 100x on a 5s budget = 500 subjective seconds of work.
- **Improvement cycle**: One pass of branch-score-select. The atomic unit of "subjective time."
- **Branch**: Generate N variant outputs from the current best.
- **Score**: Use the model itself to evaluate variants against the original prompt.
- **Select**: Keep the best variant, discard the rest.

## Architecture

### 1. Inference Engine

- Loads Qwen model via vLLM with speculative decoding
- Small draft model (e.g., Qwen2.5-0.5B) proposes tokens, large model (e.g., Qwen2.5-72B) verifies in batches
- Exposes: `generate(prompt: str, max_tokens: int) -> str`
- Configurable: model path, draft model path, quantization, device

### 2. Dilation Controller

- Input: prompt, dilation factor, wall-clock budget (advisory)
- Calculates target cycles from dilation factor: `target_cycles = dilation_factor` (1x = baseline, 2x = baseline + 1 refinement, 100x = baseline + 99 refinement cycles)
- Runs until target cycles are completed — budget is advisory, not a hard cutoff
- User can Ctrl+C at any time to get the current best result (interruptible)
- Reports: cycles completed, estimated remaining time, current quality score
- Auto-reduces branch factor when hardware is slow (falls back to branch=1 on constrained hardware)
- **No hard cap on cycles or dilation factor**

### 3. Improvement Engine

- **Branch phase**: Takes current best output + original prompt, generates N variants
  - Variants are produced by prompting the model with different improvement directives
  - Directives are task-aware: the model first classifies the task type (code, prose, reasoning, creative, etc.) and generates appropriate improvement directives dynamically
  - Fallback directive set for code tasks: fix bugs, handle edge cases, optimize, add tests, refactor, explore alternatives, stress-test, improve error handling, add features
  - Fallback directive set for non-code tasks: improve clarity, strengthen arguments, add examples, consider counterpoints, improve structure, refine tone, deepen analysis
  - After built-in directives are exhausted, the model generates its own next directive — ensuring infinite productive work
  - Branch factor N is configurable (default: 3)
- **Score phase**: Model evaluates each variant on a 0-100 scale against the original prompt
  - Scoring prompt includes: correctness, completeness, quality, elegance
  - Score is deterministic per variant (temperature 0)
  - Known limitation: self-scoring has bias. Mitigated by: using structured rubrics, comparing against the original prompt (not just "is this good?"), and tracking score deltas rather than absolute scores
- **Select phase**: Highest-scoring variant becomes the new best
  - If no variant scores higher than current best, current best is retained
  - Convergence detection: if no improvement for 5 consecutive cycles, shift to a different directive category. If all categories exhausted with no improvement, log "converged at cycle N" but continue if dilation factor demands more cycles
  - Checkpointing: best result is saved to disk every 10 cycles, so crashes don't lose all progress

### 4. CLI Interface

```
timedilate "Build a Roblox obby game" --budget 5s --factor 100x
```

- `--budget`: Wall-clock time budget (e.g., 5s, 30s, 5m)
- `--factor`: Dilation factor (e.g., 2x, 100x, 1000000x)
- `--model`: Path to main model (default: auto-detect)
- `--draft-model`: Path to draft model for speculative decoding
- `--branches`: Branch factor per cycle (default: 3)
- `--output`: Output file path (default: stdout)
- `--verbose`: Show cycle-by-cycle progress

Output during execution:
- Progress bar showing cycles completed / estimated total
- Current quality score
- Elapsed wall-clock time / budget

## Infinite Scaling Strategy

The system is infinitely scalable because:

1. **No hard cap**: Dilation factor is an arbitrary number. The system calculates cycles from it and runs them.
2. **Self-directing improvement**: The model generates its own next improvement task when the built-in directive list is exhausted. It will always find more to do.
3. **Hardware-independent**: On a single GPU, branches run sequentially. On multiple GPUs, they run in parallel. Either way, the dilation factor works — it just changes how many wall-clock seconds are needed.
4. **Diminishing returns are acceptable**: At extreme factors, improvements per cycle will shrink. But the system never stops trying, and even tiny improvements compound over thousands of cycles.

## Tech Stack

- Python 3.11+
- vLLM — inference with speculative decoding
- Qwen2.5 family — primary model family (any size)
- Click — CLI framework
- Rich — terminal UI (progress bars, formatting)

## Milestone Plan

### M1: Prove 2x (Week 1)
- Load Qwen model with speculative decoding via vLLM
- Single improvement cycle (generate + refine once)
- Measure: human eval — run same prompt with dilation=1 and dilation=2, compare outputs side by side
- CLI with basic args

### M2: Prove 5x (Week 2)
- Multiple sequential improvement cycles
- Scoring system operational
- Directive rotation
- Progress display

### M3: Prove 10x+ (Week 3)
- Branching (multiple variants per cycle)
- Branch-score-select loop
- Convergence detection
- Self-directing improvement tasks

### M4: Scale to 100x+ (Week 4)
- Performance optimization
- Adaptive cycle timing
- Web UI (basic)

### M5: Scale to 1000x+ and beyond
- Advanced branching strategies
- Multi-GPU support
- Web UI (full)

## Future Integration Points

- Architecture is deliberately modular to support future integration with self-recursive improvement systems
- Inference engine is swappable (could replace vLLM with another backend)
- Improvement engine directives are extensible

## Context Window Management

For long outputs or many cycles, the accumulated context (original prompt + current best + improvement directive) may approach the model's context limit. Strategy:

- Only pass the original prompt + current best output + current directive to the model (not full history of all cycles)
- If current best output exceeds 75% of context window, summarize it before passing to next cycle
- Cycle history (scores, directives used) is tracked in metadata, not in the model's context

## Non-Goals (for now)

- Model weight modification at runtime
- Distributed inference across multiple machines
- Training or fine-tuning
