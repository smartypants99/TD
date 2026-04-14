"""Training data generator — the model creates its own training data with full COT."""

import hashlib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from ..utils.config import GeneratorConfig
from ..utils.model_loader import ModelLoader
from ..diagnostics.engine import WeaknessReport


@dataclass
class ReasoningStep:
    """A single step in a chain-of-thought."""
    step_number: int
    content: str
    justification: str
    assumptions: list[str] = field(default_factory=list)


@dataclass
class TrainingSample:
    """A single training sample with full chain-of-thought."""
    prompt: str
    response: str
    reasoning_chain: list[ReasoningStep] = field(default_factory=list)
    target_weakness: str = ""
    domain: str = ""
    verified: bool = False
    verification_notes: str = ""
    confidence: float = 0.0
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and self.prompt:
            # Include reasoning chain in hash — two samples with the same prompt
            # and conclusion but different reasoning should both be kept.
            chain_text = "|".join(s.content for s in self.reasoning_chain) if self.reasoning_chain else ""
            self.content_hash = hashlib.md5(
                f"{self.domain}:{self.target_weakness}:{self.prompt}{chain_text}{self.response}".encode()
            ).hexdigest()

    def to_training_format(self) -> dict:
        """Convert to format suitable for fine-tuning."""
        full_response = self._build_full_response()
        return {
            "prompt": self.prompt,
            "completion": full_response,
            "metadata": {
                "domain": self.domain,
                "target_weakness": self.target_weakness,
                "num_reasoning_steps": len(self.reasoning_chain),
                "verified": self.verified,
            },
        }

    def _build_full_response(self) -> str:
        """Build the complete response including chain-of-thought."""
        parts = []
        for i, step in enumerate(self.reasoning_chain, 1):
            # Renumber steps sequentially — the model may have produced gaps
            # (e.g., steps 1, 2, 5) and training on gapped numbers teaches
            # the model to skip steps in its output.
            # For code domain: preserve newlines (collapsing destroys Python syntax).
            # For non-code: collapse multi-line content to single line — embedded
            # newlines can contain text resembling step headers.
            if self.domain == "code":
                # Strip only numbered-list-like patterns that mimic step headers
                content = re.sub(r'(?m)^\d+[.)]\s+', '', step.content).strip()
            else:
                content = " ".join(step.content.split())
            # Strip leading step-header text from content — the model's original
            # "Step 3: x = 5" would produce "Step 1: Step 3: x = 5" after renumbering,
            # teaching the model to emit double step headers.
            content = re.sub(r'^(?:step\s*\d+\s*[.):–—-]\s*)', '', content, flags=re.IGNORECASE)
            parts.append(f"Step {i}: {content}")
            # Match the generation prompt order: Step → Justification → Assumptions.
            # Training data must use the same format the model was prompted with,
            # otherwise the model learns a different output structure than it generates.
            if step.justification and step.justification != "implicit":
                parts.append(f"  Justification: {step.justification}")
            if step.assumptions:
                parts.append(f"  Assumptions: {'; '.join(step.assumptions)}")
        parts.append(f"\nConclusion: {self.response}")
        return "\n".join(parts)


# Regex patterns for parsing model output — handles many format variants
_STEP_PATTERNS = [
    re.compile(r'^step\s*(\d+)[.):\-–—]\s*(.*)', re.IGNORECASE),           # "Step 1: ..." / "Step 1 - ..." (dash/em-dash common in model output)
    re.compile(r'^(\d{1,2})[.):\-–—]\s+(.*)', re.IGNORECASE),              # "1. ..." / "1) ..." / "1: ..." / "1 - ..." (bare number, max 2 digits)
    re.compile(r'^\*\*step\s*(\d+)\*\*[.):]*\s*(.*)', re.IGNORECASE),     # "**Step 1**: ..."
    re.compile(r'^(?:first|second|third|fourth|fifth)(?=[,.:;)\s])', re.IGNORECASE),  # "First, ..." but not "Firstly..." or "First principles..."
]

_ORDINAL_MAP = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}

_JUSTIFICATION_PREFIXES = ("justification:", "because:", "reason:", "this is because",
                           "this follows because", "this works because", "the reason is")

_ASSUMPTION_PREFIXES = ("assumption:", "assumptions:", "assuming:", "we assume",
                        "given that:", "premise:")

# Solution contamination patterns — models often ignore "output ONLY the problem"
# and include answers. Compiled once at module level (not per-call).
_CONTAM_PATTERNS = re.compile(
    r'(?:\n|(?<=[.?!;] ))(?:solution|answer|step\s*1[.:)]|first,\s*(?:we|let|I)|'
    r'let\'s solve|to solve this|hint:|the answer|we can solve|let me solve)',
    re.IGNORECASE,
)


class DataGenerator:
    """Forces the model to generate training data with exhaustive reasoning."""

    def __init__(self, config: GeneratorConfig, model_loader: ModelLoader):
        self.config = config
        self.model = model_loader
        self._seen_hashes: set[str] = set()
        self._custom_solution_template: str | None = None  # from escalation

    def set_custom_solution_template(self, template: str):
        """Set a model-improved solution template (from generation escalation)."""
        self._custom_solution_template = template

    def generate_for_weaknesses(self, weaknesses: list[WeaknessReport]) -> list[TrainingSample]:
        """Generate training data targeting identified weaknesses."""
        # Reset dedup set each cycle — cross-cycle dedup is unnecessary because
        # problems are generated fresh each time, and it blocks legitimate samples
        # as the set grows over 100+ cycles.
        self._seen_hashes.clear()
        all_samples = []
        for weakness in weaknesses:
            samples = self._generate_for_weakness(weakness)
            # Deduplicate
            for s in samples:
                if s.content_hash not in self._seen_hashes:
                    self._seen_hashes.add(s.content_hash)
                    all_samples.append(s)
        return all_samples

    def _generate_for_weakness(self, weakness: WeaknessReport) -> list[TrainingSample]:
        """Generate training samples for a specific weakness.

        Phase 1: Batch-generate problems. Phase 2: Solve each with retries.
        Sample count scales with severity — minor weaknesses get fewer samples.
        """
        # Scale sample count by severity: severe weaknesses (0.5+) get full count,
        # mild weaknesses (0.1) get 20% of full count. Minimum 10 samples.
        scale = max(0.2, min(1.0, weakness.severity * 2))
        num_samples = max(10, int(self.config.samples_per_weakness * scale))

        # Batch problem generation
        problem_prompts = [
            self._build_problem_generation_prompt(weakness, i)
            for i in range(num_samples)
        ]
        batch_size = 16
        problems = []
        for start in range(0, len(problem_prompts), batch_size):
            batch = problem_prompts[start:start + batch_size]
            responses = self.model.generate_batch(batch, max_new_tokens=512,
                                                   temperature=self.config.temperature,
                                                   top_p=self.config.top_p)
            problems.extend(responses)

        # Strip solution contamination from generated problems.
        clean_problems = []
        for p in problems:
            p = p.strip()
            if not p:
                continue
            m = _CONTAM_PATTERNS.search(p)
            if m and m.start() > 20:  # keep at least 20 chars of the problem
                p = p[:m.start()].strip()
            clean_problems.append(p)

        # Phase 2a: Batch-solve first attempts for all problems at once
        valid_problems = [p for p in clean_problems if p.strip()]
        if not valid_problems:
            return []
        first_prompts = [self._build_solution_prompt(p, weakness) for p in valid_problems]
        first_solutions = []
        for start in range(0, len(first_prompts), batch_size):
            batch = first_prompts[start:start + batch_size]
            responses = self.model.generate_batch(batch, max_new_tokens=2048,
                                                   temperature=0.3)
            first_solutions.extend(responses)

        # Phase 2b: Parse results; retry individually only for failures
        samples = []
        for problem, solution in zip(valid_problems, first_solutions):
            chain = self._parse_reasoning_chain(solution)
            if len(chain) >= self.config.min_reasoning_steps:
                conclusion = self._extract_conclusion(solution)
                if conclusion:
                    samples.append(TrainingSample(
                        prompt=problem, response=conclusion,
                        reasoning_chain=chain,
                        target_weakness=f"{weakness.domain}/{weakness.subdomain}",
                        domain=weakness.domain,
                    ))
                    continue
            # Fall back to sequential retry for insufficient chains
            sample = self._solve_problem_retries_only(problem, weakness, solution, chain)
            if sample:
                samples.append(sample)

        return samples

    def _solve_problem_retries_only(
        self, problem: str, weakness: WeaknessReport,
        prev_solution: str, prev_chain: list[ReasoningStep] | None,
    ) -> TrainingSample | None:
        """Retry solving after a batched first attempt failed to produce enough steps."""
        all_solutions = [prev_solution]  # keep all attempts for conclusion fallback
        solution = prev_solution
        reasoning_chain = prev_chain

        # Start from attempt 1 (attempt 0 was the batched first try)
        for attempt in range(1, self.config.max_retries):
            prompt = self._build_elaboration_prompt(
                problem, solution, weakness,
                len(reasoning_chain) if reasoning_chain else 0,
            )
            temp = min(0.3 + attempt * 0.15, 0.8)
            solution = self.model.generate(prompt, max_new_tokens=2048, temperature=temp)
            all_solutions.append(solution)
            reasoning_chain = self._parse_reasoning_chain(solution)
            if len(reasoning_chain) >= self.config.min_reasoning_steps:
                break

        if not reasoning_chain or len(reasoning_chain) < self.config.min_reasoning_steps:
            return None

        # Try extracting conclusion from the best solution first, then fall
        # back to earlier attempts — the original batched solution often has
        # a clear conclusion even when its step count was insufficient.
        conclusion = self._extract_conclusion(solution)
        if not conclusion:
            for prev in reversed(all_solutions[:-1]):
                conclusion = self._extract_conclusion(prev)
                if conclusion:
                    break
        if not conclusion:
            return None

        return TrainingSample(
            prompt=problem, response=conclusion,
            reasoning_chain=reasoning_chain,
            target_weakness=f"{weakness.domain}/{weakness.subdomain}",
            domain=weakness.domain,
        )

    def _build_problem_generation_prompt(self, weakness: WeaknessReport, index: int) -> str:
        """Build prompt for problem generation. Uses failed examples as context."""
        base = (
            f"You are being tested on {weakness.domain}, specifically {weakness.subdomain}. "
            f"Your current failure rate in this area is {weakness.severity:.0%}.\n\n"
            f"Generate a challenging {weakness.subdomain} problem (#{index + 1}) "
            f"that requires multi-step reasoning to solve. "
            f"The problem should be self-contained and have a definite correct answer.\n"
        )

        # Include examples of failed questions for context (cycle through available evidence).
        # Filter to evidence dicts that have a 'question' key — regression weaknesses
        # or checkpoint-restored evidence may have incomplete dicts.
        usable_evidence = [e for e in weakness.evidence if e.get('question')]
        if usable_evidence:
            failed = usable_evidence[index % len(usable_evidence)]
            # Use .get() for safety — evidence dicts may be incomplete after
            # checkpoint restore or from model-generated diagnostic questions.
            # Show the failed question and expected answer, but NOT the model's
            # wrong answer. Including "Your answer: [wrong]" biases generation —
            # the model is more likely to produce problems where similar wrong
            # reasoning applies, creating a feedback loop of the same mistake type.
            base += (
                f"\nHere is an example of a question you previously got wrong:\n"
                f"Q: {failed.get('question', 'N/A')}\n"
                f"Expected answer: {failed.get('expected', 'N/A')}\n\n"
                f"Generate a DIFFERENT problem that tests the same type of understanding.\n"
            )

        base += "\nOutput ONLY the problem statement, nothing else."
        return base

    def _build_solution_prompt(self, problem: str, weakness: WeaknessReport) -> str:
        """Build prompt forcing exhaustive chain-of-thought.

        Uses model-improved template if available from generation escalation.
        """
        if self._custom_solution_template:
            try:
                return self._custom_solution_template.format(
                    problem=problem,
                    domain=weakness.domain,
                    subdomain=weakness.subdomain,
                )
            except (KeyError, ValueError) as e:
                # Log once so the user knows the template is broken — without this,
                # the template silently fails every cycle, wasting the inference
                # call that generated it in _escalate_generation.
                logger.warning(f"Custom solution template failed ({e}) — using default")
                self._custom_solution_template = None  # stop retrying

        return (
            f"Solve the following {weakness.domain}/{weakness.subdomain} problem.\n\n"
            f"PROBLEM:\n{problem}\n\n"
            f"MANDATORY FORMAT:\n"
            f"You MUST structure your answer as numbered steps.\n"
            f"For EACH step:\n"
            f"  Step N: [what you are doing]\n"
            f"  Justification: [why this step follows logically]\n"
            f"  Assumptions: [any assumptions made, or 'none']\n\n"
            f"Rules:\n"
            f"- Minimum {self.config.min_reasoning_steps} steps\n"
            f"- NO skipping steps. Every logical leap must be explicit.\n"
            f"- If uncertain about any step, state your uncertainty.\n"
            f"- End with: Conclusion: [your final answer]\n\n"
            f"BEGIN:"
        )

    @staticmethod
    def _truncate_at_boundary(text: str, max_len: int) -> str:
        """Truncate text at a sentence or line boundary, not mid-word."""
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        # Prefer line boundary
        last_nl = truncated.rfind("\n")
        if last_nl > max_len // 2:
            return truncated[:last_nl] + "..."
        # Prefer sentence boundary
        for sep in (". ", "! ", "? "):
            last_sep = truncated.rfind(sep)
            if last_sep > max_len // 2:
                return truncated[:last_sep + 1] + "..."
        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_len // 2:
            return truncated[:last_space] + "..."
        return truncated + "..."

    def _build_elaboration_prompt(self, problem: str, prev_solution: str, weakness: WeaknessReport, prev_steps: int) -> str:
        """Ask for more detail when previous attempt was insufficient."""
        parts = [
            f"You are solving a {weakness.domain}/{weakness.subdomain} problem.\n\n"
            f"Your previous solution had only {prev_steps} reasoning steps. "
            f"This is insufficient — you need at least {self.config.min_reasoning_steps}.\n\n"
            f"PROBLEM:\n{problem}\n\n"
        ]
        if prev_solution and prev_solution.strip():
            truncated = self._truncate_at_boundary(prev_solution, 1000)
            parts.append(f"YOUR PREVIOUS ATTEMPT:\n{truncated}\n\n")
        parts.append(
            f"Solve again with MORE DETAIL. Break every inference into its own step.\n"
            f"Use this exact format:\n"
            f"  Step 1: [content]\n"
            f"  Justification: [why]\n"
            f"  Assumptions: [what you assume]\n\n"
            f"Minimum {self.config.min_reasoning_steps} steps. End with 'Conclusion:'.\n\n"
            f"BEGIN:"
        )
        return "".join(parts)

    def _parse_reasoning_chain(self, solution: str) -> list[ReasoningStep]:
        """Parse solution into reasoning steps. Handles multiple format variants."""
        steps = []
        current_step_num = None
        current_content = []
        current_justification = ""
        current_assumptions = []

        in_justification = False  # tracks multi-line justifications

        def _flush_step():
            nonlocal current_step_num, current_content, current_justification, current_assumptions, in_justification
            in_justification = False
            if current_step_num is not None and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    steps.append(ReasoningStep(
                        step_number=current_step_num,
                        content=content,
                        justification=current_justification or "implicit",
                        assumptions=current_assumptions,
                    ))
            current_step_num = None
            current_content = []
            current_justification = ""
            current_assumptions = []

        for line in solution.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Strip leading bullet markers (-, *, •) so "- Step 1:" is handled.
            # Allow optional space after bullet — models produce both "- Step 1:"
            # and "-Step 1:" (no space).
            if stripped[0] in "-*•" and len(stripped) > 1:
                if stripped[1] == " ":
                    stripped = stripped[2:].strip()
                else:
                    stripped = stripped[1:].strip()
                if not stripped:
                    continue

            # Strip markdown formatting (bold, italic, headers, blockquotes)
            # before matching — models wrap conclusions in "**Conclusion:**",
            # "## Final Answer:", "> Conclusion:" etc.
            stripped = re.sub(r'^[#>]+\s*', '', stripped).strip()
            stripped = re.sub(r'^\*{1,2}(.*?)\*{1,2}$', r'\1', stripped).strip()
            lower = stripped.lower()

            # Check for conclusion markers — stop parsing steps.
            # Only use unambiguous markers: "answer:" and "thus," are too common
            # inside reasoning steps ("Answer: we need...", "Thus, we substitute...")
            if any(lower.startswith(m) for m in
                   ("conclusion:", "final answer:", "in conclusion")):
                _flush_step()
                break

            # Check for step header
            step_num = self._detect_step_header(stripped)
            if step_num is not None:
                _flush_step()
                current_step_num = step_num
                # Extract content after the header
                for pattern in _STEP_PATTERNS[:3]:
                    m = pattern.match(stripped)
                    if m:
                        content_after = m.group(2).strip() if m.lastindex >= 2 else ""
                        if content_after:
                            current_content.append(content_after)
                        break
                else:
                    # Ordinal case — strip the ordinal word prefix ("First, ..." → "...")
                    ordinal_content = stripped
                    for word in _ORDINAL_MAP:
                        if stripped.lower().startswith(word):
                            ordinal_content = stripped[len(word):].lstrip(" ,:")
                            break
                    current_content.append(ordinal_content or stripped)
                continue

            # Check for justification
            if any(lower.startswith(p) for p in _JUSTIFICATION_PREFIXES):
                current_justification = stripped.split(":", 1)[-1].strip() if ":" in stripped else stripped
                in_justification = True
                continue

            # Check for assumptions (ends justification continuation)
            if any(lower.startswith(p) for p in _ASSUMPTION_PREFIXES):
                in_justification = False
                assumptions_text = stripped.split(":", 1)[-1].strip() if ":" in stripped else stripped
                current_assumptions.extend(
                    a.strip() for a in re.split(r'[;,]', assumptions_text) if a.strip()
                )
                continue

            # Multi-line justification continuation — if the previous line was a
            # justification prefix and this line doesn't match any other prefix,
            # it's a continuation (e.g., "Justification: this is because\nthe formula...")
            # Limit to 1 continuation line — unlimited continuation misclassifies
            # content as justification when models write prose after "Justification:".
            if in_justification and current_step_num is not None:
                current_justification += " " + stripped
                in_justification = False  # only one continuation line
                continue

            in_justification = False
            # Regular content line
            if current_step_num is not None:
                current_content.append(stripped)

        _flush_step()
        return steps

    def _detect_step_header(self, line: str) -> int | None:
        """Detect if a line is a step header, return step number or None."""
        for pattern in _STEP_PATTERNS[:3]:
            m = pattern.match(line)
            if m:
                try:
                    num = int(m.group(1))
                    # Reject implausibly high step numbers (likely math, not step headers)
                    if num > 50:
                        continue
                    return num
                except (ValueError, IndexError):
                    pass

        # Check ordinals — require a delimiter after the word so "First principles"
        # isn't mistaken for step 1 (only "First," / "First:" / "First " etc.)
        lower = line.lower()
        for word, num in _ORDINAL_MAP.items():
            if lower.startswith(word) and len(lower) > len(word) and lower[len(word)] in ",.:;) \t":
                return num

        return None

    def _extract_conclusion(self, solution: str) -> str:
        """Extract the conclusion from the solution."""
        lower = solution.lower()

        # Prefer unambiguous markers first, then fall back to broader ones.
        # Two tiers: tier 1 is always safe; tier 2 only used if NO tier 1 marker exists.
        tier1 = ["conclusion:", "final answer:"]
        # "therefore " (with trailing space) was too broad — it matches mid-sentence
        # usage like "we can therefore compute". Only match "therefore" with
        # delimiters that signal conclusion context (, : \n).
        tier2 = ["answer:", "therefore,", "therefore:", "therefore\n"]
        tier1_present = any(m in lower for m in tier1)

        for marker in (tier1 if tier1_present else tier1 + tier2):
            if marker in lower:
                idx = lower.rindex(marker)  # use last occurrence
                text = solution[idx + len(marker):].strip()
                # Take until end or next section
                end_markers = ["\n\n", "\nStep ", "\n**"]
                for em in end_markers:
                    if em in text:
                        text = text[:text.index(em)]
                if text.strip():
                    return text.strip()

        # Fallback: last non-empty line that isn't a metadata line.
        # Without this filter, "Justification: ..." or "Assumptions: ..." lines
        # get returned as the "conclusion", which is nonsensical training data.
        meta_prefixes = _JUSTIFICATION_PREFIXES + _ASSUMPTION_PREFIXES
        lines = [l.strip() for l in solution.split("\n") if l.strip()]
        for line in reversed(lines):
            if not any(line.lower().startswith(p) for p in meta_prefixes):
                # Strip step header prefix if present — "Step 3: x = 5" → "x = 5"
                # so the conclusion doesn't redundantly repeat "Step N:" in training.
                for pattern in _STEP_PATTERNS[:3]:
                    m = pattern.match(line)
                    if m:
                        content = m.group(2).strip() if m.lastindex >= 2 else ""
                        if content:
                            return content
                        # Empty step header (e.g., "Step 3:") — skip it,
                        # it's not a valid conclusion.
                        break
                else:
                    # Not a step header — return the line as-is
                    return line
        # Final fallback: return last non-metadata line. lines[-1] might itself
        # be metadata (e.g., "Assumptions: none") if every non-metadata line was
        # an empty step header that triggered the break above. Filter again.
        for line in reversed(lines):
            if not any(line.lower().startswith(p) for p in meta_prefixes):
                return line
        return ""
