"""Verifier — checks every step of every reasoning chain for logical validity.

The key insight: verification is easier than generation. A step-by-step proof
can be checked mechanically even if producing it requires creativity.

This verifier operates at multiple levels:
1. Structural: does the chain have proper form?
2. Referential: does each step build on prior steps?
3. Logical: do the inferences actually follow?
4. Consistency: does the chain contradict itself?
5. Completeness: does the conclusion follow from the chain?
"""

import logging
import re
from dataclasses import dataclass, field

from ..utils.config import VerifierConfig

logger = logging.getLogger(__name__)
from ..generator.data_generator import TrainingSample, ReasoningStep


@dataclass
class StepVerification:
    """Verification result for a single reasoning step."""
    step_number: int
    valid: bool
    issues: list[str] = field(default_factory=list)
    confidence: float = 0.0
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Full verification result for a training sample."""
    accepted: bool
    step_results: list[StepVerification] = field(default_factory=list)
    overall_confidence: float = 0.0
    rejection_reasons: list[str] = field(default_factory=list)
    total_checks: int = 0
    checks_passed: int = 0


# Common logical connectors that indicate inferential structure.
# Includes both logical connectors AND mathematical/procedural language
# (math proofs use "substituting", "applying" more than "therefore").
# Multi-word markers checked via substring, single-word markers checked via
# word-boundary regex to avoid "so" matching "also", "since" matching "convincing".
_INFERENCE_MARKERS_MULTI = {
    "given that", "it follows", "means that", "leads to", "results in",
    "we can conclude", "this shows", "which gives",
    "by definition", "from the formula", "using the",
    "we get", "we obtain", "we find", "this is because",
    # Math/science-specific inference language — without these, valid justifications
    # like "by the chain rule" or "by induction" fail the inferential check.
    # Avoid overly broad 2-word phrases ("by the", "from the") that match
    # any English text — require 3+ words or specific rule names.
    "according to", "by induction", "by contradiction",
    "we know that", "we have that", "which means that",
    "this gives us", "by the formula", "by the rule",
    "by the definition", "by the theorem", "by the chain",
    "from the equation", "from the definition", "from the result",
}
_INFERENCE_MARKERS_SINGLE = re.compile(
    r'\b(?:therefore|thus|hence|consequently|because|since|implies|'
    r'substituting|applying|evaluating|computing|simplifying|'
    r'rearranging|expanding|factoring|yields)\b'
)

# Contradiction indicators
_RE_STEP_REF = re.compile(r'step\s*\d')
_RE_PRIOR_REF = re.compile(r'(?:above|previous|earlier)')
_RE_MATH_TOKEN = re.compile(r'\d|^[a-z]\^')

_REF_STOPWORDS = frozenset({
    "this", "that", "with", "from", "have", "which", "where", "there",
    "their", "about", "would", "could", "should", "been", "were", "they",
    "them", "then", "than", "what", "when", "will", "each", "made",
    "after", "before", "into", "more", "also", "very", "just", "only",
    "step", "value", "result", "number", "since", "using", "given",
    "first", "second", "third", "next", "above", "below", "need",
    "know", "find", "show", "solve", "answer", "problem", "equation",
    "total", "equal", "means", "case", "take", "make", "like", "some",
})

# Short math/science terms that are meaningful for reference tracking despite
# being under the 5-char threshold. Without these, a step that references "sin",
# "log", or "sum" from a prior step fails the reference check.
_MATH_SHORT_WORDS = frozenset({
    "sin", "cos", "tan", "log", "exp", "sum", "max", "min", "gcd", "mod",
    "lim", "int", "det", "dim", "div", "abs", "arg", "inf", "sup",
})

_NEGATION_PAIRS_RAW = [
    ("is", "is not"), ("are", "are not"), ("can", "cannot"),
    ("increase", "decrease"), ("increasing", "decreasing"),
    ("more", "less"), ("greater", "smaller"), ("positive", "negative"),
    ("always", "never"), ("possible", "impossible"),
]
# Precompute word sets — these were reconstructed via set(pos.split()) on every
# step × prior × pair iteration (400+ times per sample). Frozen at module level.
_NEGATION_PAIRS = [
    (frozenset(pos.split()), frozenset(neg.split()))
    for pos, neg in _NEGATION_PAIRS_RAW
]

# Words too common to signal meaningful context overlap in contradiction checks
_TRIVIAL_WORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are",
    "was", "were", "has", "have", "had", "will", "can", "its",
    # Reasoning words that inflate overlap without indicating logical connection
    "step", "value", "answer", "number", "result", "because", "therefore",
    "then", "since", "thus", "hence", "given", "find", "solve", "show",
    "know", "need", "note", "means", "equal", "total",
})

# Words from prompt templates that appear in nearly every question — not meaningful
# for checking whether a step references the original problem.
_TEMPLATE_WORDS = frozenset({
    "solve", "following", "problem", "reasoning", "steps",
    "explain", "answer", "shows", "prove", "given",
    "every", "using", "write", "which", "their", "about",
    "should", "would", "could", "being", "those", "these",
    "there", "where", "other", "after", "before", "between",
})


class Verifier:
    """Validates reasoning chains at structural, referential, and logical levels."""

    def __init__(self, config: VerifierConfig):
        self.config = config
        self._model_verifier = None  # set during escalation

    def set_model_verifier(self, model_loader):
        """Enable model-assisted verification (escalation phase)."""
        self._model_verifier = model_loader

    def verify_batch(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """Verify a batch of samples. Returns only accepted samples.

        Model-assisted verification is batched for efficiency — heuristic checks
        run first, then model verification runs in one batched call for all
        samples that passed heuristic checks.
        """
        # Phase 1: Heuristic verification (fast, no GPU)
        heuristic_results: list[tuple[TrainingSample, VerificationResult]] = []
        for sample in samples:
            result = self._verify_heuristic(sample)
            heuristic_results.append((sample, result))

        # Phase 2: Batched model verification for passing samples.
        # Wrapped in try/except — if model inference fails (OOM, etc.), we still
        # have valid heuristic results. Model verification is a bonus, not a gate.
        if self.config.use_model_verification and self._model_verifier:
            try:
                # Include borderline rejections (within 15% of threshold) so the
                # model can overturn false heuristic rejections. Pure heuristics
                # can't catch valid reasoning that happens to miss a keyword.
                borderline_threshold = self.config.min_confidence_for_accept * 0.85
                candidates = [(s, r) for s, r in heuristic_results
                              if r.overall_confidence >= borderline_threshold]
                if candidates:
                    prompts = []
                    for sample, _ in candidates:
                        chain_text = sample._build_full_response()
                        prompts.append(
                            f"You are a logic checker. Examine this reasoning chain for errors.\n\n"
                            f"PROBLEM:\n{sample.prompt}\n\n"
                            f"REASONING:\n{chain_text}\n\n"
                            f"Is every step logically valid? Does the conclusion follow? "
                            f"Answer ONLY 'VALID' or 'INVALID: <reason>'."
                        )
                    # Chunk to avoid OOM — verification prompts are long (~500 tokens each)
                    verify_batch_size = 16
                    responses = []
                    for start in range(0, len(prompts), verify_batch_size):
                        batch = prompts[start:start + verify_batch_size]
                        responses.extend(self._model_verifier.generate_batch(
                            batch, max_new_tokens=256, temperature=0.0,
                        ))
                    # Guard against length mismatch — if generate_batch returns
                    # fewer responses than prompts (OOM mid-batch), zip silently
                    # drops trailing candidates, leaving them with stale state.
                    if len(responses) != len(candidates):
                        logger.warning(
                            f"Model verification returned {len(responses)} responses for "
                            f"{len(candidates)} candidates — skipping model verification"
                        )
                        raise RuntimeError("response count mismatch")
                    for (sample, result), response in zip(candidates, responses):
                        result.total_checks += 1
                        resp_text = response.strip()
                        first_word = re.sub(r'[*`_]', '', resp_text.split()[0]).upper() if resp_text else ""
                        if first_word == "VALID":
                            result.checks_passed += 1
                        else:
                            # Preserve the model's reason — "INVALID: step 2 assumes X"
                            # is far more useful for debugging than a generic message.
                            model_reason = resp_text[:200] if resp_text else "no response"
                            result.rejection_reasons.append(f"Model verification: {model_reason}")
                        # Recompute confidence with model check included.
                        # Model can both downgrade (INVALID) and upgrade (VALID for
                        # borderline samples) the acceptance decision.
                        result.overall_confidence = result.checks_passed / max(result.total_checks, 1)
                        result.accepted = result.overall_confidence >= self.config.min_confidence_for_accept
            except Exception as e:
                logger.warning(f"Model-assisted verification failed ({type(e).__name__}): {e} — using heuristic results only")

        verified = []
        for sample, result in heuristic_results:
            if result.accepted and result.overall_confidence >= self.config.min_confidence_for_accept:
                sample.verified = True
                sample.confidence = result.overall_confidence
                sample.verification_notes = f"Passed {result.checks_passed}/{result.total_checks} checks"
                verified.append(sample)
            else:
                sample.verified = False
                sample.verification_notes = "; ".join(result.rejection_reasons)
        return verified

    def verify(self, sample: TrainingSample) -> VerificationResult:
        """Verify a single sample (public API, includes model verification)."""
        result = self._verify_heuristic(sample)
        if self.config.use_model_verification and self._model_verifier:
            # Include borderline samples (within 15% of threshold) — same policy
            # as verify_batch. Without this, single-sample verification is stricter
            # than batch verification, causing inconsistent accept/reject decisions.
            borderline_threshold = self.config.min_confidence_for_accept * 0.85
            if result.overall_confidence >= borderline_threshold:
                model_valid, model_reason = self._model_verify(sample)
                result.total_checks += 1
                if model_valid:
                    result.checks_passed += 1
                else:
                    result.rejection_reasons.append(f"Model verification: {model_reason}")
                result.overall_confidence = result.checks_passed / max(result.total_checks, 1)
                result.accepted = result.overall_confidence >= self.config.min_confidence_for_accept
        return result

    def _verify_heuristic(self, sample: TrainingSample) -> VerificationResult:
        """Heuristic verification without model-assisted check."""
        if not sample.reasoning_chain:
            return VerificationResult(
                accepted=False,
                rejection_reasons=["No reasoning chain present"],
            )

        step_results = []
        total_checks = 0
        checks_passed = 0

        for i, step in enumerate(sample.reasoning_chain):
            prior_steps = sample.reasoning_chain[:i]
            step_result = self._verify_step(step, prior_steps, sample.prompt)
            step_results.append(step_result)
            total_checks += len(step_result.checks_passed) + len(step_result.checks_failed)
            checks_passed += len(step_result.checks_passed)

            if not step_result.valid and self.config.reject_on_any_gap:
                return VerificationResult(
                    accepted=False,
                    step_results=step_results,
                    overall_confidence=checks_passed / max(total_checks, 1),
                    rejection_reasons=[
                        f"Step {step.step_number}: {'; '.join(step_result.issues)}"
                    ],
                    total_checks=total_checks,
                    checks_passed=checks_passed,
                )

        chain_issues, chain_checks_run = self._verify_chain(sample)
        total_checks += chain_checks_run
        chain_passed = chain_checks_run - len(chain_issues)
        checks_passed += chain_passed
        if chain_issues:
            if self.config.reject_on_any_gap:
                return VerificationResult(
                    accepted=False,
                    step_results=step_results,
                    overall_confidence=checks_passed / max(total_checks, 1),
                    rejection_reasons=chain_issues,
                    total_checks=total_checks,
                    checks_passed=checks_passed,
                )

        overall_confidence = checks_passed / max(total_checks, 1)
        accepted = overall_confidence >= self.config.min_confidence_for_accept

        # Collect all failure reasons for rejected samples — not just the
        # aggregate confidence. Without this, debugging why samples are rejected
        # requires reconstructing the checks from step_results.
        rejection_reasons = []
        if not accepted:
            rejection_reasons.append(f"Confidence {overall_confidence:.2f} below threshold {self.config.min_confidence_for_accept}")
            if chain_issues:
                rejection_reasons.extend(chain_issues)

        return VerificationResult(
            accepted=accepted,
            step_results=step_results,
            overall_confidence=overall_confidence,
            rejection_reasons=rejection_reasons,
            total_checks=total_checks,
            checks_passed=checks_passed,
        )

    def _verify_step(
        self,
        step: ReasoningStep,
        prior_steps: list[ReasoningStep],
        original_prompt: str,
    ) -> StepVerification:
        """Verify a single reasoning step with multiple check types."""
        passed = []
        failed = []
        # Use position in chain (no prior steps = first step) rather than
        # step_number, which depends on what the model wrote (could be 0, 2, etc.)
        is_first_step = len(prior_steps) == 0

        # Check 1: Content substance
        if step.content and len(step.content.strip()) >= 10:
            passed.append("content_substance")
        else:
            failed.append("content_substance")

        # Check 2: Justification exists and is meaningful.
        # Skip for first step — it typically restates the problem ("Given: ..."),
        # which is grounding, not inference. Check 4 already verifies grounding.
        # Penalizing both would double-count, and for 2-step chains (the minimum),
        # a single unjustified first step would push confidence below threshold.
        if self.config.check_logical_validity and not is_first_step:
            if step.justification and step.justification != "implicit" and len(step.justification) >= 10:
                passed.append("justification_present")
                # Check 2b: justification uses inferential language
                just_lower = step.justification.lower()
                has_inference = (any(m in just_lower for m in _INFERENCE_MARKERS_MULTI)
                                or bool(_INFERENCE_MARKERS_SINGLE.search(just_lower)))
                if has_inference:
                    passed.append("justification_inferential")
                else:
                    failed.append("justification_inferential")
            else:
                failed.append("justification_present")

        # Check 3: References prior steps (not for first step)
        if self.config.check_step_completeness and prior_steps:
            if self._step_references_prior(step, prior_steps):
                passed.append("references_prior")
            else:
                failed.append("references_prior")

        # Check 4: First step grounds in problem
        if self.config.check_assumption_grounding and is_first_step:
            if self._step_grounds_in_prompt(step, original_prompt):
                passed.append("grounded_in_prompt")
            else:
                failed.append("grounded_in_prompt")

        # Check 5: No internal contradiction within step
        if self._check_internal_consistency(step):
            passed.append("internal_consistency")
        else:
            failed.append("internal_consistency")

        # Check 6: Step doesn't contradict prior steps
        if prior_steps:
            if self._check_cross_step_consistency(step, prior_steps):
                passed.append("cross_step_consistency")
            else:
                failed.append("cross_step_consistency")

        # Check 7: Assumptions are explicit when present (first step only).
        # Only penalize if the step makes claims that need grounding but lists
        # no assumptions. Steps that simply restate the problem need none.
        if is_first_step:
            if step.assumptions:
                passed.append("assumptions_stated")
            elif step.justification and step.justification != "implicit":
                # Step has a justification → making a claim → should state assumptions
                failed.append("assumptions_stated")
            else:
                # Step is just restating/setting up — mark as passed to keep the
                # check count closer to non-first steps. Without this, first steps
                # run fewer checks (3-4 vs 6), making each check worth 25-33% of
                # confidence vs 17% for non-first steps — a single first-step failure
                # tanks overall confidence disproportionately.
                passed.append("assumptions_stated")

        valid = len(failed) == 0
        confidence = len(passed) / max(len(passed) + len(failed), 1)

        issues = [f"Failed check: {f}" for f in failed]

        return StepVerification(
            step_number=step.step_number,
            valid=valid,
            issues=issues,
            confidence=confidence,
            checks_passed=passed,
            checks_failed=failed,
        )

    def _step_references_prior(self, step: ReasoningStep, prior_steps: list[ReasoningStep]) -> bool:
        """Check if a step references concepts from ANY single prior step."""
        step_text = re.sub(r'[*`_]', '', f"{step.content} {step.justification}".lower())

        # Quick check: explicit step number or back-reference
        if _RE_STEP_REF.search(step_text) or _RE_PRIOR_REF.search(step_text):
            return True

        # Check each prior step individually — we need coherent overlap with
        # ONE step, not scattered words across many.
        # Optimization: only check the last 5 prior steps (most likely referents).
        step_words = set(re.findall(r'\S+', re.sub(r'[.,!?;:"\']', ' ', step_text)))
        for prior in prior_steps[-5:]:
            # Include prior's justification — it often contains the key terms
            # the next step references (e.g., "by the Pythagorean theorem").
            prior_text = prior.content
            if prior.justification and prior.justification != "implicit":
                prior_text = f"{prior_text} {prior.justification}"
            clean_prior = re.sub(r'[*`_]', '', prior_text.lower())
            prior_words = set(re.findall(r'\S+', re.sub(r'[.,!?;:"\']', ' ', clean_prior)))
            significant = {w for w in prior_words if len(w) > 4 and w not in _REF_STOPWORDS}
            # Include math-relevant short words (sin, cos, log, sum, max, min, etc.)
            # and tokens containing digits/operators — these are domain-specific terms
            # that strongly signal a real reference, not coincidence.
            significant |= {w for w in prior_words if _RE_MATH_TOKEN.search(w)}
            significant |= {w for w in prior_words if len(w) == 3 and w in _MATH_SHORT_WORDS}
            # Use word-set intersection, not substring — prevents partial matches
            overlap = len(significant & step_words)
            # Lower threshold for short steps (conclusion-like "Therefore F = 50")
            # that reference variables/values from prior steps but use few words.
            threshold = 2 if len(step_words) < 15 else 3
            if overlap >= threshold:
                return True

        return False

    def _step_grounds_in_prompt(self, step: ReasoningStep, prompt: str) -> bool:
        """Check if the first step references the original problem."""
        step_text = re.sub(r'[*`_]', '', f"{step.content} {step.justification}".lower())
        # Strip punctuation so "velocity." matches "velocity", "x=5" stays as-is
        step_words = set(re.findall(r'\S+', re.sub(r'[.,!?;:"\']', ' ', step_text)))
        prompt_clean = re.sub(r'[*`_]', '', prompt.lower())
        prompt_words = set(re.findall(r'\S+', re.sub(r'[.,!?;:"\']', ' ', prompt_clean)))
        significant = {w for w in prompt_words if len(w) > 4 and w not in _TEMPLATE_WORDS}
        # Include math tokens (x^3, 2x, etc.) and short domain words from the prompt
        significant |= {w for w in prompt_words if _RE_MATH_TOKEN.search(w)}
        significant |= {w for w in prompt_words if len(w) == 3 and w in _MATH_SHORT_WORDS}
        # Use word-set membership, not substring — "force" shouldn't match "reinforce"
        overlap = len(significant & step_words)
        # Lower threshold when the prompt has very few significant words (common
        # in math: "Find the derivative of x^3") — 1 match is sufficient grounding
        # when there are few words to match on.
        threshold = 1 if len(significant) <= 3 else 2
        return overlap >= threshold

    def _check_internal_consistency(self, step: ReasoningStep) -> bool:
        """Check if a step contradicts itself (bidirectional)."""
        # Include justification — it can contradict the content (e.g., content
        # says "x = 5" while justification says "since x cannot equal 5").
        parts = [step.content]
        if step.justification and step.justification != "implicit":
            parts.append(step.justification)
        text = " ".join(parts).lower()
        # Split on sentence-ending punctuation and semicolons, but NOT decimal points.
        # Naive [.!?] splits "3.14" into "3" and "14", creating nonsensical
        # fragments that false-positive on contradiction checks for math/science.
        # Semicolons are important — "x is positive; x is not positive" is a
        # contradiction that's invisible when treated as one sentence.
        sentences = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)|[!?;]', text) if s.strip()]

        if len(sentences) < 2:
            return True

        def _clean_words(s):
            return set(re.findall(r'\S+', re.sub(r'[.,!?;:"\'()\[\]{}+=<>]', ' ', s)))

        for i, s1 in enumerate(sentences):
            s1_words = _clean_words(s1)
            for s2 in sentences[i + 1:]:
                s2_words = _clean_words(s2)
                for pos_words, neg_words in _NEGATION_PAIRS:
                    for a, a_words, b, b_words in [(s1, s1_words, s2, s2_words), (s2, s2_words, s1, s1_words)]:
                        if pos_words <= a_words and neg_words <= b_words:
                            a_context = a_words - pos_words - _TRIVIAL_WORDS
                            b_context = b_words - neg_words - _TRIVIAL_WORDS
                            if len(a_context & b_context) >= 3:
                                return False
        return True

    def _check_cross_step_consistency(self, step: ReasoningStep, prior_steps: list[ReasoningStep]) -> bool:
        """Check if a step contradicts any recent prior step (bidirectional).

        Only checks the last 5 prior steps — contradicting step 1 from step 50
        is usually legitimate (problem setup vs. derived result), and checking
        all O(N) prior steps per step makes verification O(N²) per sample.
        """
        # Include justification text — a step's justification can contradict
        # prior content (or vice versa), and checking only content misses this.
        step_parts = [step.content]
        if step.justification and step.justification != "implicit":
            step_parts.append(step.justification)
        step_text = " ".join(step_parts).lower()
        step_words = set(re.findall(r'\S+', re.sub(r'[.,!?;:"\'()\[\]{}+=<>]', ' ', step_text)))

        for prior in prior_steps[-5:]:
            prior_parts = [prior.content]
            if prior.justification and prior.justification != "implicit":
                prior_parts.append(prior.justification)
            prior_text = " ".join(prior_parts).lower()
            prior_words = set(re.findall(r'\S+', re.sub(r'[.,!?;:"\'()\[\]{}+=<>]', ' ', prior_text)))
            for pos_words, neg_words in _NEGATION_PAIRS:
                # Direction 1: prior asserts, step negates
                if pos_words <= prior_words and neg_words <= step_words:
                    prior_context = prior_words - pos_words - _TRIVIAL_WORDS
                    step_context = step_words - neg_words - _TRIVIAL_WORDS
                    if len(prior_context & step_context) >= 3:
                        return False
                # Direction 2: prior negates, step asserts
                if neg_words <= prior_words and pos_words <= step_words:
                    prior_context = prior_words - neg_words - _TRIVIAL_WORDS
                    step_context = step_words - pos_words - _TRIVIAL_WORDS
                    if len(prior_context & step_context) >= 3:
                        return False
        return True

    def _verify_chain(self, sample: TrainingSample) -> tuple[list[str], int]:
        """Chain-level verification checks. Returns (issues, checks_run)."""
        issues = []
        checks_run = 0

        # Check 1: Must have a conclusion
        checks_run += 1
        if not sample.response or len(sample.response.strip()) < 5:
            issues.append("No conclusion or conclusion too brief")

        # Check 2: Must have minimum steps
        checks_run += 1
        min_steps = self.config.min_chain_steps
        if len(sample.reasoning_chain) < min_steps:
            issues.append(f"Only {len(sample.reasoning_chain)} step(s) — need at least {min_steps}")

        # Check 3: Conclusion must follow from final step (only when both exist)
        if sample.reasoning_chain and sample.response:
            checks_run += 1
            last_step = sample.reasoning_chain[-1]
            # Strip punctuation for consistent matching — "42." should match "42",
            # "answer," should match "answer". Every other check does this.
            # Include justification — the last step's justification often contains
            # the connecting term to the conclusion (e.g., "by substitution, x = 5").
            last_text = last_step.content
            if last_step.justification and last_step.justification != "implicit":
                last_text = f"{last_text} {last_step.justification}"
            last_words = set(re.findall(r'\S+', re.sub(r'[.,!?;:"\']', ' ', last_text.lower())))
            conclusion_words = set(re.findall(r'\S+', re.sub(r'[.,!?;:"\']', ' ', sample.response.lower())))
            # Include numeric tokens (any word containing digits) regardless of length
            # so conclusions like "42" or "x=5" match final steps mentioning them
            significant_last = {w for w in last_words if (len(w) > 3 and w not in _TRIVIAL_WORDS) or any(c.isdigit() for c in w)}
            # Use a lower length threshold for conclusion words — conclusions
            # are often very short ("no", "yes", "42"). Without this, short text
            # conclusions produce an empty significant_conc, and the intersection
            # is always empty, causing check 3 to ALWAYS fail for short answers.
            significant_conc = {w for w in conclusion_words if (len(w) > 2 and w not in _TRIVIAL_WORDS) or any(c.isdigit() for c in w)}
            if not (significant_last & significant_conc):
                issues.append("Final step does not connect to conclusion — logical gap")

        return issues, checks_run

    def _model_verify(self, sample: TrainingSample) -> tuple[bool, str]:
        """Use the model itself to verify a reasoning chain (escalation phase).

        Returns (is_valid, reason_text). reason_text is the model's explanation
        when invalid, or empty string when valid.
        """
        if not self._model_verifier:
            return True, ""

        chain_text = sample._build_full_response()
        verify_prompt = (
            f"You are a logic checker. Examine this reasoning chain for errors.\n\n"
            f"PROBLEM:\n{sample.prompt}\n\n"
            f"REASONING:\n{chain_text}\n\n"
            f"Is every step logically valid? Does the conclusion follow? "
            f"Answer ONLY 'VALID' or 'INVALID: <reason>'."
        )

        response = self._model_verifier.generate(
            verify_prompt,
            max_new_tokens=256,
            temperature=0.0,
        )
        resp_text = response.strip()
        # Strip markdown formatting — models may respond "**VALID**" or "`VALID`"
        first_word = re.sub(r'[*`_]', '', resp_text.split()[0]).upper() if resp_text else ""
        if first_word == "VALID":
            return True, ""
        return False, resp_text[:200] if resp_text else "no response"
