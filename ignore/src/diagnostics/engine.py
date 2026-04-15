"""Diagnostics engine — finds what the model is bad at."""

import hashlib
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, gcd
from typing import Optional

import torch

# Memory limiter for subprocess code execution — imported once at module level
_PREEXEC_FN = None
if sys.platform != "win32":
    import resource as _resource
    _MEM_LIMIT = 512 * 1024 * 1024  # 512MB
    def _preexec_limit_mem():
        _resource.setrlimit(_resource.RLIMIT_AS, (_MEM_LIMIT, _MEM_LIMIT))
    _PREEXEC_FN = _preexec_limit_mem

from ..utils.config import DiagnosticsConfig
from ..utils.model_loader import ModelLoader


@dataclass
class WeaknessReport:
    """A diagnosed weakness in the model."""
    domain: str
    subdomain: str
    severity: float  # 0-1, higher = worse
    evidence: list[dict] = field(default_factory=list)
    weak_layers: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DiagnosticResult:
    """Full diagnostic result for one cycle."""
    cycle: int
    timestamp: float
    weaknesses: list[WeaknessReport] = field(default_factory=list)
    domain_scores: dict[str, float] = field(default_factory=dict)
    domain_question_counts: dict[str, int] = field(default_factory=dict)
    layer_health: dict[str, float] = field(default_factory=dict)
    total_questions: int = 0
    total_correct: int = 0

    @property
    def overall_score(self) -> float:
        if not self.domain_scores:
            return 0.0
        # Weight by question count so domains with more questions (and thus more
        # reliable scores) contribute proportionally. Domains with 5 questions
        # shouldn't have the same influence as domains with 200.
        if self.domain_question_counts:
            total_q = sum(self.domain_question_counts.values())
            if total_q > 0:
                return sum(
                    score * self.domain_question_counts.get(domain, 1)
                    for domain, score in self.domain_scores.items()
                ) / total_q
        return sum(self.domain_scores.values()) / len(self.domain_scores)


# Each domain -> subdomain -> list of parameterized templates
# Templates with {A},{B},{C} get randomized nouns for anti-memorization
QUESTION_TEMPLATES = {
    "reasoning": {
        "syllogism": [
            {"prompt": "If all {A} are {B} and all {B} are {C}, are all {A} {C}? Explain step by step.", "expected": "yes", "check_type": "contains"},
            {"prompt": "If all {A} are {B} and some {B} are {C}, are all {A} {C}? Explain step by step.", "expected": "no", "check_type": "contains"},
            {"prompt": "If no {A} are {B} and all {C} are {A}, are any {C} also {B}? Explain step by step.", "expected": "no", "check_type": "contains"},
            {"prompt": "If some {A} are {B} and some {B} are {C}, are some {A} necessarily {C}? Explain.", "expected": "no", "check_type": "contains"},
            {"prompt": "All {A} are {B}. No {B} are {C}. Are any {A} {C}? Prove it.", "expected": "no", "check_type": "contains"},
        ],
        "cognitive_bias": [
            {"prompt": "A bat and ball cost ${total} total. The bat costs ${diff} more than the ball. How much does the ball cost? Show your work.", "expected": "{ball_answer}", "check_type": "contains", "_params": "bat_ball"},
            {"prompt": "If it takes {n} machines {n} minutes to make {n} widgets, how long would it take {m} machines to make {m} widgets? Show your work.", "expected": "{n}", "check_type": "contains", "_params": "machines"},
            {"prompt": "A lily pad doubles in size every day. If it takes {days} days to cover the whole lake, how many days to cover half? Show reasoning.", "expected": "{half_answer}", "check_type": "contains", "_params": "lilypad"},
        ],
        "counterfactual": [
            {"prompt": "If the sun rose in the west, which direction would shadows point at noon in the northern hemisphere? Reason step by step.", "expected": "east", "check_type": "contains"},
            {"prompt": "If gravity were repulsive instead of attractive, what would happen to the ocean tides? Reason step by step.", "expected": "away", "check_type": "contains"},
        ],
        "causal": [
            {"prompt": "A plant dies. The soil is dry. Did the dry soil cause the plant to die, or did the dead plant cause the soil to dry? What additional information would you need?", "expected": "additional information", "check_type": "contains"},
            {"prompt": "Ice cream sales and drowning deaths both increase in summer. Does ice cream cause drowning? Explain the logical error.", "expected": "correlation", "check_type": "contains"},
        ],
    },
    "math": {
        "calculus": [
            {"prompt": "What is the derivative of x^{n} + {a}x^2 - {b}x + {c}? Show each step.", "expected": "{deriv_answer}", "check_type": "contains", "_params": "derivative"},
            {"prompt": "Find the integral of 2x*cos(x^2) dx. Show your substitution.", "expected": "sin(x^2)", "check_type": "math_equiv"},
            {"prompt": "Find the limit as x approaches 0 of sin(x)/x. Prove your answer.", "expected": "1", "check_type": "contains"},
            {"prompt": "Find the second derivative of e^(2x) + ln(x). Show all steps.", "expected": "4*exp(2*x) - 1/x**2", "check_type": "math_equiv"},
        ],
        "algebra": [
            {"prompt": "Solve: {base}^x = {result}. Show your reasoning.", "expected": "{exp_answer}", "check_type": "contains", "_params": "exponential"},
            {"prompt": "Factor completely: x^3 - {cube}. Show each step.", "expected": "(x - {root})", "check_type": "contains", "_params": "factor_cube"},
            {"prompt": "Solve the system: {a1}x + {b1}y = {c1}, {a2}x + {b2}y = {c2}. Show all work.", "expected": "x = {x_answer}", "check_type": "contains", "_params": "system"},
            {"prompt": "Solve the system: {a1}x + {b1}y = {c1}, {a2}x + {b2}y = {c2}. Find y. Show all work.", "expected": "y = {y_answer}", "check_type": "contains", "_params": "system"},
        ],
        "number_theory": [
            {"prompt": "Is {num} prime? Explain your reasoning step by step.", "expected": "{prime_answer}", "check_type": "contains", "_params": "primality"},
            {"prompt": "What is the GCD of {a} and {b}? Use the Euclidean algorithm and show each step.", "expected": "{gcd_answer}", "check_type": "contains", "_params": "gcd"},
        ],
        "probability": [
            {"prompt": "You flip a fair coin {n} times. What is the probability of getting exactly {k} heads? Show the calculation.", "expected": "{prob_answer}", "check_type": "contains", "_params": "coin_flip"},
        ],
    },
    "code": {
        "implementation": [
            {"prompt": "Write a Python function that checks if a string is a palindrome, handling edge cases (empty string, spaces, case).", "expected": "def", "check_type": "code_executes"},
            {"prompt": "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes.", "expected": "def", "check_type": "code_executes"},
            {"prompt": "Write a Python function that merges two sorted lists into one sorted list without using built-in sort.", "expected": "def", "check_type": "code_executes"},
            {"prompt": "Write a Python function that implements binary search on a sorted list. Return the index or -1.", "expected": "def", "check_type": "code_executes"},
        ],
        "debugging": [
            {"prompt": "This Python code has a bug: `def fib(n): return fib(n-1) + fib(n-2)`. What's wrong and how do you fix it? Explain step by step.", "expected": "base case", "check_type": "contains"},
            {"prompt": "This code has a bug: `def avg(lst): return sum(lst) / len(lst)`. What happens with an empty list and how do you fix it?", "expected": "empty", "check_type": "contains"},
        ],
        "complexity": [
            {"prompt": "What is the time complexity of this: `for i in range(n): for j in range(i, n): pass`? Prove it.", "expected": "n^2", "check_type": "contains"},
            {"prompt": "What is the time complexity of binary search? Prove it mathematically.", "expected": "log", "check_type": "contains"},
        ],
    },
    "logic": {
        "fallacies": [
            {"prompt": "If it's raining, the ground is wet. The ground is wet. Is it necessarily raining? Name the fallacy.", "expected": "affirming the consequent", "check_type": "contains"},
            {"prompt": "Everyone who studied passed. Alice passed. Did Alice study? Explain the logical error.", "expected": "affirming the consequent", "check_type": "contains"},
            {"prompt": "'No true Scotsman would do X. But McGregor did X. He's not a true Scotsman.' What fallacy is this?", "expected": "no true scotsman", "check_type": "contains"},
            {"prompt": "'You can't prove ghosts don't exist, so they must exist.' What fallacy is this?", "expected": "burden of proof", "check_type": "contains"},
            {"prompt": "'Everyone is buying this product, so it must be good.' What fallacy is this?", "expected": "bandwagon", "check_type": "contains"},
        ],
        "propositional": [
            {"prompt": "Is 'If P then Q' logically equivalent to 'If not Q then not P'? Prove using a truth table.", "expected": "yes", "check_type": "contains"},
            {"prompt": "Is 'P or Q' the same as 'not (not P and not Q)'? Prove using De Morgan's law.", "expected": "yes", "check_type": "contains"},
        ],
        "modal": [
            {"prompt": "Is it possible for something to be necessarily true but not actually true? Explain using modal logic.", "expected": "no", "check_type": "contains"},
        ],
    },
    "science": {
        "physics": [
            {"prompt": "A ball is thrown straight up at {v} m/s. Ignoring air resistance, how high does it go? (g=10 m/s^2) Show all steps.", "expected": "{height_answer}", "check_type": "contains", "_params": "projectile"},
            {"prompt": "Two objects of mass {m1} kg and {m2} kg are {d} meters apart. What is the gravitational force between them? (G=6.67e-11) Show work.", "expected": "{grav_answer}", "check_type": "math_equiv", "_params": "gravity"},
        ],
        "biology": [
            {"prompt": "Explain why antibiotics don't work on viruses. Be specific about the mechanism.", "expected": "cell", "check_type": "contains"},
            {"prompt": "Explain how CRISPR-Cas9 gene editing works at the molecular level.", "expected": "guide RNA", "check_type": "contains"},
        ],
        "chemistry": [
            {"prompt": "Balance this equation: Fe + O2 -> Fe2O3. Show your work.", "expected": "4Fe", "check_type": "contains"},
            {"prompt": "What is the pH of a 0.01 M HCl solution? Show the calculation.", "expected": "2", "check_type": "contains"},
        ],
    },
    "common_sense": {
        "physical": [
            {"prompt": "Can you fit a basketball inside a mailbox? Explain your reasoning about the physical dimensions.", "expected": "no", "check_type": "contains"},
            {"prompt": "If you drop a bowling ball and a feather in a vacuum, which hits the ground first? Explain.", "expected": "same", "check_type": "contains"},
            {"prompt": "Can a person lift a car with one hand? Explain the physics involved.", "expected": "no", "check_type": "contains"},
            {"prompt": "If you pour water into a cup that already has a hole in the bottom, will the cup fill up? Explain.", "expected": "no", "check_type": "contains"},
            {"prompt": "Can sound travel through outer space? Explain why or why not.", "expected": "no", "check_type": "contains"},
        ],
        "social": [
            {"prompt": "If someone at a funeral is laughing loudly, is this likely appropriate? Explain.", "expected": "no", "check_type": "contains"},
            {"prompt": "Is it appropriate to clap after a surgeon finishes an operation? Explain the social norms.", "expected": "no", "check_type": "contains"},
            {"prompt": "If a stranger asks for your home address on the street, should you give it? Explain.", "expected": "no", "check_type": "contains"},
        ],
        "temporal": [
            {"prompt": "Can you eat breakfast before you wake up? Explain the logical impossibility or possibility.", "expected": "no", "check_type": "contains"},
            {"prompt": "Can yesterday come after tomorrow? Explain.", "expected": "no", "check_type": "contains"},
            {"prompt": "Can you remember events from the future? Explain why or why not.", "expected": "no", "check_type": "contains"},
            {"prompt": "If today is Monday, what day was it 3 days ago? Show your reasoning.", "expected": "friday", "check_type": "contains"},
        ],
        "spatial": [
            {"prompt": "If you are facing north and turn 90 degrees clockwise, which direction are you facing? Explain.", "expected": "east", "check_type": "contains"},
            {"prompt": "Can a cube have exactly 5 faces? Explain using geometry.", "expected": "no", "check_type": "contains"},
        ],
    },
    "language_understanding": {
        "ambiguity": [
            {"prompt": "What does 'I saw her duck' mean? List all possible interpretations.", "expected": "duck", "check_type": "contains"},
            {"prompt": "What does 'Time flies like an arrow; fruit flies like a banana' mean? Explain each interpretation.", "expected": "flies", "check_type": "contains"},
            {"prompt": "What does 'Visiting relatives can be boring' mean? List all interpretations.", "expected": "relatives", "check_type": "contains"},
            {"prompt": "What does 'The chicken is ready to eat' mean? Explain both interpretations.", "expected": "chicken", "check_type": "contains"},
        ],
        "implication": [
            {"prompt": "'Some students passed the exam.' Does this imply some failed? Explain pragmatic vs logical.", "expected": "not necessarily", "check_type": "contains"},
            {"prompt": "'He managed to finish on time.' Does 'managed' imply difficulty? Explain.", "expected": "difficulty", "check_type": "contains"},
            {"prompt": "'Not all birds can fly.' Does this mean some birds can fly? Explain the logic.", "expected": "some", "check_type": "contains"},
        ],
        "sentiment": [
            {"prompt": "'Oh great, another meeting.' Is this positive or negative? Explain sarcasm detection.", "expected": "sarcas", "check_type": "contains"},
            {"prompt": "'What a wonderful way to spend my Saturday — doing taxes.' Positive or negative? Explain.", "expected": "sarcas", "check_type": "contains"},
            {"prompt": "'I could care less about the result.' What does the speaker actually mean? Explain.", "expected": "couldn't care less", "check_type": "contains"},
        ],
        "reference": [
            {"prompt": "'The trophy would not fit in the suitcase because it was too big.' What does 'it' refer to? Explain.", "expected": "trophy", "check_type": "contains"},
            {"prompt": "'The city council refused the protesters a permit because they feared violence.' Who feared violence? Explain.", "expected": "council", "check_type": "contains"},
        ],
    },
    "abstraction": {
        "pattern": [
            {"prompt": "What comes next: 1, 1, 2, 3, 5, 8, 13, ...? Explain the pattern.", "expected": "21", "check_type": "contains"},
            {"prompt": "What comes next: 2, 6, 12, 20, 30, ...? Explain the pattern.", "expected": "42", "check_type": "contains"},
            {"prompt": "What comes next: 1, 4, 9, 16, 25, ...? Explain the pattern.", "expected": "36", "check_type": "contains"},
            {"prompt": "What comes next: 1, 3, 6, 10, 15, ...? Explain the pattern.", "expected": "21", "check_type": "contains"},
            {"prompt": "What comes next: 2, 3, 5, 7, 11, 13, ...? Explain the pattern.", "expected": "17", "check_type": "contains"},
            {"prompt": "What comes next: 1, 8, 27, 64, 125, ...? Explain the pattern.", "expected": "216", "check_type": "contains"},
        ],
        "analogy": [
            {"prompt": "Hot is to cold as light is to ___. Explain the relationship.", "expected": "dark", "check_type": "contains"},
            {"prompt": "Bird is to nest as human is to ___. Explain the analogy.", "expected": "house", "check_type": "contains"},
            {"prompt": "Painter is to canvas as writer is to ___. Explain the analogy.", "expected": "paper", "check_type": "contains"},
            {"prompt": "Fish is to water as bird is to ___. Explain the analogy.", "expected": "air", "check_type": "contains"},
        ],
        "classification": [
            {"prompt": "Which doesn't belong: apple, banana, carrot, grape? Explain your reasoning.", "expected": "carrot", "check_type": "contains"},
            {"prompt": "Which doesn't belong: circle, square, triangle, cube? Explain your reasoning.", "expected": "cube", "check_type": "contains"},
            {"prompt": "Which doesn't belong: 2, 3, 5, 9, 11? Explain your reasoning.", "expected": "9", "check_type": "contains"},
        ],
    },
}

# Module-level constants for Unicode superscript normalization in _check_answer
_SUPERSCRIPT_TABLE = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_RE_SUPERSCRIPT = re.compile(r'(?<=[^⁰¹²³⁴⁵⁶⁷⁸⁹])[⁰¹²³⁴⁵⁶⁷⁸⁹]+')

_NOUNS = ["bloops", "razzles", "lazzles", "flinks", "morgs", "pliffs", "zents", "quargs",
          "dreps", "snivs", "wumps", "glorps", "brixes", "tazzles", "nimps", "florbs",
          "grepts", "shlinks", "vorps", "klems", "frobs", "glints", "praxes", "qualms"]

_LANG_TAGS = frozenset({"python", "python3", "py", "javascript", "js", "typescript", "ts",
                        "java", "go", "rust", "ruby", "bash", "sh", "sql",
                        "cpp", "c", "csharp", "html", "css", "json", "yaml"})


def _generate_params(param_type: str, seed: int) -> dict:
    """Generate randomized parameters for question templates."""
    rng = random.Random(seed)

    if param_type == "bat_ball":
        ball = rng.choice([5, 10, 15, 20])  # cents
        diff = rng.choice([100, 200, 50])  # cents
        total = (ball * 2 + diff) / 100
        return {"total": f"{total:.2f}", "diff": f"{diff/100:.2f}", "ball_answer": f"{ball/100:.2f}"}

    if param_type == "machines":
        n = rng.choice([3, 5, 7, 10])
        m = rng.choice([50, 100, 200])
        return {"n": str(n), "m": str(m)}

    if param_type == "lilypad":
        days = rng.choice([30, 48, 60])
        return {"days": str(days), "half_answer": str(days - 1)}

    if param_type == "derivative":
        n = rng.randint(3, 5)  # min 3 so leading term is distinct from ax^2
        a = rng.randint(1, 6)
        b = rng.randint(1, 10)
        c = rng.randint(1, 10)
        # Full derivative: nx^(n-1) + 2ax - b
        # Check for the leading term in the model's response
        return {"n": str(n), "a": str(a), "b": str(b), "c": str(c),
                "deriv_answer": f"{n}x^{n-1}"}

    if param_type == "exponential":
        base = rng.choice([2, 3, 5])
        exp = rng.randint(2, 6)
        return {"base": str(base), "result": str(base**exp), "exp_answer": str(exp)}

    if param_type == "factor_cube":
        root = rng.choice([2, 3, 4, 5])
        return {"cube": str(root**3), "root": str(root)}

    if param_type == "system":
        x, y = rng.randint(1, 5), rng.randint(1, 5)
        # Ensure non-singular system (det != 0) so a unique solution exists.
        # Without this, some coefficient combos produce det=0 (e.g., a1=2,b1=1,a2=2,b2=-1),
        # and the "expected" answer is wrong — the system has no unique solution.
        for _ in range(20):
            a1, b1 = rng.randint(1, 4), rng.randint(1, 4)
            a2, b2 = rng.randint(1, 4), rng.randint(-3, -1)
            if a1 * b2 - a2 * b1 != 0:
                break
        else:
            # Fallback: guaranteed non-singular
            a1, b1, a2, b2 = 1, 1, 1, -1
        return {"a1": str(a1), "b1": str(b1), "c1": str(a1*x + b1*y),
                "a2": str(a2), "b2": str(b2), "c2": str(a2*x + b2*y),
                "x_answer": str(x), "y_answer": str(y)}

    if param_type == "primality":
        # Mix of primes and composites
        composites = [91, 57, 87, 51, 119, 133, 143, 161]
        primes = [97, 89, 83, 79, 71, 67, 61, 59]
        is_prime = rng.choice([True, False])
        num = rng.choice(primes if is_prime else composites)
        return {"num": str(num), "prime_answer": "yes" if is_prime else "no"}

    if param_type == "gcd":
        a = rng.randint(20, 200)
        b = rng.randint(20, 200)
        return {"a": str(a), "b": str(b), "gcd_answer": str(gcd(a, b))}

    if param_type == "coin_flip":
        n = rng.choice([3, 4, 5])
        k = rng.randint(1, n - 1)
        prob = comb(n, k) / (2**n)
        frac = Fraction(prob).limit_denominator(1000)
        return {"n": str(n), "k": str(k), "prob_answer": str(frac)}

    if param_type == "projectile":
        v = rng.choice([10, 20, 30, 40])
        height = v**2 // 20  # v^2/(2g)
        return {"v": str(v), "height_answer": str(height)}

    if param_type == "gravity":
        m1 = rng.choice([500, 1000, 2000, 5000])
        m2 = rng.choice([500, 1000, 2000, 5000])
        d = rng.choice([5, 10, 20, 50])
        force = 6.67e-11 * m1 * m2 / (d ** 2)
        # Use math_equiv check type so different notations all match.
        # Store the numeric value as a string for sympy comparison.
        return {"m1": str(m1), "m2": str(m2), "d": str(d),
                "grav_answer": f"{force:.4g}"}

    return {}


class DiagnosticsEngine:
    """Rapidly probes the model to find its weakest areas."""

    def __init__(self, config: DiagnosticsConfig, model_loader: ModelLoader):
        self.config = config
        self.model = model_loader
        self._seen_hashes: set[str] = set()
        self._model_generated_questions: dict[str, list[dict]] = {}  # from escalation

    def run(self, cycle: int) -> DiagnosticResult:
        """Run full diagnostics."""
        # Reset seen hashes each cycle — prevents hash set from growing unboundedly
        # across 100+ cycles (which would block most question variants from generating).
        # Anti-memorization comes from randomized params per cycle, not cross-cycle dedup.
        self._seen_hashes.clear()
        result = DiagnosticResult(cycle=cycle, timestamp=time.time())

        for domain in self.config.domains:
            score, evidence = self._probe_domain(domain, cycle)
            result.domain_scores[domain] = score
            result.domain_question_counts[domain] = len(evidence)
            result.total_questions += len(evidence)
            result.total_correct += sum(1 for e in evidence if e["correct"])

            if score < self.config.confidence_threshold:
                subweaknesses = self._drill_down(domain, evidence)
                result.weaknesses.extend(subweaknesses)

        if self.config.activation_analysis:
            layer_health = self._analyze_layers_with_activations()
            result.layer_health = layer_health
            self._correlate_weak_layers(result)

        result.weaknesses.sort(key=lambda w: w.severity, reverse=True)
        # Free VRAM from diagnostic inferences (200+ generate calls across all
        # domains + activation probes). Without this, the CUDA allocator is
        # heavily fragmented when generation/training phases need large contiguous
        # blocks for KV cache and gradient checkpointing.
        torch.cuda.empty_cache()
        return result

    def generate_adaptive_questions(self, weak_domains: list[str], cycle: int = 0):
        """Escalation: ask the model to generate diagnostic questions for its weak areas.

        Uses cycle number in the prompt to get different questions each time.
        """
        # Vary the difficulty focus each cycle to get diverse questions
        focus_areas = ["conceptual understanding", "edge cases and exceptions",
                       "multi-step reasoning", "common misconceptions", "applied problems"]
        # Batch all domain prompts for efficiency — sequential generate() calls
        # waste GPU time on kernel launch overhead and KV cache recomputation.
        prompts = []
        for domain in weak_domains:
            focus = focus_areas[cycle % len(focus_areas)]
            prompts.append(
                f"Generate 5 challenging {domain} questions focused on {focus}. "
                f"These are for cycle #{cycle} — make them DIFFERENT from previous rounds. "
                f"Each question must have a definite correct answer. "
                f"Format each as:\n"
                f"Q: [question]\n"
                f"A: [short answer]\n\n"
                f"Make them progressively harder."
            )
        responses = self.model.generate_batch(prompts, max_new_tokens=1024, temperature=0.7)
        # Guard against length mismatch (same pattern as verifier)
        if len(responses) < len(weak_domains):
            responses.extend([""] * (len(weak_domains) - len(responses)))
        for domain, response in zip(weak_domains, responses):
            questions = self._parse_generated_questions(response, domain)
            # Only replace if we got questions — if parsing failed (model returned
            # unparseable output), keep the previous cycle's valid questions rather
            # than wiping them out with an empty list.
            if questions:
                self._model_generated_questions[domain] = questions

    def _parse_generated_questions(self, response: str, domain: str) -> list[dict]:
        """Parse model-generated questions into question format."""
        questions = []
        lines = response.split("\n")
        current_q = None
        for line in lines:
            stripped = line.strip()
            # Handle "Q:", "1. Q:", "1) Q:" — models often number their questions
            q_match = re.match(r'^(?:\d+[.)]\s*)?Q:\s*(.*)', stripped, re.IGNORECASE)
            a_match = re.match(r'^(?:\d+[.)]\s*)?A:\s*(.*)', stripped, re.IGNORECASE)
            if q_match:
                current_q = q_match.group(1).strip()
            elif a_match and current_q:
                answer = a_match.group(1).strip()
                # Sanitize: model answers are often verbose ("Yes, because X").
                # Extract only the core answer — take text before first comma,
                # period, or "because"/"since" clause. Long answers as expected
                # values almost never match with "contains" check.
                # Lower first since the expected value is lowered anyway — avoids
                # index mismatch from Unicode case folding changing string length.
                answer = answer.lower()
                for sep in [" because ", " since ", ". ", ", "]:
                    idx = answer.find(sep)
                    if idx != -1:
                        answer = answer[:idx]
                        break
                # Skip answers that are still too long to be useful keywords
                if len(answer) > 80:
                    current_q = None
                    continue
                questions.append({
                    "prompt": current_q + " Show your reasoning step by step.",
                    "expected": answer.lower().strip(),
                    "check_type": "contains",
                    "subdomain": "model_generated",
                })
                current_q = None
        return questions

    def _probe_domain(self, domain: str, cycle: int) -> tuple[float, list[dict]]:
        """Probe a domain with randomized questions."""
        questions = self._generate_questions(domain, cycle)

        # Add model-generated questions if available (from escalation)
        if domain in self._model_generated_questions:
            questions.extend(self._model_generated_questions[domain])

        if not questions:
            return 1.0, []

        evidence = []
        batch_prompts = [q["prompt"] for q in questions]
        batch_meta = questions

        for batch_start in range(0, len(batch_prompts), self.config.batch_size):
            batch_p = batch_prompts[batch_start:batch_start + self.config.batch_size]
            batch_m = batch_meta[batch_start:batch_start + self.config.batch_size]

            responses = self.model.generate_batch(batch_p, max_new_tokens=512, temperature=0.0)

            # Guard against partial responses — if generate_batch returns fewer
            # than expected (OOM mid-batch), pad with empty strings so all questions
            # get scored (as incorrect). Without this, zip silently drops questions,
            # inflating the domain score.
            if len(responses) < len(batch_p):
                responses.extend([""] * (len(batch_p) - len(responses)))

            for q, response in zip(batch_m, responses):
                correct = self._check_answer(response, q["expected"], q["check_type"])
                evidence.append({
                    "question": q["prompt"],
                    "expected": q["expected"],
                    # Truncate response in evidence to save memory — full responses
                    # for 200+ questions per domain would be megabytes.
                    "response": response[:500],
                    "correct": correct,
                    "domain": domain,
                    "subdomain": q.get("subdomain", "general"),
                })

        correct_count = sum(1 for e in evidence if e["correct"])
        score = correct_count / len(evidence) if evidence else 0.0
        return score, evidence

    def _generate_questions(self, domain: str, cycle: int) -> list[dict]:
        """Generate questions for a domain up to questions_per_domain limit."""
        templates = QUESTION_TEMPLATES.get(domain, {})
        questions = []
        # Seeded RNG for reproducibility within a cycle+domain.
        # Use hashlib (not hash()) — hash() output varies across Python sessions
        # with PYTHONHASHSEED randomization, breaking cross-run reproducibility.
        seed_bytes = hashlib.md5(f"{domain}:{cycle}".encode()).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:4], "big"))

        for subdomain, template_list in templates.items():
            for template in template_list:
                variants = self._create_variants(template, subdomain, cycle)
                questions.extend(variants)

        # Pad up to configured limit with extra variants
        target = self.config.questions_per_domain
        attempts = 0
        prev_count = -1
        while len(questions) < target and attempts < target * 2:
            # If no new questions were generated in the last iteration,
            # stop — this domain doesn't have enough parameterizable templates.
            if len(questions) == prev_count:
                break
            prev_count = len(questions)
            attempts += 1
            for subdomain, template_list in templates.items():
                if len(questions) >= target:
                    break
                if not template_list:
                    continue
                template = rng.choice(template_list)
                # Use cycle * 1000 + attempts to avoid seed collision — cycle + attempts
                # means cycle=5,attempts=3 collides with cycle=8,attempts=0, reducing
                # cross-cycle question diversity.
                extra = self._create_variants(template, subdomain, cycle * 1000 + attempts)
                questions.extend(extra)

        # If domain has no templates at all
        if not questions:
            questions.append({
                "prompt": f"Explain a fundamental concept in {domain} with step-by-step reasoning.",
                "expected": "",
                "check_type": "nonempty",
                "subdomain": "general",
            })

        rng.shuffle(questions)
        return questions[:target]  # cap at limit

    def _create_variants(self, template: dict, subdomain: str, cycle: int) -> list[dict]:
        """Create randomized variants of a question template."""
        variants = []
        prompt = template["prompt"]
        param_type = template.get("_params")

        if "{A}" in prompt:
            # Syllogism — random nouns, seeded for reproducibility
            seed_bytes = hashlib.md5(f"{subdomain}:{cycle}:{prompt[:50]}".encode()).digest()
            rng = random.Random(int.from_bytes(seed_bytes[:4], "big"))
            for i in range(3):
                nouns = rng.sample(_NOUNS, 3)
                variant_prompt = prompt.format(A=nouns[0], B=nouns[1], C=nouns[2])
                h = hashlib.md5(variant_prompt.encode()).hexdigest()
                if h not in self._seen_hashes:
                    self._seen_hashes.add(h)
                    variants.append({
                        "prompt": variant_prompt,
                        "expected": template["expected"],
                        "check_type": template["check_type"],
                        "subdomain": subdomain,
                    })
        elif param_type:
            # Parameterized template — generate 3 random variants (match syllogism count)
            for variant_idx in range(3):
                # Use template content for seed, not id() (memory address varies across runs)
                seed_bytes = hashlib.md5(f"{subdomain}:{cycle}:{template['prompt'][:50]}:{variant_idx}".encode()).digest()
                seed = int.from_bytes(seed_bytes[:4], "big")
                params = _generate_params(param_type, seed)
                if params:
                    try:
                        variant_prompt = prompt.format(**params)
                        expected = template["expected"].format(**params)
                        h = hashlib.md5(variant_prompt.encode()).hexdigest()
                        if h not in self._seen_hashes:
                            self._seen_hashes.add(h)
                            variants.append({
                                "prompt": variant_prompt,
                                "expected": expected,
                                "check_type": template["check_type"],
                                "subdomain": subdomain,
                            })
                    except (KeyError, ValueError, IndexError):
                        # Don't fall back to raw template with unsubstituted
                        # placeholders like "{base}^x = {result}" — the model
                        # would get tested on nonsensical literal brace text.
                        # Skip this variant (not the whole template — other seeds
                        # may produce valid params).
                        continue
        else:
            h = hashlib.md5(prompt.encode()).hexdigest()
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                variants.append({
                    "prompt": prompt,
                    "expected": template["expected"],
                    "check_type": template["check_type"],
                    "subdomain": subdomain,
                })

        return variants

    def _drill_down(self, domain: str, evidence: list[dict]) -> list[WeaknessReport]:
        """Identify specific subdomains of failure."""
        subdomain_results: dict[str, dict] = {}
        for e in evidence:
            sd = e["subdomain"]
            if sd not in subdomain_results:
                subdomain_results[sd] = {"total": 0, "failures": []}
            subdomain_results[sd]["total"] += 1
            if not e["correct"]:
                subdomain_results[sd]["failures"].append(e)

        weaknesses = []
        for subdomain, data in subdomain_results.items():
            if not data["failures"]:
                continue
            raw_severity = len(data["failures"]) / max(data["total"], 1)
            # Discount severity for low sample counts — 1/1 failures is less
            # reliable than 50/100. Use Wilson lower bound approximation:
            # for n < 10, discount by sqrt(n/10) so 1 question → 0.32x, 5 → 0.71x.
            n = data["total"]
            confidence_discount = min(1.0, (n / 10) ** 0.5) if n < 10 else 1.0
            severity = raw_severity * confidence_discount
            # Cap evidence to 20 items — full lists can be 100+ items for weak
            # subdomains, wasting memory and slowing checkpoint/log serialization.
            capped_evidence = data["failures"][:20]
            weaknesses.append(WeaknessReport(
                domain=domain,
                subdomain=subdomain,
                severity=severity,
                evidence=capped_evidence,
                description=f"Fails {severity:.0%} of {subdomain} in {domain} ({len(data['failures'])}/{data['total']})",
            ))
        return weaknesses

    def _analyze_layers_with_activations(self) -> dict[str, float]:
        """Analyze layers using weight norms and activation health."""
        layer_info = self.model.get_layer_info()
        health = {}
        norms = {name: info["norm"] for name, info in layer_info.items()}
        if not norms:
            return health

        mean_norm = sum(norms.values()) / len(norms)
        std_norm = (sum((n - mean_norm) ** 2 for n in norms.values()) / len(norms)) ** 0.5

        for name, norm in norms.items():
            if std_norm > 0:
                z_score = abs(norm - mean_norm) / std_norm
                health[name] = max(0.0, 1.0 - (z_score / 3.0))
            else:
                health[name] = 1.0

        # Multiple diverse probes for activation analysis — a single narrow probe
        # (e.g., only math) gives misleading health for code/language layers.
        probes = [
            "Explain step by step why 2 + 2 = 4, starting from the Peano axioms.",
            "Write a Python function to reverse a linked list. Explain your approach.",
            "Is the following argument valid? All cats are mammals. Some mammals are pets. Therefore some cats are pets.",
        ]
        # Accumulate health scores across probes and average
        probe_healths: dict[str, list[float]] = {}
        for probe in probes:
            try:
                with self.model.capture_activations() as capture:
                    self.model.generate(probe, max_new_tokens=128, temperature=0.0)

                # Free KV cache from the probe inference before the next probe.
                # Without this, 3 probes' KV caches stay allocated until GC,
                # fragmenting VRAM before training needs contiguous blocks.
                torch.cuda.empty_cache()

                for act_name, stats in capture.activations.items():
                    act_health = 1.0
                    # Dead neurons: scale penalty with severity. 10% dead = -0.1,
                    # 50% dead = -0.5, 100% dead = -1.0 (completely unhealthy).
                    act_health -= stats.dead_ratio
                    # Exploding activations
                    if stats.std > 100:
                        act_health -= 0.3
                    # Vanishing activations (near-constant output)
                    elif stats.std < 0.01:
                        act_health -= 0.2
                    clamped = max(0.0, act_health)
                    if act_name not in probe_healths:
                        probe_healths[act_name] = []
                    probe_healths[act_name].append(clamped)
            except Exception:
                continue

        # Apply averaged activation health to parameter health.
        # Use min(weight_health, blended) so activation analysis can LOWER health
        # (catching dead/exploding neurons that weight norms miss) but never RAISE
        # it above the weight-norm signal. A layer with abnormal weight norms
        # (health=0.0) and healthy activations (1.0) should stay unhealthy — the
        # weight issue is real even if activations happen to look OK on 3 probes.
        for act_name, scores in probe_healths.items():
            avg_act_health = sum(scores) / len(scores)
            prefix = act_name + "."
            for param_name in health:
                if param_name.startswith(prefix):
                    blended = 0.5 * health[param_name] + 0.5 * avg_act_health
                    health[param_name] = min(health[param_name], blended)

        return health

    def _correlate_weak_layers(self, result: DiagnosticResult):
        """Assign weak layers to weaknesses based on layer type heuristics.

        Instead of running N extra forward passes (one per weakness),
        use structural heuristics: attention layers correlate with reasoning
        weaknesses, MLP layers correlate with knowledge weaknesses.
        """
        if not result.layer_health or not result.weaknesses:
            return

        sorted_health = sorted(result.layer_health.items(), key=lambda x: x[1])
        cutoff_idx = max(1, int(len(sorted_health) * self.config.weak_layer_percentile))
        globally_weak = [name for name, _ in sorted_health[:cutoff_idx]]

        # Categorize weak layers by type
        attention_layers = [n for n in globally_weak if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn"))]
        mlp_layers = [n for n in globally_weak if any(k in n for k in ("gate_proj", "up_proj", "down_proj", "mlp", "fc"))]

        reasoning_domains = {"reasoning", "math", "logic", "abstraction", "code"}
        knowledge_domains = {"science", "common_sense", "language_understanding"}

        # Distribute layers across weaknesses of the same type so different
        # weaknesses get different (partially overlapping) layer sets.
        # Without this, every reasoning weakness gets identical layers → identical
        # LoRA ranks, defeating the purpose of per-weakness targeting.
        reasoning_weaknesses = [w for w in result.weaknesses if w.domain in reasoning_domains]
        knowledge_weaknesses = [w for w in result.weaknesses if w.domain in knowledge_domains]
        other_weaknesses = [w for w in result.weaknesses if w.domain not in reasoning_domains and w.domain not in knowledge_domains]

        def _distribute(weaknesses: list, layer_pool: list):
            if not weaknesses or not layer_pool:
                for w in weaknesses:
                    w.weak_layers = list(globally_weak)
                return
            # Sort by severity (highest first) so the worst weaknesses get
            # the unhealthiest layers. Each weakness gets a sliding window.
            weaknesses_sorted = sorted(weaknesses, key=lambda w: w.severity, reverse=True)
            # Minimum window size: at least 3 layers or 1/3 of pool, whichever is larger.
            # This ensures meaningful differentiation even with few layers.
            min_window = max(3, len(layer_pool) // 3)
            window = max(min_window, len(layer_pool) // max(len(weaknesses_sorted), 1))
            window = min(window, len(layer_pool))  # can't exceed pool
            # Stride between windows — if more weaknesses than layer pool permits,
            # stride=0 and all get the same layers (unavoidable).
            stride = max(1, (len(layer_pool) - window) // max(len(weaknesses_sorted) - 1, 1)) if len(weaknesses_sorted) > 1 else 0
            for i, w in enumerate(weaknesses_sorted):
                start = min(i * stride, len(layer_pool) - window)
                w.weak_layers = layer_pool[start:start + window]

        _distribute(reasoning_weaknesses, attention_layers)
        _distribute(knowledge_weaknesses, mlp_layers)
        _distribute(other_weaknesses, globally_weak)

    def _check_answer(self, response: str, expected: str, check_type: str) -> bool:
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()

        if check_type == "nonempty":
            # Reject refusals/non-answers — must be substantive, not just long
            text = response.strip()
            if len(text) <= 20:
                return False
            refusal_markers = ("i don't know", "i'm not sure", "i cannot",
                               "i can't", "unable to", "not enough information")
            return not any(m in response_lower for m in refusal_markers)
        if check_type == "contains":
            # Normalize exponentiation syntax before matching — models write x^2,
            # x**2, x², and the expected answer uses ^. Normalize all to ^ so they match.
            response_lower = response_lower.replace("**", "^")
            # Unicode superscripts → ^ notation (only if any are present)
            if _RE_SUPERSCRIPT.search(response_lower):
                response_lower = _RE_SUPERSCRIPT.sub(
                    lambda m: '^' + m.group(0).translate(_SUPERSCRIPT_TABLE),
                    response_lower)

            # Word-boundary match to prevent "no" matching "know"/"notice",
            # "2" matching "12"/"200", etc.
            # Allow flexible spacing around '=' — models write "y=3", "y = 3",
            # "y =3" interchangeably. After re.escape (which doesn't escape = or space),
            # replace literal " = " with \s*=\s* so all variants match.
            escaped = re.escape(expected_lower)
            if '=' in escaped:
                escaped = re.sub(r'\s*=\s*', r'\\s*=\\s*', escaped)
            pattern = r'(?<!\w)' + escaped + r'(?!\w)'
            # For very short expected answers ("no", "yes"), restrict search to
            # conclusion-like parts of the response to reduce false positives.
            if len(expected_lower) <= 3:
                # Check conclusion markers first
                for marker in ("conclusion:", "answer:", "therefore,", "therefore ", "so,", "so the answer"):
                    if marker in response_lower:
                        conclusion_part = response_lower[response_lower.rindex(marker):]
                        if re.search(pattern, conclusion_part):
                            return True
                # Check first sentence and last few sentences. Checking ALL sentences
                # for "no"/"yes" creates false positives: "No simplification needed"
                # matches \bno\b even when the actual answer is "yes." Restrict to
                # positions where models typically state their final answer.
                sentences = [s.strip() for s in re.split(r'[.!?\n]', response_lower) if s.strip()]
                if sentences:
                    # Use dict.fromkeys to deduplicate while preserving order —
                    # if there's only 1 sentence, [:1] + [-3:] would search it twice.
                    check_parts = list(dict.fromkeys(sentences[:1] + sentences[-3:]))
                    return any(re.search(pattern, part) for part in check_parts)
                return False
            return bool(re.search(pattern, response_lower))
        if check_type == "exact":
            return response_lower == expected_lower
        if check_type == "math_equiv":
            return self._check_math_equivalence(response, expected)
        if check_type == "code_executes":
            return self._check_code_executes(response)
        return False

    def _check_math_equivalence(self, response: str, expected: str) -> bool:
        # Normalize scientific notation variants (6.67 × 10^-6 → 6.67e-6)
        def _normalize_sci(text: str) -> str:
            text = text.replace("×", "*").replace("\\times", "*")
            # "6.67 * 10^{-6}" or "6.67 * 10^-6" → "6.67e-6"
            text = re.sub(r'(\d+\.?\d*)\s*\*\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?',
                          lambda m: f"{m.group(1)}e{m.group(2)}", text)
            return text
        response = _normalize_sci(response)
        expected = _normalize_sci(expected)
        # Normalize exponentiation syntax but preserve multiplication
        clean_response = response.replace("**", "^")
        clean_expected = expected.replace(" ", "").replace("**", "^")

        # Quick check: expected appears as a standalone token (word boundaries).
        # Plain substring would false-positive: expected "2" matching "12" or "32".
        expected_nospace = clean_expected.replace(" ", "")
        if re.search(r'(?<!\w)' + re.escape(expected_nospace) + r'(?!\w)',
                      clean_response.replace(" ", "")):
            return True

        # Extract a math expression from the response — look for common patterns
        # like "= sin(x^2)", "answer: sin(x^2)", or the last line with math symbols
        expr_text = None
        # Try after last "=" sign
        if "=" in clean_response:
            expr_text = clean_response.split("=")[-1].strip()
        # Try after "answer:" or "result:"
        if not expr_text:
            m = re.search(r'(?:answer|result|equals?)\s*:?\s*(.+)', clean_response, re.IGNORECASE)
            if m:
                expr_text = m.group(1).strip()
        # Fallback: last non-empty line
        if not expr_text:
            lines = [l.strip() for l in clean_response.split("\n") if l.strip()]
            expr_text = lines[-1] if lines else clean_response

        # Strip trailing prose (take only the math-looking prefix)
        expr_text = re.split(r'[,.](?:\s|$)', expr_text)[0].strip()

        try:
            from sympy.parsing.sympy_parser import parse_expr
            from sympy import simplify

            # Sanitize before parsing — parse_expr uses eval() internally and can
            # execute arbitrary Python. Strip anything that isn't math-like.
            def _sanitize(text: str) -> str:
                text = text.replace("^", "**").replace(" ", "")
                # Normalize "e**(...)" to "exp(...)" so sympy treats it as
                # Euler's number, not a symbol named 'e'. Models write "e^(2x)"
                # which becomes "e**(2*x)" — without this, sympy creates Symbol('e').
                text = re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', text)
                # Also handle "e**x" (no parens)
                text = re.sub(r'\be\*\*(\w+)', r'exp(\1)', text)
                # Reject if it contains dangerous names
                if re.search(r'(?:import|exec|eval|open|system|compile|getattr|__|lambda|exit|quit|input|print|breakpoint)', text):
                    return ""
                # Reject excessively large exponents (can hang sympy)
                if re.search(r'\*\*\d{4,}', text):
                    return ""
                # Only allow: digits, word chars, operators, parens, period.
                # Comma is excluded — "1,2" parses as a tuple in sympy, causing
                # TypeError in complex() and nonsensical expand() results.
                if not re.match(r'^[\d\w+\-*/().eE]+$', text):
                    return ""
                # Length limit — very long expressions are likely garbage
                if len(text) > 200:
                    return ""
                return text

            safe_resp = _sanitize(expr_text)
            safe_exp = _sanitize(clean_expected)
            if not safe_resp or not safe_exp:
                return False

            # local_dict={} prevents parse_expr from resolving to arbitrary names
            resp_expr = parse_expr(safe_resp, local_dict={})
            exp_expr = parse_expr(safe_exp, local_dict={})
            # simplify can hang on adversarial inputs — use a weaker but bounded check
            diff = resp_expr - exp_expr
            # Try numeric evaluation first (fast, handles most cases)
            try:
                return abs(complex(diff.evalf())) < 1e-6
            except (TypeError, ValueError):
                # simplify() can hang on complex expressions — impose a generous
                # but bounded attempt via expand() first (cheaper than simplify).
                from sympy import expand
                expanded = expand(diff)
                if expanded == 0:
                    return True
                # simplify() can hang on adversarial inputs — bound it.
                # signal.alarm only works on Unix main thread; fall back to
                # skipping simplify entirely if unavailable (expand already
                # handles most real math expressions).
                import signal
                import threading
                if sys.platform != "win32" and threading.current_thread() is threading.main_thread():
                    def _timeout_handler(signum, frame):
                        raise TimeoutError("sympy simplify timed out")
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(2)
                    try:
                        result = simplify(expanded) == 0
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                    return result
                # Non-main thread or Windows: skip simplify, it's too risky
                return False
        except Exception:
            pass
        return False

    def _check_code_executes(self, response: str) -> bool:
        """Check if code compiles, runs, and the function actually works.

        Just defining a function always succeeds — we need to CALL it with
        basic test inputs to detect runtime errors.
        """
        code = self._extract_code(response)
        if not code:
            return False
        try:
            compile(code, "<string>", "exec")
        except SyntaxError:
            return False

        # Append a basic smoke-test call. Use the LAST top-level function defined,
        # not the first — models often define helpers (is_prime, swap, etc.) before
        # the main solution function. The last function is most likely the entry point.
        # Handle class-based solutions: if the function is a method (has `self`),
        # find the enclosing class and instantiate it first.
        all_funcs = list(re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)', code))
        func_match = all_funcs[-1] if all_funcs else None
        if func_match:
            func_name = func_match.group(1).lower()
            raw_params = func_match.group(2).strip()
            param_list = [p.strip() for p in raw_params.split(",") if p.strip()] if raw_params else []
            # Strip type annotations and defaults for matching — "self: 'Cls'" → "self",
            # "n: int = 5" → "n". Only need the bare name for method detection and counting.
            bare_names = [re.split(r'[=:]', p)[0].strip() for p in param_list]
            is_method = bare_names and bare_names[0] in ("self", "cls")
            param_count = len([n for n in bare_names if n not in ("self", "cls")])

            # For class methods, find enclosing class and call via instance
            fname = func_match.group(1)
            if is_method:
                class_match = re.search(r'class\s+(\w+)', code)
                if class_match:
                    fname = f'{class_match.group(1)}().{func_match.group(1)}'
                # else: standalone def with self param — unusual but just try it
            if any(k in func_name for k in ("palindrome", "string", "str")):
                # Use truthiness, not == True — function may return 1, "racecar", etc.
                test_call = f'assert {fname}("racecar"); assert not {fname}("hello")'
            elif any(k in func_name for k in ("prime", "sieve", "primes")):
                test_call = f'r = {fname}(30); assert 2 in r and 29 in r and 4 not in r'
            elif any(k in func_name for k in ("merge", "sort")):
                test_call = f'assert {fname}([1,3,5], [2,4,6]) == [1,2,3,4,5,6]'
            elif any(k in func_name for k in ("search", "binary")):
                # Expect index 2. Don't include True in the tuple — True == 1 in
                # Python, so `1 in (2, True)` is True, letting index-1 returns pass.
                test_call = f'r = {fname}([1,2,3,4,5], 3); assert r == 2 or r is True, f"got {{r}}"'
            elif param_count == 1:
                test_call = f'{fname}([1,2,3])'
            elif param_count == 0:
                test_call = f'{fname}()'
            else:
                test_call = f'{fname}([1,2,3], [4,5,6])'

            # Only swallow TypeError for generic fallback calls where we're
            # guessing the arg types. For heuristic-matched tests (palindrome,
            # primes, etc.) a TypeError IS a real bug — the function signature
            # doesn't match what it claims to implement.
            is_heuristic_match = any(k in func_name for k in
                ("palindrome", "string", "str", "prime", "sieve", "primes",
                 "merge", "sort", "search", "binary"))
            if is_heuristic_match:
                code = (f"{code}\n\n# Smoke test\n{test_call}\n")
            else:
                code = (f"{code}\n\n# Smoke test\n"
                        f"import sys as _sys\n"
                        f"try:\n    {test_call}\n"
                        f"except TypeError:\n    _sys.exit(0)  # arg type guess, not a real bug\n")

        # Run in subprocess for isolation — keep minimal env so Python stdlib works
        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
            "PYTHONHOME": os.environ.get("PYTHONHOME", ""),
            "PYTHONPATH": "",  # empty, not inherited — block imports of project code
        }
        safe_env = {k: v for k, v in safe_env.items() if v}  # drop empty

        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, timeout=self.config.code_execution_timeout,
                env=safe_env, preexec_fn=_PREEXEC_FN,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _extract_code(self, response: str) -> str:
        # Prefer python-tagged blocks; if multiple, take the longest (most likely real code)
        if "```python" in response:
            parts = response.split("```python")[1:]
            blocks = [p.split("```")[0].strip() for p in parts]
            blocks = [b for b in blocks if b]
            if blocks:
                return max(blocks, key=len)
        if "```" in response:
            parts = response.split("```")
            # Odd-indexed parts are inside fences
            blocks = []
            for i in range(1, len(parts), 2):
                block = parts[i]
                # Strip language tag on the first line (e.g., "python\n...", "js\n...")
                first_nl = block.find("\n")
                if first_nl != -1 and first_nl < 20 and block[:first_nl].strip().lower() in _LANG_TAGS:
                    block = block[first_nl + 1:]
                blocks.append(block.strip())
            blocks = [b for b in blocks if b]
            if blocks:
                return max(blocks, key=len)
        if "def " in response:
            lines = response.split("\n")
            all_code_lines = []
            current_func_lines = []
            in_code = False
            base_indent = 0
            for line in lines:
                # Also capture class definitions so class-based solutions include
                # the class header. Without this, methods are extracted without
                # their class, and smoke tests can't instantiate ClassName().
                # Capture decorators — buffer @lines until we see a def/class.
                # Must check BEFORE class/def so "@dataclass\nclass Foo:" works.
                if line.strip().startswith("@") and not in_code:
                    if not current_func_lines:
                        current_func_lines = []
                    current_func_lines.append(line)
                    continue
                if line.strip().startswith("class ") and not in_code:
                    # Same logic as def — check if current_func_lines has a
                    # complete def/class (not just decorators) before flushing.
                    has_def_or_class = any(
                        l.strip().startswith(("def ", "class ")) for l in current_func_lines)
                    if current_func_lines and has_def_or_class:
                        all_code_lines.extend(current_func_lines)
                        all_code_lines.append("")
                        current_func_lines = []
                    in_code = True
                    base_indent = len(line) - len(line.lstrip())
                    current_func_lines.append(line)
                    continue
                if line.strip().startswith("def "):
                    # If current_func_lines has only decorators (no def yet),
                    # they belong to this def — don't flush them.
                    has_def = any(l.strip().startswith(("def ", "class ")) for l in current_func_lines)
                    if current_func_lines and has_def:
                        all_code_lines.extend(current_func_lines)
                        all_code_lines.append("")  # blank line between functions
                        current_func_lines = []
                    in_code = True
                    base_indent = len(line) - len(line.lstrip())
                    current_func_lines.append(line)
                    continue
                if in_code:
                    if not line.strip():
                        current_func_lines.append(line)
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= base_indent and not line.strip().startswith("def"):
                        in_code = False
                        # Don't break — there may be more functions
                        continue
                    current_func_lines.append(line)
            if current_func_lines:
                all_code_lines.extend(current_func_lines)
            code_text = "\n".join(all_code_lines)
            # Dedent if the code was embedded inside indented prose
            if all_code_lines and all_code_lines[0] != all_code_lines[0].lstrip():
                code_text = textwrap.dedent(code_text)
            return code_text
        return ""
