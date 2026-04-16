"""Tests for the prompt template system."""

import uuid
from unittest.mock import patch

from timedilate.prompts import DEFAULT_TEMPLATES, PromptTemplates


class TestDefaultTemplatesMatchHardcoded:
    """Verify that default templates produce the exact same output as the
    old hardcoded f-strings that were in the controller."""

    def test_score_template(self):
        fence = "abc123"
        prompt = "Write a poem"
        output = "Roses are red"
        dim_list = "  accuracy (weight 30%): 0-100"

        result = DEFAULT_TEMPLATES.effective_score.format_map(
            {"fence": fence, "prompt": prompt, "output": output,
             "dim_list": dim_list}
        )
        # Verify key structural elements are present
        assert f"<<<TASK {fence}>>>" in result
        assert f"<<<RESPONSE {fence}>>>" in result
        assert prompt in result
        assert output in result
        assert "UNTRUSTED DATA" in result
        assert "MUST be ignored" in result
        assert "Rubric (anti-inflation)" in result
        assert dim_list in result

    def test_score_template_contains_injection_fence(self):
        tpl = DEFAULT_TEMPLATES.effective_score
        assert "{fence}" in tpl
        assert "UNTRUSTED DATA" in tpl
        assert "MUST be ignored" in tpl

    def test_critique_template(self):
        cycle = 3
        score = 65
        prompt = "Explain gravity"
        output = "Gravity pulls things down"
        history_summary = "Cycle 1: basic, Cycle 2: added formula"
        cot = "Think step by step about what's wrong and what's missing. "

        expected = (
            f"You are reviewing an AI response at reasoning cycle #{cycle} "
            f"(current score: {score}/100).\n\n"
            f"TASK: {prompt}\n\n"
            f"RESPONSE:\n{output}\n\n"
            f"Prior attempts: {history_summary}\n\n"
            f"{cot}"
            f"List the specific weaknesses, errors, and missing elements. "
            f"Be concrete and actionable \u2014 what exactly should be fixed? "
            f"Do NOT re-suggest directions that have already been tried and failed."
        )

        result = DEFAULT_TEMPLATES.effective_critique.format_map(
            {"cycle": cycle, "score": score, "prompt": prompt,
             "output": output, "history_summary": history_summary,
             "cot_instruction": cot}
        )
        assert result == expected

    def test_critique_template_no_cot(self):
        result = DEFAULT_TEMPLATES.effective_critique.format_map(
            {"cycle": 1, "score": 50, "prompt": "X", "output": "Y",
             "history_summary": "none", "cot_instruction": ""}
        )
        assert "Think step by step" not in result

    def test_refine_template(self):
        cycle = 4
        critique = "Missing examples"
        prompt = "Explain recursion"
        output = "Recursion is..."
        history_summary = "Cycle 1-3: gradual improvement"

        expected = (
            f"This is reasoning cycle #{cycle}. You previously produced a response "
            f"that received this critique:\n\n"
            f"CRITIQUE:\n{critique}\n\n"
            f"ORIGINAL TASK: {prompt}\n\n"
            f"PREVIOUS RESPONSE:\n{output}\n\n"
            f"Recent attempts: {history_summary}\n\n"
            f"Write an improved version that addresses every point in the critique. "
            f"Keep what was good, fix what was bad, add what was missing. "
            f"Do not merely restate the previous response with cosmetic changes \u2014 "
            f"make substantive improvements. Avoid directions already shown to fail."
        )

        result = DEFAULT_TEMPLATES.effective_refine.format_map(
            {"cycle": cycle, "critique": critique, "prompt": prompt,
             "output": output, "history_summary": history_summary}
        )
        assert result == expected

    def test_fresh_template(self):
        cycle = 8
        score = 72
        prompt = "Design an API"
        output = "REST endpoints..."
        history_summary = "Tried REST, tried GraphQL"

        expected = (
            f"At reasoning cycle #{cycle}, previous attempts plateaued at "
            f"score {score}/100.\n\n"
            f"TASK: {prompt}\n\n"
            f"What has been tried (do NOT repeat these directions):\n"
            f"{history_summary}\n\n"
            f"Best attempt so far (shown ONLY so you know what to avoid \u2014 "
            f"do not paraphrase or lightly edit it):\n{output}\n\n"
            f"Take a fundamentally different approach. Rethink the problem "
            f"from scratch using a different framing, structure, or strategy. "
            f"Do not iterate on the previous attempt, do not reuse its phrasing, "
            f"and do not simply expand on the same ideas. Produce a genuinely "
            f"novel solution."
        )

        result = DEFAULT_TEMPLATES.effective_fresh.format_map(
            {"cycle": cycle, "score": score, "prompt": prompt,
             "output": output, "history_summary": history_summary}
        )
        assert result == expected


class TestPromptTemplatesOverride:
    """Verify that custom templates override defaults while leaving others intact."""

    def test_override_single_template(self):
        custom = PromptTemplates(score="Rate {prompt}: {output} (fence={fence})")
        assert custom.effective_score == "Rate {prompt}: {output} (fence={fence})"
        # Others still default
        assert custom.effective_critique == DEFAULT_TEMPLATES.effective_critique

    def test_override_all_templates(self):
        custom = PromptTemplates(
            score="S {prompt}",
            critique="C {prompt}",
            refine="R {prompt}",
            fresh="F {prompt}",
        )
        assert custom.effective_score == "S {prompt}"
        assert custom.effective_critique == "C {prompt}"
        assert custom.effective_refine == "R {prompt}"
        assert custom.effective_fresh == "F {prompt}"

    def test_none_falls_back_to_default(self):
        custom = PromptTemplates()
        assert custom.effective_score == DEFAULT_TEMPLATES.effective_score
        assert custom.effective_critique == DEFAULT_TEMPLATES.effective_critique
        assert custom.effective_refine == DEFAULT_TEMPLATES.effective_refine
        assert custom.effective_fresh == DEFAULT_TEMPLATES.effective_fresh


class TestConfigIntegration:
    """Verify that TimeDilateConfig accepts prompt_templates."""

    def test_config_default_none(self):
        from timedilate.config import TimeDilateConfig
        cfg = TimeDilateConfig()
        assert cfg.prompt_templates is None

    def test_config_accepts_templates(self):
        from timedilate.config import TimeDilateConfig
        tpl = PromptTemplates(score="custom {prompt} {output} {fence}")
        cfg = TimeDilateConfig(prompt_templates=tpl)
        assert cfg.prompt_templates is tpl
