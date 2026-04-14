import re


class Scorer:
    RUBRIC = (
        "Rate the following output on a scale of 0-100 based on these criteria:\n"
        "- Correctness: Does it accurately address the task? (0-25)\n"
        "- Completeness: Does it fully address all aspects? (0-25)\n"
        "- Quality: Is it well-structured and polished? (0-25)\n"
        "- Elegance: Is the approach clean and well-designed? (0-25)\n\n"
        "Respond with ONLY a single integer 0-100. Nothing else."
    )

    def build_scoring_prompt(self, original_prompt: str, output: str) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"Output to score:\n{output}\n\n"
            f"{self.RUBRIC}"
        )

    def parse_score(self, raw: str) -> int:
        numbers = re.findall(r"\d+", raw)
        if not numbers:
            return 0
        score = int(numbers[0])
        return max(0, min(100, score))
