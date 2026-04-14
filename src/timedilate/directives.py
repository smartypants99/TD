CODE_DIRECTIVES = [
    "Fix any bugs or logical errors in this code.",
    "Add handling for edge cases (empty input, invalid types, boundary values).",
    "Optimize the performance of this code.",
    "Add comprehensive error handling and input validation.",
    "Refactor for readability and clean code principles.",
    "Explore an alternative algorithmic approach.",
    "Add inline comments explaining complex logic.",
    "Stress-test: what happens with very large inputs?",
    "Add useful features that serve the original goal.",
]

PROSE_DIRECTIVES = [
    "Improve clarity and conciseness of the writing.",
    "Strengthen the arguments with better evidence or reasoning.",
    "Add concrete examples to illustrate key points.",
    "Consider and address counterarguments.",
    "Improve the overall structure and flow.",
    "Refine the tone to better match the intended audience.",
    "Deepen the analysis where it's currently surface-level.",
    "Remove redundancy and filler.",
    "Add a stronger opening and conclusion.",
]

GENERAL_DIRECTIVES = [
    "Improve the overall quality and completeness.",
    "Fix any errors or inaccuracies.",
    "Add more detail where it's needed.",
    "Consider alternative approaches.",
    "Make the output more useful for the end user.",
    "Improve structure and organization.",
    "Add examples or illustrations.",
    "Polish and refine the output.",
]

TASK_KEYWORDS = {
    "code": [
        "function", "code", "program", "script", "implement", "build",
        "create a ", "write a ", "class", "api", "app", "game", "website",
        "algorithm", "roblox", "python", "javascript",
    ],
    "prose": [
        "essay", "write about", "explain", "describe", "article", "blog",
        "letter", "email", "story", "poem", "report", "an essay",
    ],
}


class DirectiveGenerator:
    def classify_task(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        scores = {"code": 0, "prose": 0}
        for task_type, keywords in TASK_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower:
                    scores[task_type] += 1
        if scores["code"] > scores["prose"]:
            return "code"
        if scores["prose"] > scores["code"]:
            return "prose"
        return "general"

    def get_directives(self, task_type: str) -> list[str]:
        if task_type == "code":
            return CODE_DIRECTIVES
        if task_type == "prose":
            return PROSE_DIRECTIVES
        return GENERAL_DIRECTIVES

    def next_directive(self, task_type: str, cycle_index: int) -> str:
        directives = self.get_directives(task_type)
        return directives[cycle_index % len(directives)]

    def directive_for_weakness(self, weakest_aspect: str) -> str:
        """Generate a directive that targets a specific weakness."""
        aspect_directives = {
            "correctness": "Focus on fixing any bugs, logical errors, or factual inaccuracies. Verify every claim and every code path.",
            "completeness": "The output is missing important parts. Add anything that was asked for but not delivered. Cover all aspects of the task.",
            "quality": "Improve structure, readability, and polish. Better formatting, clearer organization, more professional presentation.",
            "elegance": "Refactor for cleaner design. Simplify where possible, use better patterns, make the approach more efficient and well-designed.",
        }
        return aspect_directives.get(weakest_aspect, "Improve the overall quality.")

    def generate_custom_directive_prompt(
        self, task_type: str, original_prompt: str, current_output: str
    ) -> str:
        return (
            f"Given the task: {original_prompt}\n\n"
            f"And the current output:\n{current_output}\n\n"
            f"The following standard improvements have already been applied multiple times.\n"
            f"Suggest ONE novel, specific improvement that would make this output meaningfully better.\n"
            f"Be creative — think of something not on this list: {', '.join(self.get_directives(task_type))}\n"
            f"Respond with just the improvement directive, one sentence, nothing else."
        )
