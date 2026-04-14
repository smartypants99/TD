from timedilate.directives import DirectiveGenerator


def test_classify_code_task():
    gen = DirectiveGenerator()
    assert gen.classify_task("Write a Python function to sort a list") == "code"


def test_classify_prose_task():
    gen = DirectiveGenerator()
    assert gen.classify_task("Write an essay about climate change") == "prose"


def test_classify_general_task():
    gen = DirectiveGenerator()
    assert gen.classify_task("Do something") == "general"


def test_get_directives_code():
    gen = DirectiveGenerator()
    directives = gen.get_directives("code")
    assert len(directives) > 0
    assert all(isinstance(d, str) for d in directives)


def test_get_directives_prose():
    gen = DirectiveGenerator()
    directives = gen.get_directives("prose")
    assert len(directives) > 0


def test_cycle_through_directives():
    gen = DirectiveGenerator()
    directives = gen.get_directives("code")
    for i in range(len(directives) + 5):
        d = gen.next_directive("code", i)
        assert isinstance(d, str)


def test_custom_directive_generation_prompt():
    gen = DirectiveGenerator()
    prompt = gen.generate_custom_directive_prompt(
        "code", "Write a sorting function", "def sort(lst): return sorted(lst)"
    )
    assert "improve" in prompt.lower() or "novel" in prompt.lower()
