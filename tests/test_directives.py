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


def test_directive_for_weakness():
    gen = DirectiveGenerator()
    d = gen.directive_for_weakness("correctness")
    assert "bug" in d.lower() or "error" in d.lower() or "fix" in d.lower()
    d = gen.directive_for_weakness("completeness")
    assert "missing" in d.lower() or "add" in d.lower()
    d = gen.directive_for_weakness("quality")
    assert "structure" in d.lower() or "readab" in d.lower()
    d = gen.directive_for_weakness("elegance")
    assert "refactor" in d.lower() or "clean" in d.lower()


def test_directive_for_unknown_weakness():
    gen = DirectiveGenerator()
    d = gen.directive_for_weakness("unknown")
    assert "improve" in d.lower()
