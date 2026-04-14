import pytest


@pytest.fixture
def sample_prompt():
    return "Write a Python function that reverses a string."


@pytest.fixture
def sample_output():
    return "def reverse_string(s):\n    return s[::-1]"
