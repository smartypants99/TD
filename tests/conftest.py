import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def clean_meta_files():
    """Ensure no stale meta-learning files affect tests."""
    meta_path = Path(".timedilate_checkpoints/.meta.json")
    if meta_path.exists():
        meta_path.unlink()
    yield
    if meta_path.exists():
        meta_path.unlink()


@pytest.fixture
def sample_prompt():
    return "Write a Python function that reverses a string."


@pytest.fixture
def sample_output():
    return "def reverse_string(s):\n    return s[::-1]"
