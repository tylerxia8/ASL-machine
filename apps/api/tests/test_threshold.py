"""Threshold policy tests — mirror web logic."""


def evaluate(prompt: str, predicted: str, confidence: float) -> str:
    if predicted != prompt:
        return "fail"
    if confidence >= 0.9:
        return "pass"
    if confidence >= 0.7:
        return "retry"
    return "fail"


def test_wrong_label():
    assert evaluate("hello", "goodbye", 0.99) == "fail"


def test_pass():
    assert evaluate("hello", "hello", 0.91) == "pass"


def test_retry():
    assert evaluate("hello", "hello", 0.75) == "retry"


def test_low_confidence():
    assert evaluate("hello", "hello", 0.5) == "fail"
