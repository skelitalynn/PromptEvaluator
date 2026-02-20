from evaluators import PlanAndSolveEvaluator, PromptEvaluator
from tests.fakes import FakeLLM


def test_parse_json_with_plain_json():
    text = '{"overall": 8, "clarity": 9}'
    parsed = PromptEvaluator.parse_json(text)
    assert parsed["overall"] == 8
    assert parsed["clarity"] == 9


def test_parse_json_with_wrapped_text():
    text = 'Result:\n{"overall": 6, "problems": "too vague"}\nThanks'
    parsed = PromptEvaluator.parse_json(text)
    assert parsed["overall"] == 6
    assert parsed["problems"] == "too vague"


def test_parse_json_invalid_returns_empty_dict():
    text = "not a json payload"
    assert PromptEvaluator.parse_json(text) == {}


def test_extract_steps_prefers_numbered_or_bulleted_lines():
    evaluator = PlanAndSolveEvaluator(llm=FakeLLM([]))
    plan = (
        "Plan for review\n"
        "1. Check clarity\n"
        "2) Check constraints\n"
        "- Check output format\n"
        "Summary line"
    )
    steps = evaluator._extract_steps(plan)
    assert steps == [
        "1. Check clarity",
        "2) Check constraints",
        "- Check output format",
    ]


def test_extract_steps_fallback_to_non_empty_lines():
    evaluator = PlanAndSolveEvaluator(llm=FakeLLM([]))
    plan = "First line\nSecond line\n\nThird line"
    steps = evaluator._extract_steps(plan)
    assert steps == ["First line", "Second line", "Third line"]


def test_plan_and_solve_evaluate_returns_expected_shape():
    llm = FakeLLM(
        [
            "1. Check clarity\n2. Check specificity",
            "clarity analysis",
            "specificity analysis",
            '{"overall": 7, "problems": "ok", "clarity": 7}',
        ]
    )
    evaluator = PlanAndSolveEvaluator(llm)
    result = evaluator.evaluate("Write an article about AI")

    assert "plan" in result
    assert "step_analyses" in result
    assert "final_raw" in result
    assert "final_json" in result
    assert result["final_json"]["overall"] == 7
