from reflection_agent import ReflectionPromptAgent
from tests.fakes import FakeLLM


def test_reflection_stops_when_target_score_reached():
    llm = FakeLLM(
        [
            '{"overall": 9, "problems": "minor"}',
        ]
    )
    agent = ReflectionPromptAgent(llm, max_iterations=3, target_overall=8)
    result = agent.run("Write quicksort")

    assert result["iterations"] == 1
    assert result["final_feedback"] == "Target score reached."
    assert result["final_evaluation_json"]["overall"] == 9


def test_reflection_stops_when_evaluation_is_reliable():
    llm = FakeLLM(
        [
            '{"overall": 6, "problems": "needs constraints"}',
            "Evaluation is reliable.",
        ]
    )
    agent = ReflectionPromptAgent(llm, max_iterations=3, target_overall=8)
    result = agent.run("Write quicksort")

    assert result["iterations"] == 1
    assert result["final_feedback"] == "Evaluation is reliable."
    assert result["final_evaluation_json"]["overall"] == 6


def test_reflection_respects_max_iterations():
    llm = FakeLLM(
        [
            '{"overall": 5, "problems": "too vague"}',
            "Need more detail.",
            "Write a typed Python quicksort and return code only.",
            '{"overall": 6, "problems": "still weak constraints"}',
            "Need stricter output format.",
            "Write quicksort with strict JSON output contract.",
        ]
    )
    agent = ReflectionPromptAgent(llm, max_iterations=2, target_overall=8)
    result = agent.run("Write quicksort")

    assert result["iterations"] == 2
    assert result["final_evaluation_json"]["overall"] == 6
    assert result["final_prompt"] == "Write quicksort with strict JSON output contract."
