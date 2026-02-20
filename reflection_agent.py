from typing import Dict, Optional

from evaluators import PromptEvaluator
from llm_helpers import call_llm_safe
from prompts import REFINE_PROMPT, REFLECTION_PROMPT


class Memory:
    def __init__(self):
        self.records = []

    def add(self, record_type: str, content):
        self.records.append({"type": record_type, "content": content})

    def last(self, record_type: str):
        for record in reversed(self.records):
            if record["type"] == record_type:
                return record["content"]
        return None


class ReflectionPromptAgent:
    def __init__(self, llm, max_iterations: int = 2, target_overall: int = 8):
        self.llm = llm
        self.memory = Memory()
        self.max_iterations = max_iterations
        self.target_overall = target_overall

    @staticmethod
    def _safe_overall(evaluation_json: Dict) -> Optional[float]:
        if not evaluation_json:
            return None
        overall = evaluation_json.get("overall")
        try:
            return float(overall)
        except (TypeError, ValueError):
            return None

    def run(self, prompt: str) -> Dict:
        evaluator = PromptEvaluator(self.llm)
        current_prompt = prompt
        final_feedback = ""
        iterations = 0
        ok = True
        error_type = None
        error_message = ""

        for i in range(self.max_iterations):
            iterations = i + 1

            evaluation_call = evaluator.evaluate_result(current_prompt)
            evaluation_raw = evaluation_call["content"]
            evaluation_json = evaluator.parse_json(evaluation_raw)
            self.memory.add(
                "evaluation",
                {
                    "prompt": current_prompt,
                    "raw": evaluation_raw,
                    "json": evaluation_json,
                    "call": evaluation_call,
                },
            )

            if not evaluation_call["ok"]:
                ok = False
                error_type = evaluation_call["error_type"]
                error_message = evaluation_call["error_message"]
                final_feedback = f"Evaluation failed: {error_type}"
                break

            overall = self._safe_overall(evaluation_json)
            if overall is not None and overall >= self.target_overall:
                final_feedback = "Target score reached."
                break

            reflection_text = REFLECTION_PROMPT.format(
                prompt=current_prompt,
                evaluation=evaluation_raw,
            )
            feedback_call = call_llm_safe(self.llm, [{"role": "user", "content": reflection_text}])
            feedback = feedback_call["content"]
            self.memory.add("reflection", {"text": feedback, "call": feedback_call})
            final_feedback = feedback

            if not feedback_call["ok"]:
                ok = False
                error_type = feedback_call["error_type"]
                error_message = feedback_call["error_message"]
                final_feedback = f"Reflection failed: {error_type}"
                break

            if "Evaluation is reliable." in feedback and overall is not None:
                break

            refine_text = REFINE_PROMPT.format(prompt=current_prompt, feedback=feedback)
            refine_call = call_llm_safe(self.llm, [{"role": "user", "content": refine_text}])
            improved_prompt = refine_call["content"]
            self.memory.add("refined_prompt", {"text": improved_prompt, "call": refine_call})

            if not refine_call["ok"]:
                ok = False
                error_type = refine_call["error_type"]
                error_message = refine_call["error_message"]
                final_feedback = f"Refinement failed: {error_type}"
                break

            if not improved_prompt.strip():
                break
            current_prompt = improved_prompt.strip()

        final_evaluation = self.memory.last("evaluation") or {}
        return {
            "ok": ok,
            "error_type": error_type,
            "error_message": error_message,
            "final_prompt": current_prompt,
            "final_evaluation_raw": final_evaluation.get("raw", ""),
            "final_evaluation_json": final_evaluation.get("json", {}),
            "final_feedback": final_feedback,
            "iterations": iterations,
            "memory": self.memory.records,
        }
