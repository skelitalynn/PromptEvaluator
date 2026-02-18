from typing import Dict, Optional

from evaluators import PromptEvaluator
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

        for i in range(self.max_iterations):
            iterations = i + 1
            evaluation_raw = evaluator.evaluate(current_prompt)
            evaluation_json = evaluator.parse_json(evaluation_raw)
            self.memory.add(
                "evaluation",
                {
                    "prompt": current_prompt,
                    "raw": evaluation_raw,
                    "json": evaluation_json,
                },
            )

            overall = self._safe_overall(evaluation_json)
            if overall is not None and overall >= self.target_overall:
                final_feedback = "Target score reached."
                break

            reflection_text = REFLECTION_PROMPT.format(
                prompt=current_prompt,
                evaluation=evaluation_raw,
            )
            feedback = self.llm.think([{"role": "user", "content": reflection_text}]) or ""
            self.memory.add("reflection", feedback)
            final_feedback = feedback

            if "Evaluation is reliable." in feedback and overall is not None:
                break

            refine_text = REFINE_PROMPT.format(prompt=current_prompt, feedback=feedback)
            improved_prompt = self.llm.think([{"role": "user", "content": refine_text}]) or ""
            self.memory.add("refined_prompt", improved_prompt)

            if not improved_prompt.strip():
                break
            current_prompt = improved_prompt.strip()

        final_evaluation = self.memory.last("evaluation") or {}
        return {
            "final_prompt": current_prompt,
            "final_evaluation_raw": final_evaluation.get("raw", ""),
            "final_evaluation_json": final_evaluation.get("json", {}),
            "final_feedback": final_feedback,
            "iterations": iterations,
            "memory": self.memory.records,
        }
