import json
import re
from typing import Dict, List

from llm_helpers import call_llm_safe
from prompts import (
    EVALUATION_PROMPT_TEMPLATE,
    EXECUTOR_PROMPT,
    PLANNER_PROMPT,
    SYNTHESIS_PROMPT,
)


class PromptEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate_result(self, prompt: str) -> Dict:
        prompt_text = EVALUATION_PROMPT_TEMPLATE.format(prompt=prompt)
        messages = [{"role": "user", "content": prompt_text}]
        return call_llm_safe(self.llm, messages)

    def evaluate(self, prompt: str) -> str:
        result = self.evaluate_result(prompt)
        return result["content"]

    def evaluate_as_json(self, prompt: str) -> Dict:
        raw = self.evaluate(prompt)
        return self.parse_json(raw)

    @staticmethod
    def parse_json(text: str) -> Dict:
        if not text:
            return {}
        cleaned = text.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return {}

        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}


class PlanAndSolveEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def plan_result(self, prompt: str) -> Dict:
        messages = [{"role": "user", "content": PLANNER_PROMPT.format(prompt=prompt)}]
        return call_llm_safe(self.llm, messages)

    def plan(self, prompt: str) -> str:
        result = self.plan_result(prompt)
        return result["content"]

    def _extract_steps(self, plan: str) -> List[str]:
        lines = [line.strip() for line in plan.splitlines() if line.strip()]
        numbered = [
            line
            for line in lines
            if re.match(r"^(\d+[\.\)]\s+|[-\*]\s+)", line)
        ]
        return numbered or lines

    def execute_result(self, prompt: str, plan: str) -> Dict:
        steps = self._extract_steps(plan)
        history = []
        errors = []

        for step in steps:
            messages = [
                {
                    "role": "user",
                    "content": EXECUTOR_PROMPT.format(prompt=prompt, plan=plan, step=step),
                }
            ]
            result = call_llm_safe(self.llm, messages)
            history.append(f"{step}\n{result['content']}")
            if not result["ok"]:
                errors.append(
                    {
                        "step": step,
                        "error_type": result["error_type"],
                        "error_message": result["error_message"],
                        "attempts": result["attempts"],
                    }
                )

        return {
            "ok": len(errors) == 0,
            "content": "\n\n".join(history),
            "errors": errors,
        }

    def execute(self, prompt: str, plan: str) -> str:
        result = self.execute_result(prompt, plan)
        return result["content"]

    def evaluate(self, prompt: str) -> Dict:
        plan_call = self.plan_result(prompt)
        plan_text = plan_call["content"]

        if not plan_call["ok"]:
            return {
                "ok": False,
                "error_type": plan_call["error_type"],
                "error_message": plan_call["error_message"],
                "plan": "",
                "step_analyses": "",
                "final_raw": "",
                "final_json": {},
                "errors": [
                    {
                        "stage": "plan",
                        "error_type": plan_call["error_type"],
                        "error_message": plan_call["error_message"],
                        "attempts": plan_call["attempts"],
                    }
                ],
            }

        execute_call = self.execute_result(prompt, plan_text)
        step_analyses = execute_call["content"]
        final_messages = [
            {
                "role": "user",
                "content": SYNTHESIS_PROMPT.format(
                    prompt=prompt,
                    step_analyses=step_analyses,
                ),
            }
        ]
        final_call = call_llm_safe(self.llm, final_messages)
        final_raw = final_call["content"]

        errors = list(execute_call["errors"])
        if not final_call["ok"]:
            errors.append(
                {
                    "stage": "synthesis",
                    "error_type": final_call["error_type"],
                    "error_message": final_call["error_message"],
                    "attempts": final_call["attempts"],
                }
            )

        return {
            "ok": len(errors) == 0,
            "error_type": errors[0]["error_type"] if errors else None,
            "error_message": errors[0]["error_message"] if errors else "",
            "plan": plan_text,
            "step_analyses": step_analyses,
            "final_raw": final_raw,
            "final_json": PromptEvaluator.parse_json(final_raw),
            "errors": errors,
        }
