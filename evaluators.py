import json
import re
from typing import Dict, List, Optional

from prompts import (
    EVALUATION_PROMPT_TEMPLATE,
    EXECUTOR_PROMPT,
    PLANNER_PROMPT,
    SYNTHESIS_PROMPT,
)


class PromptEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, prompt: str) -> str:
        #将用户prompt填入评分模板
        prompt_text = EVALUATION_PROMPT_TEMPLATE.format(prompt=prompt)
        messages = [{"role": "user", "content": prompt_text}]
        #调用think返回原始文本
        return self.llm.think(messages) or ""

    def evaluate_as_json(self, prompt: str) -> Dict:
        raw = self.evaluate(prompt)
        return self.parse_json(raw)

    #容错解析
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

    def plan(self, prompt: str) -> str:
        messages = [{"role": "user", "content": PLANNER_PROMPT.format(prompt=prompt)}]
        return self.llm.think(messages) or ""

    def _extract_steps(self, plan: str) -> List[str]:
        lines = [line.strip() for line in plan.splitlines() if line.strip()]
        numbered = [
            line
            for line in lines
            if re.match(r"^(\d+[\.\)]\s+|[-\*]\s+)", line)
        ]
        return numbered or lines

    def execute(self, prompt: str, plan: str) -> str:
        steps = self._extract_steps(plan)
        history = []

        for step in steps:
            messages = [
                {
                    "role": "user",
                    "content": EXECUTOR_PROMPT.format(prompt=prompt, plan=plan, step=step),
                }
            ]
            result = self.llm.think(messages) or ""
            history.append(f"{step}\n{result}")

        return "\n\n".join(history)

    def evaluate(self, prompt: str) -> Dict:
        plan_text = self.plan(prompt)
        step_analyses = self.execute(prompt, plan_text)
        final_messages = [
            {
                "role": "user",
                "content": SYNTHESIS_PROMPT.format(
                    prompt=prompt,
                    step_analyses=step_analyses,
                ),
            }
        ]
        final_raw = self.llm.think(final_messages) or ""

        return {
            "plan": plan_text,
            "step_analyses": step_analyses,
            "final_raw": final_raw,
            "final_json": PromptEvaluator.parse_json(final_raw),
        }
