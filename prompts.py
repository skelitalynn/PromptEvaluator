# 评分模板（基础版）
# 作用：按固定维度给分，并要求输出 JSON，方便程序解析。
EVALUATION_PROMPT_TEMPLATE = """
You are a professional Prompt Quality Evaluator.

Evaluate the following prompt on a scale of 1-10 in these dimensions:

1. Clarity
2. Specificity
3. Constraints
4. Output Format Definition

Return your answer strictly in JSON format:

{{
  "clarity": score,
  "specificity": score,
  "constraints": score,
  "format_definition": score,
  "overall": score,
  "problems": "short diagnosis",
  "improvement_suggestions": "actionable advice"
}}

Prompt:
{prompt}
"""

# 规划模板（Plan-and-Solve 第 1 阶段）
# 作用：先拆解评估步骤。
PLANNER_PROMPT = """
You are a Prompt Analysis Planner.

Break down how to evaluate the following prompt.
Return a numbered list of evaluation steps.

Prompt:
{prompt}
"""

# 执行模板（Plan-and-Solve 第 2 阶段）
# 作用：针对当前步骤返回分析结果。
EXECUTOR_PROMPT = """
You are evaluating a prompt.

Original Prompt:
{prompt}

Evaluation Plan:
{plan}

Current Step:
{step}

Return only the analysis for this step.
"""

# 综合模板（Plan-and-Solve 收尾阶段）
# 作用：把分步分析汇总为最终 JSON 评分。
SYNTHESIS_PROMPT = """
You are a senior Prompt Quality Evaluator.

Original Prompt:
{prompt}

Step Analyses:
{step_analyses}

Now provide a final scoring result strictly in JSON:
{{
  "clarity": score,
  "specificity": score,
  "constraints": score,
  "format_definition": score,
  "overall": score,
  "problems": "short diagnosis",
  "improvement_suggestions": "actionable advice"
}}
"""

# 反思模板（Reflection 阶段）
# 作用：判断上一轮评估是否可靠。
REFLECTION_PROMPT = """
You are a strict Prompt Quality Reviewer.

Original Prompt:
{prompt}

Previous Evaluation:
{evaluation}

If the evaluation is flawed, explain why.
If acceptable, reply exactly: "Evaluation is reliable."
"""

# 优化模板（Refine 阶段）
# 作用：根据反馈改写原始 prompt，仅返回改写后 prompt。
REFINE_PROMPT = """
Improve the original prompt based on this feedback:

Original Prompt:
{prompt}

Feedback:
{feedback}

Return only the improved prompt.
"""
