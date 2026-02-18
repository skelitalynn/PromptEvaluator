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

PLANNER_PROMPT = """
You are a Prompt Analysis Planner.

Break down how to evaluate the following prompt.
Return a numbered list of evaluation steps.

Prompt:
{prompt}
"""

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

REFLECTION_PROMPT = """
You are a strict Prompt Quality Reviewer.

Original Prompt:
{prompt}

Previous Evaluation:
{evaluation}

If the evaluation is flawed, explain why.
If acceptable, reply exactly: "Evaluation is reliable."
"""

REFINE_PROMPT = """
Improve the original prompt based on this feedback:

Original Prompt:
{prompt}

Feedback:
{feedback}

Return only the improved prompt.
"""
