# Prompt Evaluator & Refiner

An MVP project for Chapter 4 style agent design:

1. Scoring
2. Diagnosis
3. Refinement

Includes:

- Basic evaluator (single call)
- Plan-and-Solve evaluator (plan then execute)
- Reflection agent (evaluate -> reflect -> refine -> re-evaluate)

## 4.1 Environment Setup

### Install dependencies

```bash
pip install openai python-dotenv
```

### Configure `.env`

```env
LLM_API_KEY="xxx"
LLM_MODEL_ID="xxx"
LLM_BASE_URL="xxx"
LLM_TIMEOUT="60"
```

The project reuses `HelloAgentsLLM` and only calls:

```python
llm.think(messages)
```

## 4.2 Basic Version

`PromptEvaluator` sends a single evaluation prompt and returns result text.

## 4.3 Plan-and-Solve Version

`PlanAndSolveEvaluator` runs:

1. Planner phase: generate evaluation steps
2. Executor phase: analyze each step
3. Synthesis phase: produce final JSON scoring

## 4.4 Reflection Upgrade

`ReflectionPromptAgent` runs iterative optimization:

1. Evaluate prompt
2. Reflect on evaluation quality
3. Refine prompt
4. Re-evaluate until target score or iteration limit

## Run

```bash
python main.py
```

Then choose mode:

- `1` Basic evaluator
- `2` Plan-and-Solve evaluator
- `3` Reflection agent
