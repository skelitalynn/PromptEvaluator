import json

from HelloAgentsLLM import HelloAgentsLLM
from evaluators import PlanAndSolveEvaluator, PromptEvaluator
from reflection_agent import ReflectionPromptAgent


def run_basic(llm):
    evaluator = PromptEvaluator(llm)
    user_prompt = input("Enter a prompt to evaluate:\n").strip()
    result = evaluator.evaluate_result(user_prompt)
    print("\nEvaluation Result:")
    if result["ok"]:
        print(result["content"])
    else:
        print(f"Error: {result['error_type']} - {result['error_message']}")


def run_plan_and_solve(llm):
    evaluator = PlanAndSolveEvaluator(llm)
    user_prompt = input("Enter a prompt to evaluate with Plan-and-Solve:\n").strip()
    result = evaluator.evaluate(user_prompt)

    if not result["ok"]:
        print("\nPlan-and-Solve failed:")
        print(f"Error: {result['error_type']} - {result['error_message']}")
        if result["errors"]:
            print("Error details:")
            for item in result["errors"]:
                print(json.dumps(item, ensure_ascii=False))
        return

    print("\nPlan:")
    print(result["plan"])
    print("\nStep Analyses:")
    print(result["step_analyses"])
    print("\nFinal Evaluation (Raw):")
    print(result["final_raw"])
    print("\nFinal Evaluation (JSON parsed):")
    print(json.dumps(result["final_json"], ensure_ascii=False, indent=2))


def run_reflection(llm):
    user_prompt = input("Enter a prompt to optimize with Reflection Agent:\n").strip()
    agent = ReflectionPromptAgent(llm, max_iterations=3, target_overall=8)
    result = agent.run(user_prompt)

    if not result["ok"]:
        print("\nReflection run failed:")
        print(f"Error: {result['error_type']} - {result['error_message']}")

    print(f"\nIterations: {result['iterations']}")
    print("\nFinal Prompt:")
    print(result["final_prompt"])
    print("\nFinal Feedback:")
    print(result["final_feedback"])
    print("\nFinal Evaluation (Raw):")
    print(result["final_evaluation_raw"])
    print("\nFinal Evaluation (JSON parsed):")
    print(json.dumps(result["final_evaluation_json"], ensure_ascii=False, indent=2))


def main():
    llm = HelloAgentsLLM()

    print("Prompt Evaluator & Refiner")
    print("1. Basic Evaluator (Single Call)")
    print("2. Plan-and-Solve Evaluator")
    print("3. Reflection Agent (Iterative Refinement)")
    choice = input("Choose mode (1/2/3): ").strip()

    if choice == "1":
        run_basic(llm)
    elif choice == "2":
        run_plan_and_solve(llm)
    elif choice == "3":
        run_reflection(llm)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
