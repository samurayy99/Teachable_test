def eval_responses(responses, solution, **args):
    success_count = 0
    for response in responses:
        if is_equivalent(response, solution):
            success_count += 1
    return {"success": success_count / len(responses)}

def is_equivalent(answer, solution):
    # This can be a simple string comparison or a more complex logic
    # For example, for numeric answers, you might want to compare within a tolerance
    try:
        return float(answer.strip()) == float(solution.strip())
    except ValueError:
        # Handle non-numeric comparisons
        return answer.strip().lower() == solution.strip().lower()

def voted_answer(responses):
    # Select the most frequent response
    if not responses:
        return ""
    return max(set(responses), key=responses.count)

def evaluate_accuracy(validation_data, eval_func):
    total_accuracy = 0
    for instance in validation_data:
        responses = generate_model_responses(instance["problem"]) # This should call your LLM
        eval_result = eval_func(responses, instance["solution"])
        total_accuracy += eval_result['success']
        print(f"Problem ID {instance['id']} - Success Rate: {eval_result['success']}")
    return total_accuracy / len(validation_data)

def generate_model_responses(problem):
    # This function should generate responses from your model
    # Placeholder for actual model response
    return ["Model's response to the problem"]