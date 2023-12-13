import json
from evaluation_script import evaluate_accuracy

# Load validation data
def load_validation_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    validation_data = load_validation_data('testing/validation_data.json')
    overall_accuracy = evaluate_accuracy(validation_data, eval_responses)
    print(f"Overall Accuracy: {overall_accuracy}")