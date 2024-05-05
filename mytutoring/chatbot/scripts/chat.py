import random
import json
import os
import torch
from .model import NeuralNet

from .nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get the directory where this script is located
current_directory = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(current_directory, 'intents.json')

with open(intents_path, 'r') as json_data:
    intents = json.load(json_data)

data_path = os.path.join(current_directory, 'data.pth')
data = torch.load(data_path)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Constants for accessing MCQ data
MCQ_TAGS = {
    "python": "python_mcq",
    "nodejs": "nodejs_mcq",
    "fullstack": "fullstack_mcq"
}

# Function to fetch an MCQ based on a specific topic
def get_mcq_by_topic(topic):
    tag = MCQ_TAGS.get(topic)
    # print('tag', topic)
    tag = topic
    if tag:
        print('intents', intents)
        for intent in intents['intents']:
            print('tag', tag in intent['tag'])

            if tag in intent['tag']:
                # Select a random question from the intent
                return random.choice(intent['responses'])
    return None

# Function to check if the user's response is correct
def check_mcq_answer(user_answer, question_info):
    correct_answer = question_info["correct"]
    explanation = question_info["explanation"]

    # Determine if the user's answer matches the correct one
    if user_answer.strip().lower() == correct_answer.strip().lower():
        return f"Correct! {explanation}"
    else:
        return f"Incorrect. The correct answer is {correct_answer}. {explanation}"


def get_response(msg, previous_question=None):
    print("previous question", previous_question)

    if previous_question:
        # The user is answering an MCQ, check the response
        return check_mcq_answer(msg, previous_question)
    
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # Check if the tag corresponds to a known MCQ topic
        if tag in MCQ_TAGS.values():
            print("check", tag)
            question = get_mcq_by_topic(tag)
            print("question", question)
            if question:
                # Return the question with the choices
                return {
                    "question": question["question"],
                    "choices": question["choices"],
                    "context": question
                }
            
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    previous_question = None
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence, previous_question)

        if isinstance(resp, dict):
            # If the response contains a question, it's an MCQ
            print("Here's your question:")
            print(resp["question"])
            print("Options:", ", ".join(resp["choices"]))
            print(resp)
            previous_question = resp.get("context", None)
        else:
            # If it's just a simple text response
            print(resp)
            previous_question = None  # Reset the context for new questions

