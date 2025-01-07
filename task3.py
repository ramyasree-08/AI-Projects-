import numpy as np
print("Warning: TensorFlow is not installed in this environment. Switching to a simulated environment.")
data = {
    "hello": "Hi there!",
    "how are you?": "I'm good, thank you! How about you?",
    "what's your name?": "I'm ChatBot, your assistant.",
    "what can you do?": "I can help you with your questions.",
    "bye": "Goodbye! Have a great day!"
}

def chatbot_response(input_text):
    input_text = input_text.lower()
    response = data.get(input_text, "I'm sorry, I didn't understand that.")
    return response
def simulate_chat():
    simulated_inputs = [
        "hello",
        "how are you?",
        "what's your name?",
        "what can you do?",
        "bye"
    ]

    for user_input in simulated_inputs:
        print(f"You: {user_input}")
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("ChatBot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"ChatBot: {response}")

simulate_chat()
