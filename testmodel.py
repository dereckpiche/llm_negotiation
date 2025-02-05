import vllm

# Initialize the model
model = vllm.LLM(model='meta-llama/Meta-Llama-3.1-8B-Instruct')  # Replace with your desired model

def chat_with_model():
    print("Start chatting with the model! Type 'exit' to stop.")
    conversation_history = []  # List to store the conversation history

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # Append user input to the conversation history
        conversation_history.append(f"You: {user_input}")
        
        # Generate a response from the model using the conversation history
        response = model.generate("\n".join(conversation_history))
        
        # Append model response to the conversation history
        conversation_history.append(f"Model: {response}")
        
        print(f"Model: {response}")

if __name__ == "__main__":
    chat_with_model()
