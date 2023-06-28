import streamlit as st
import utils
from transformers import Conversation, pipeline

chatbot = pipeline(
    "conversational", model="facebook/blenderbot-400M-distill", max_length=1000
)


@st.cache_data()  # Decorator suggests based on input args return values of func are cached
def monolith_llm_response(user_input):
    """Run the user input through the LLM and return the response."""
    # Step 1: Initialize the conversation history
    conversation = Conversation(**st.session_state.conversation_history)

    # Step 2: Add the latest user input
    conversation.add_user_input(user_input)

    # Step 3: Generate a response
    _ = chatbot(conversation)  # '_' is throwaway var, used when actual value is not needed


def main():
    """Builds frontend app's layout using Streamlit"""
    st.title("Monolithic ChatBot App")

    col1, col2 = st.columns(2)  # Creates 2 col and assigns them
    with col1:
        utils.clear_conversation()

    # Get user input
    # "walrus operator"(:=) assigns user text if entered to var 'user_input'
    if user_input := st.text_input("Ask your question 👇", key="user_input"):
        monolith_llm_response(user_input)

    # Display the entire conversation on the frontend
    utils.display_conversation(st.session_state.conversation_history)

    # Download conversation code runs last to ensure the latest messages are captured
    with col2:
        utils.download_conversation()


if __name__ == "__main__":
    main()
