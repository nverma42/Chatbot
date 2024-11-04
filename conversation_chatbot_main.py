"""
conversation_chatbot_main.py

This script serves as the entry point for
    running the Mental Health Chatbot, handling user interaction, and
managing the runtime logic.
    The chatbot provides mental health support by processing user queries and
generating appropriate responses,
    based on whether the query is informational or emotional.

Key Features:
- The chatbot categorizes user queries as either informational
    (related to mental health) or emotional (support-seeking).
- Informational queries retrieve answers from a pre-defined FAQ dataset.
- Emotional queries are mapped to conversation
    graphs derived from counseling sessions, allowing for multi-turn conversations.

Runtime Logic:
- Initializes the MentalHealthChatbot class, loads necessary datasets,
    and trains classifiers for query categorization
  and emotion prediction.
- Runs an interactive loop where users can input queries and
    receive real-time responses from the chatbot.
- Expanded termination options allow users to exit by typing any of the following:
    'exit', 'quit', 'q', 'x', 'e', or by
  pressing 'Ctrl+C' (KeyboardInterrupt).

Usage:
- Run the script to start an interactive session with the chatbot.
- Input your query when prompted, and the chatbot will provide an
    appropriate response based on the type of query.
- To exit the session, type 'exit', 'quit', 'q', 'x', or 'e',
    or press 'Ctrl+C' for a graceful termination.

Note:
- Ensure that all necessary datasets
    (FAQ, empathetic dialogues, counseling conversations) are available and paths
  are correctly specified using command-line flags or default values.
"""
from absl import app, flags
from mental_health_chatbot import MentalHealthChatbot
import torch
import logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'faq_data_path', './data/Mental_Health_FAQ.csv', 'Path to the FAQ dataset')
flags.DEFINE_string(
    'conversations_data_path',
    'hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json',
    'Path to the mental health counseling conversations dataset')
flags.DEFINE_float('test_size', 0.3, 'Test set size as a fraction')
flags.DEFINE_integer('random_state', 42, 'Random seed for reproducibility')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_device():
    """
    Checks for available CUDA devices and returns the appropriate device string.

    Returns:
        device (str): The device string ('cuda:X' or 'cpu').
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA is available. Number of GPUs: {num_gpus}")
        # For simplicity, we'll use the GPU with the most available memory.
        # Alternatively, we could distribute models across multiple GPUs.
        # Here we select the GPU with the lowest memory usage.
        free_memory = []
        for idx in range(num_gpus):
            props = torch.cuda.get_device_properties(idx)
            # Assuming that more free memory indicates less usage
            free_memory.append((idx, props.total_memory))
        # Sort GPUs based on free memory (descending)
        sorted_gpus = sorted(free_memory, key=lambda x: x[1], reverse=True)
        selected_gpu = sorted_gpus[0][0]
        device = f'cuda:{selected_gpu}'
        logger.info(f"Selected GPU: {device}")
    else:
        logger.info("CUDA is not available. Using CPU.")
        device = 'cpu'
    return device


def main(argv):
    """
    Runs the interactive loop for the Mental Health Chatbot.

    The chatbot processes user input and categorizes it into informational or emotional types.
    """
    device = get_available_device()

    chatbot = MentalHealthChatbot(
        faq_data_path=FLAGS.faq_data_path,
        conversations_data_path=FLAGS.conversations_data_path,
        test_size=FLAGS.test_size,
        random_state=FLAGS.random_state,
        device=device  # Pass the device to the chatbot
    )

    # Load data and train models
    chatbot.load_data()
    chatbot.preprocess_data()
    chatbot.train_logistic_classifier()
    chatbot.build_knn_classifier()

    # Start the interactive loop
    print("Welcome to the Mental Health Chatbot. Type 'exit', 'quit', 'q', 'x', or press 'Ctrl+C' to quit.")
    try:
        while True:
            user_input = input("You: ")
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q', 'x', 'e']:
                print("Chatbot: Take care!")
                break
            # Get response from the chatbot
            response = chatbot.respond_to_query(user_input)
            print(f"Chatbot: {response}")
    except KeyboardInterrupt:
        print("\nChatbot: Exiting. Take care!")  # Graceful exit on Ctrl+C


if __name__ == '__main__':
    app.run(main)
