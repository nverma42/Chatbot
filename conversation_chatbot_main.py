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
from summarization_engine import SummarizationEngine
import torch
import logging
import subprocess
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'faq_data_path', './data/Mental_Health_FAQ.csv', 'Path to the FAQ dataset')
flags.DEFINE_string(
    'sentence_encoder', 'paraphrase-MiniLM-L6-v2', 'string of the Sentence Transform')
flags.DEFINE_string(
    'conversations_data_path',
    'hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json',
    'Path to the mental health counseling conversations dataset')
flags.DEFINE_float('test_size', 0.3, 'Test set size as a fraction')
flags.DEFINE_integer('random_state', 42, 'Random seed for reproducibility')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('networkx').setLevel(logging.WARNING)
logging.getLogger('gensim').setLevel(logging.WARNING)

def get_gpu_memory_info():
    """
    Uses nvidia-smi to get accurate memory usage for each GPU.

    Returns:
        List[Tuple[int, float]]: List of tuples containing (GPU ID, free memory in GB).
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
        memory_info = result.decode('utf-8').strip().split('\n')
        gpu_info = [(i, float(mem)) for i, mem in enumerate(memory_info)]
        return gpu_info
    except subprocess.CalledProcessError as e:
        logger.error("Failed to run nvidia-smi: " + str(e))
        return []


def get_best_available_device(min_memory_required=4.0):
    """
    Checks available GPUs using nvidia-smi and selects the best GPU based on free memory.
    Falls back to CPU if no GPU has enough available memory.

    Args:
        min_memory_required (float): Minimum required free memory in GB.

    Returns:
        str: The best device string ('cuda:X' or 'cpu').
    """
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        available_gpus = [(gpu_id, mem)
                          for gpu_id, mem in gpu_info if mem >= min_memory_required]
        if available_gpus and torch.cuda.is_available():
            # Sort by the most free memory available
            selected_gpu = max(available_gpus, key=lambda x: x[1])[0]
            logger.info(
                f"Selected GPU: cuda:{selected_gpu} with {dict(gpu_info)[selected_gpu]:.2f} GB free")
            return f'cuda:{selected_gpu}'
    logger.warning("No suitable GPUs available. Falling back to CPU.")
    return 'cpu'


def main(argv):
    """
    Runs the interactive loop for the Mental Health Chatbot.

    The chatbot processes user input and categorizes it into informational or emotional types.
    """
    device = get_best_available_device(min_memory_required=4.0)

    # Ensure that model loads properly on the selected device
    try:
        chatbot = MentalHealthChatbot(
            faq_data_path=FLAGS.faq_data_path,
            conversations_data_path=FLAGS.conversations_data_path,
            test_size=FLAGS.test_size,
            random_state=FLAGS.random_state,
            sentence_encoder=FLAGS.sentence_encoder,
            device=torch.device(device)
        )
    except torch.OutOfMemoryError:
        logger.error(
            "\n!!\nOut of memory on all selected devices. \n!!\nRetrying on CPU.")
        device = torch.device('cpu')
        chatbot = MentalHealthChatbot(
            faq_data_path=FLAGS.faq_data_path,
            conversations_data_path=FLAGS.conversations_data_path,
            test_size=FLAGS.test_size,
            random_state=FLAGS.random_state,
            sentence_encoder=FLAGS.sentence_encoder,
            device=device
        )

    # Load data and train models
    chatbot.load_data()
    chatbot.preprocess_data()
    chatbot.train_logistic_classifier()
    chatbot.build_knn_classifier()

    # Start the interactive loop
    summarization_engine = SummarizationEngine()
    session_log = []

    print("\n\nWelcome to the Mental Health Chatbot. \n\nType 'exit', 'quit', 'q', 'x', or press 'Ctrl+C' to quit.")
    print("Type 'summarize' at any time to get a summary of the conversation so far.\n")
    try:
        while True:
            user_input = input("You: ")
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q', 'x', 'e']:
                print("Chatbot: Take care!")
                # Provide summary at the end of the session
                if session_log:
                    full_text = ' '.join(session_log)
                    summary = summarization_engine.summarize_text(full_text)
                    print("\nSession Summary:")
                    print(summary)
                break
            elif user_input.lower() == 'summarize':
                # Provide on-demand summary
                if session_log:
                    full_text = ' '.join(session_log)
                    summary = summarization_engine.summarize_text(full_text)
                    print("\nSummary:")
                    print(summary)
                else:
                    print("\nNo conversation history to summarize yet.")
                continue
            elif (len(user_input) == 0):
                continue

            # Get response from the chatbot
            response = chatbot.respond_to_query(user_input)
            print(f"Chatbot: {response}")
            # Append the interaction to the session log
            session_log.append(f"You: {user_input}")
            session_log.append(f"Chatbot: {response}")
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        print("\n\nChatbot: Exiting. \nTake care!\n\n")
        # Provide summary at the end of the session
        if session_log:
            full_text = ' '.join(session_log)
            summary = summarization_engine.summarize_text(full_text)
            print("\nSession Summary:")
            print(summary)


if __name__ == '__main__':
    app.run(main)
