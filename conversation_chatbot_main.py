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
import subprocess
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        if available_gpus:
            # Sort by the most free memory available
            selected_gpu = max(available_gpus, key=lambda x: x[1])[0]
            logger.info(
                f"Selected GPU: cuda:{selected_gpu} with {dict(gpu_info)[selected_gpu]:.2f} GB free")
            return f'cuda:{selected_gpu}'
    logger.warning("No suitable GPUs available. Falling back to CPU.")
    return 'cpu'


def get_available_devices(min_memory_required=4 * 1024**3):
    """
    Checks all available CUDA devices and returns a list of device IDs that have
    at least 'min_memory_required' bytes of free memory.

    Args:
        min_memory_required (int): Minimum required free memory in bytes.

    Returns:
        List[int]: List of available CUDA device IDs.
    """
    available_devices = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA is available. Number of GPUs: {num_gpus}")

        for idx in range(num_gpus):
            # Get total and free memory on the GPU
            _ = torch.cuda.memory_stats(idx)
            reserved = torch.cuda.memory_reserved(idx)
            allocated = torch.cuda.memory_allocated(idx)
            free_memory = reserved - allocated
            total_memory = torch.cuda.get_device_properties(idx).total_memory

            logger.info(
                f"GPU {idx}: Total Memory: {total_memory / (1024**3):.2f} GB, Free Memory: {free_memory / (1024**3):.2f} GB")

            if free_memory >= min_memory_required:
                available_devices.append(idx)
    else:
        logger.info("CUDA is not available. Using CPU.")

    return available_devices


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
    print("\n\nWelcome to the Mental Health Chatbot. \n\nType 'exit', 'quit', 'q', 'x', or press 'Ctrl+C' to quit.")
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
        # Graceful exit on Ctrl+C
        print("\n\nChatbot: Exiting. \nTake care!\n\n")


if __name__ == '__main__':
    app.run(main)
