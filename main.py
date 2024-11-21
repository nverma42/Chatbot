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
import logging
import subprocess
from typing import List, Tuple, Optional
import torch
from absl import app, flags
from mental_health_chatbot import MentalHealthChatbot
from summarization_engine import EnhancedSummarizationEngine
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
# Add new flags for summarization engine
flags.DEFINE_string(
    'sentence_transformer_model',
    'paraphrase-MiniLM-L6-v2',
    'Model name for sentence transformer'
)
flags.DEFINE_float(
    'summarization_lambda',
    0.7,
    'Lambda parameter for MMR summarization'
)
flags.DEFINE_integer(
    'summary_sentences',
    3,
    'Number of sentences in summary'
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('networkx').setLevel(logging.WARNING)
logging.getLogger('gensim').setLevel(logging.WARNING)


def get_gpu_memory_info() -> List[Tuple[int, float]]:
    """
    Uses nvidia-smi to get accurate memory usage for each GPU.

    Returns:
        List[Tuple[int, float]]: List of tuples containing (GPU ID, free memory in GB).
        Returns empty list if nvidia-smi is not available or fails.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        memory_info = result.stdout.strip().split('\n')
        # Convert to GB
        return [(i, float(mem)/1024) for i, mem in enumerate(memory_info)]
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.debug(f"nvidia-smi command failed: {str(e)}")
        return []


def check_cuda_availability() -> bool:
    """
    Checks if CUDA is available and properly configured.

    Returns:
        bool: True if CUDA is available and working, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Try to create a small tensor on GPU
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        return True
    except RuntimeError:
        return False


def get_best_available_device(min_memory_required: float = 4.0) -> Tuple[str, Optional[List[int]]]:
    """
    Determines the best available device(s) for running the model.

    Args:
        min_memory_required (float): Minimum required free memory in GB per GPU

    Returns:
        Tuple[str, Optional[List[int]]]: (device string, list of GPU IDs if multiple available)
    """
    # First check if CUDA is properly available
    if not check_cuda_availability():
        logger.info("CUDA is not available, using CPU")
        return 'cpu', None

    # Get GPU memory information
    gpu_info = get_gpu_memory_info()
    if not gpu_info:
        logger.info("Could not get GPU information, falling back to CPU")
        return 'cpu', None

    # Filter GPUs with enough memory
    available_gpus = [(gpu_id, mem)
                      for gpu_id, mem in gpu_info if mem >= min_memory_required]

    if not available_gpus:
        logger.warning(
            f"No GPUs with sufficient memory ({min_memory_required}GB required), using CPU")
        return 'cpu', None

    # Sort by available memory
    available_gpus.sort(key=lambda x: x[1], reverse=True)

    if len(available_gpus) == 1:
        gpu_id = available_gpus[0][0]
        logger.info(
            f"Selected single GPU: cuda:{gpu_id} with {available_gpus[0][1]:.2f}GB free")
        return f'cuda:{gpu_id}', None
    else:
        gpu_ids = [gpu_id for gpu_id, _ in available_gpus]
        logger.info(
            f"Selected multiple GPUs: {gpu_ids} with {[mem for _, mem in available_gpus]} GB free")
        return 'cuda', gpu_ids


def initialize_chatbot(device: str, gpu_ids: Optional[List[int]]) -> MentalHealthChatbot:
    """
    Initializes the chatbot with proper error handling and fallback mechanisms.

    Args:
        device (str): The primary device to use ('cuda:X' or 'cpu')
        gpu_ids (Optional[List[int]]): List of GPU IDs for multi-GPU setup

    Returns:
        MentalHealthChatbot: Initialized chatbot instance
    """
    try:
        return MentalHealthChatbot(
            faq_data_path=FLAGS.faq_data_path,
            conversations_data_path=FLAGS.conversations_data_path,
            test_size=FLAGS.test_size,
            random_state=FLAGS.random_state,
            sentence_encoder=FLAGS.sentence_encoder,
            device=torch.device(device),
            gpu_ids=gpu_ids
        )
    except torch.cuda.OutOfMemoryError:
        logger.warning("Out of memory on GPU, falling back to CPU")
        return MentalHealthChatbot(
            faq_data_path=FLAGS.faq_data_path,
            conversations_data_path=FLAGS.conversations_data_path,
            test_size=FLAGS.test_size,
            random_state=FLAGS.random_state,
            sentence_encoder=FLAGS.sentence_encoder,
            device=torch.device('cpu'),
            gpu_ids=None
        )
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise


def main(argv):
    """
    Main function for the Mental Health Chatbot with enhanced device handling and robust error management.
    Handles both interactive chat sessions and model initialization with proper device allocation.
    """
    # Set up logging with proper formatting
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # Suppress verbose logging from dependencies
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('networkx').setLevel(logging.WARNING)
    logging.getLogger('gensim').setLevel(logging.WARNING)

    # Get the best available device
    device, gpu_ids = get_best_available_device(min_memory_required=4.0)
    logger.info(f"Selected device: {device}" +
                (f" with GPUs: {gpu_ids}" if gpu_ids else ""))

    try:
        # Initialize chatbot with proper error handling
        logger.info("Initializing chatbot...")
        chatbot = initialize_chatbot(device, gpu_ids)

        # Initialize summarization engine with same device configuration
        logger.info("Initializing summarization engine...")
        summarization_engine = EnhancedSummarizationEngine(
            model_name=FLAGS.sentence_transformer_model,
            lambda_param=FLAGS.summarization_lambda,
            top_k=FLAGS.summary_sentences,
            device=device
        )

        # Load and prepare data
        logger.info("Loading data...")
        chatbot.load_data()

        logger.info("Preprocessing data...")
        chatbot.preprocess_data()

        logger.info("Training logistic classifier...")
        chatbot.train_logistic_classifier()

        logger.info("Building KNN classifier...")
        chatbot.build_knn_classifier()

        # Initialize session storage
        session_log = []

        # Print welcome message
        print("\nWelcome to the Mental Health Chatbot!")
        print("----------------------------------------")
        print("Commands:")
        print("- Type 'exit', 'quit', 'q', 'x', or 'e' to end the session")
        print("- Type 'summarize' to get a summary of the conversation")
        print("- Press Ctrl+C to exit at any time")
        print("----------------------------------------\n")

        # Main interaction loop
        while True:
            try:
                user_input = input("You: ").strip()

                # Handle exit commands
                if user_input.lower() in ['exit', 'quit', 'q', 'x', 'e']:
                    print("\nChatbot: Thank you for talking with me. Take care!")

                    # Provide session summary if there was a conversation
                    if session_log:
                        print("\nHere's a summary of our conversation:")
                        full_text = ' '.join(session_log)
                        summary_result = summarization_engine.summarize(
                            full_text,
                            query="What were the main points discussed in this conversation?"
                        )
                        print(summary_result['summary'])
                    break

                # Handle summarize command
                elif user_input.lower() == 'summarize':
                    if session_log:
                        print("\nConversation Summary:")
                        full_text = ' '.join(session_log)
                        summary_result = summarization_engine.summarize(
                            full_text,
                            query="What were the key points in our recent discussion?"
                        )
                        print(summary_result['summary'])

                        # Print debug info if in debug mode
                        if logger.getEffectiveLevel() == logging.DEBUG:
                            print("\nDebug Information:")
                            print(
                                f"Total Sentences: {summary_result['original_sentences']}")
                            print(
                                f"Selected Sentences: {summary_result['selected_sentences']}")
                            print(
                                f"Similarity Scores: {summary_result['similarity_scores']}")
                    else:
                        print("\nNo conversation history to summarize yet.")
                    continue

                # Handle empty input
                if not user_input:
                    print(
                        "Chatbot: I didn't catch that. Could you please say something?")
                    continue

                # Get response from chatbot
                try:
                    response = chatbot.respond_to_query(user_input)
                    print(f"Chatbot: {response}")

                    # Log the interaction
                    session_log.append(f"You: {user_input}")
                    session_log.append(f"Chatbot: {response}")

                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    print(
                        "Chatbot: I apologize, but I'm having trouble generating a response. Could you try rephrasing your question?")

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n\nChatbot: Conversation ended. Take care!")

                # Provide summary if there was a conversation
                if session_log:
                    print("\nHere's a summary of our conversation:")
                    full_text = ' '.join(session_log)
                    summary_result = summarization_engine.summarize(
                        full_text,
                        query="What were the main points discussed in this conversation?"
                    )
                    print(summary_result['summary'])
                break

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        print(
            "\nI apologize, but I've encountered a technical issue and need to shut down.")
        print("Please try running the program again.")
        raise

    finally:
        # Cleanup if needed
        logger.info("Shutting down chatbot system...")
        torch.cuda.empty_cache()  # Clear GPU memory if it was used


if __name__ == '__main__':
    app.run(main)
