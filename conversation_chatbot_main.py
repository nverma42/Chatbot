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
# Read empathetic dialogues context
from absl import app, flags
from mental_health_chatbot import MentalHealthChatbot

FLAGS = flags.FLAGS
'''
We do not neeed data/canned_response.csv.
The canned response file is the mental health FAQ file.

The empathetic data path represents the situation and emotion.
Our emotion classifier needs to produce one of the 32 emotions
encoded in this dataset. This classifier will be used to classify the emotion
of user's query i.e. angry, furious, sad etc.

Instead of producing canned responses to each emotion, we can create
a conversation graph based on previous counseling sessions.
A conversation graph will enable us to have history and thus
chatbot will be able to have meaninful conversations.
We will use this data: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
So we need to add df_C dataframe to get this data.

Using conversation graph is better than producing canned responses.

How will this process work?

This data has conversations with a counselor.
We will classify each of these conversations with an emotion because we have emotion classifier now.
Thus each emotion will have a separate graph. Let us say when someone is
sad, how conversation with a counseler proceeds is very predictable.
We will first classify user's emotion.
The ChatBot will then map user's query to a graph.
Then the conversation will proceed accordingly.
We do not have a lot of conversation data, so chatbot won't
be able to chat for long but for our project this should be fine.
'''
flags.DEFINE_string(
    'faq_data_path', './data/Mental_Health_FAQ.csv', 'Path to the FAQ dataset')
flags.DEFINE_string('empathetic_data_path', 'hf://datasets/bdotloh/empathetic-dialogues-contexts/',
                    'Path to the empathetic dialogues dataset')
flags.DEFINE_string('conversations_data_path', './data/mental_health_counseling_conversations.csv',
                    'Path to the mental health counseling conversations dataset')
flags.DEFINE_float('test_size', 0.3, 'Test set size as a fraction')
flags.DEFINE_integer('random_state', 42, 'Random seed for reproducibility')


def main(argv):
    """
    Runs the interactive loop.
    """
    chatbot = MentalHealthChatbot(
        faq_data_path=FLAGS.faq_data_path,
        empathetic_data_path=FLAGS.empathetic_data_path,
        conversations_data_path=FLAGS.conversations_data_path,
        test_size=FLAGS.test_size,
        random_state=FLAGS.random_state
    )

    # Load data and train models
    chatbot.load_data()
    chatbot.preprocess_data()
    chatbot.train_logistic_classifier()
    chatbot.build_knn_classifier()
    chatbot.train_emotion_classifier()

    # Interactive loop
    print("Welcome to the Mental Health Chatbot. Type 'exit', 'quit', 'q', 'x', or press 'Ctrl+C' to quit.")
    try:
        while True:
            user_input = input("You: ")
            # Expanded exit options
            if user_input.lower() in ['exit', 'quit', 'q', 'x', 'e']:
                print("Chatbot: Take care!")
                break
            response = chatbot.respond_to_query(user_input)
            print(f"Chatbot: {response}")
    except KeyboardInterrupt:
        print("\nChatbot: Exiting. Take care!")  # Graceful exit on Ctrl+C


if __name__ == '__main__':
    app.run(main)
