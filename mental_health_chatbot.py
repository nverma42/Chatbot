"""
# Approach:
# Use Sentence Transformer to encode each question
# Build a logistic classifier to separate the question in two
# categories:
# A - User seeking information related to mental health
# B - User seeking emotional support
# For question category A, build a KNN classifier to generate a canned informational response.
# For question category B, predict the emotion using LinearSVC classifier
# For each emotion, generate a canned response.
# Report precision, recall, F1 score for all the classifiers.
# Important: Need canned response repository

While the current scope focuses on canned responses, 
    consider integrating advanced NLP models like GPT-3 
    for more dynamic responses in future iterations. 
Also, think about the ethical implications and the necessity for the chatbot to 
    recognize when to direct users to professional help.
"""
# Read empathetic dialogues context
import pandas as pd
from absl import app, flags
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'faq_data_path', './data/Mental_Health_FAQ.csv', 'Path to the FAQ dataset')
flags.DEFINE_string('empathetic_data_path', 'hf://datasets/bdotloh/empathetic-dialogues-contexts/',
                    'Path to the empathetic dialogues dataset')
flags.DEFINE_float('test_size', 0.3, 'Test set size as a fraction')
flags.DEFINE_integer('random_state', 42, 'Random seed for reproducibility')


class MentalHealthChatbot:
    """A chatbot for mental health support."""

    def __init__(self, faq_data_path, empathetic_data_path, test_size, random_state):
        """
        Initializes the MentalHealthChatbot with data paths and configuration parameters.

        Args:
            faq_data_path (str): Path to the FAQ dataset (CSV file).
            empathetic_data_path (str): Path to the empathetic dialogues dataset (directory).
            test_size (float): Fraction of the data to reserve for testing (0 < test_size < 1).
            random_state (int): Seed for random number generation to ensure reproducibility.

        Attributes:
            model (SentenceTransformer): Pretrained Sentence-BERT model for encoding sentences.
            logistic_classifier (LogisticRegression): Classifier for determining the type of query (informational vs. emotional).
            knn_classifier (NearestNeighbors): KNN classifier for retrieving informational responses (set later).
            emotion_classifier (LinearSVC): Classifier for predicting emotions in emotional queries (set later).
            df_A (DataFrame): DataFrame containing FAQ questions and answers.
            df_B (DataFrame): DataFrame containing empathetic dialogues context and emotion.
            X (np.array): Encoded representations of both FAQ and empathetic dialogue data.
            y (list): Labels corresponding to the FAQ and empathetic dialogue data.
            canned_responses (dict): Predefined emotional responses keyed by predicted emotion.
        """
        self.faq_data_path = faq_data_path
        self.empathetic_data_path = empathetic_data_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.logistic_classifier = LogisticRegression()
        # Placeholder for additional classifiers
        self.knn_classifier = None
        self.emotion_classifier = None
        self.canned_responses = {
            'sadness': "I'm sorry to hear that you're feeling sad. Would you like to talk about it?",
            'anger': "It seems like you're feeling angry. I'm here to listen if you'd like to share more.",
            'joy': "That's wonderful to hear!",
            # Add more emotions and corresponding responses
        }

    def load_data(self):
        """
        Loads the FAQ and empathetic dialogues datasets into DataFrames.

        FAQ data is expected to be in CSV format containing columns for questions and answers.
        Empathetic dialogues data should contain columns for context and associated emotions.

        Returns:
            None
        """
        self.df_A = pd.read_csv(self.faq_data_path)
        print("FAQ Data Columns:", self.df_A.columns)

        splits = {'train': 'train.csv',
                  'validation': 'valid.csv', 'test': 'test.csv'}
        self.df_B = pd.read_csv(self.empathetic_data_path + splits['train'])
        print("Empathetic Dialogues Columns:", self.df_B.columns)

    def preprocess_data(self):
        """
        Preprocesses and encodes the FAQ questions and empathetic dialogue contexts using Sentence-BERT.

        The method concatenates FAQ questions and empathetic dialogue contexts into a single dataset, 
        then encodes the data into vector representations using the pretrained Sentence-BERT model.

        It also generates binary labels: 0 for informational queries and 1 for emotional queries.

        Returns:
            None
        """
        class_A_questions = self.df_A['Questions']
        class_A_labels = [0] * len(class_A_questions)

        class_B_contexts = self.df_B['situation']
        class_B_labels = [1] * len(class_B_contexts)

        all_data = pd.concat(
            [class_A_questions, class_B_contexts], ignore_index=True).to_numpy()
        self.X = self.model.encode(all_data)
        self.y = class_A_labels + class_B_labels

    def train_logistic_classifier(self):
        """
        Trains a logistic regression classifier to categorize user queries as either informational or emotional.

        The classifier is trained on the encoded representations of FAQ and empathetic dialogues data, 
        with binary labels indicating the type of query.

        After training, the method evaluates the classifier using the test dataset and prints precision, recall, 
        and F1 score for the model's performance.

        Returns:
            None
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        self.logistic_classifier.fit(X_train, y_train)
        y_pred = self.logistic_classifier.predict(X_test)

        print('\nClassification Report:\n',
              classification_report(y_test, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        print(
            f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

    def build_knn_classifier(self):
        """
        Builds a K-Nearest Neighbors (KNN) classifier for retrieving informational responses.

        The FAQ questions are encoded and stored in the KNN classifier. When a new query categorized as 
        informational is received, the KNN classifier finds the closest matching FAQ question to return a response.

        Returns:
            None
        """
        faq_embeddings = self.model.encode(self.df_A['Questions'].tolist())
        self.knn_classifier = NearestNeighbors(
            n_neighbors=1, algorithm='ball_tree').fit(faq_embeddings)

    def get_informational_response(self, query):
        """
        Generates a response for informational queries.

        Args:
            query (str): The user's query to find an informational response.

        Returns:
            str: The informational response based on the closest match in the FAQ data.

        Note:
            The `distances` variable, although not returned, 
            contains the distances between the query and the closest matching FAQ. 
            It can be used to evaluate how closely the query matches the informational responses. 
            May be useful for debugging or future enhancements.
        """
        query_embedding = self.model.encode([query])
        distances, indices = self.knn_classifier.kneighbors(query_embedding)
        index = indices[0][0]
        response = self.df_A['Answers'].iloc[index]
        return response

    def train_emotion_classifier(self):
        """
        Trains a Linear Support Vector Classifier (LinearSVC) to predict emotions from emotional support queries.

        The classifier is trained on the encoded contexts from the empathetic dialogues dataset and their 
        associated emotions. It is used to determine the emotional state of users seeking emotional support.

        After training, the method evaluates the classifier on the test data and prints a classification report 
        (precision, recall, F1 score).

        Returns:
            None
        """
        emotions = self.df_B['emotion']
        contexts = self.df_B['context']
        X_emotion = self.model.encode(contexts.tolist())
        y_emotion = emotions.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X_emotion, y_emotion, test_size=self.test_size, random_state=self.random_state
        )

        self.emotion_classifier = LinearSVC()
        self.emotion_classifier.fit(X_train, y_train)
        y_pred = self.emotion_classifier.predict(X_test)

        print('\nEmotion Classification Report:\n',
              classification_report(y_test, y_pred))

    def get_emotional_response(self, query):
        """
        Generates an emotional support response based on the user's query and predicted emotional state.

        The query is classified into one of several emotional states using the trained emotion classifier, and 
        a predefined (canned) response is returned based on the predicted emotion.

        Args:
            query (str): The user's emotional query.

        Returns:
            str: A canned response tailored to the predicted emotional state of the user.
        """
        query_embedding = self.model.encode([query])
        predicted_emotion = self.emotion_classifier.predict(query_embedding)[0]
        response = self.canned_responses.get(
            predicted_emotion, "I'm here to listen.")
        return response

    def respond_to_query(self, query):
        """
        Responds to a user query by first categorizing it and then generating an appropriate response.

        The method uses the logistic regression classifier to determine whether the query is informational or 
        emotional. It then routes the query to the appropriate method for generating a response: 
        `get_informational_response` or `get_emotional_response`.

        Args:
            query (str): The user's input query.

        Returns:
            str: The chatbot's response (informational or emotional) based on the query type.
        """
        query_embedding = self.model.encode([query])
        category = self.logistic_classifier.predict(query_embedding)[0]

        if category == 0:
            response = self.get_informational_response(query)
        else:
            response = self.get_emotional_response(query)

        return response

    def run(self):
        """
        Executes the main flow of the chatbot, including data loading, training, and user interaction.

        After loading and preprocessing the data, the method trains the classifiers and initiates an interactive 
        loop where users can input queries. The chatbot will continue to respond to user queries until the user 
        types 'exit' or 'quit'.

        Returns:
            None
        """
        self.load_data()
        self.preprocess_data()
        self.train_logistic_classifier()
        self.build_knn_classifier()
        self.train_emotion_classifier()

        # Interactive loop
        print("Welcome to the Mental Health Chatbot. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Chatbot: Take care!")
                break
            response = self.respond_to_query(user_input)
            print(f"Chatbot: {response}")


def main(argv):
    chatbot = MentalHealthChatbot(
        faq_data_path=FLAGS.faq_data_path,
        empathetic_data_path=FLAGS.empathetic_data_path,
        test_size=FLAGS.test_size,
        random_state=FLAGS.random_state
    )
    chatbot.run()


if __name__ == '__main__':
    app.run(main)
