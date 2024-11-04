"""
Approach:
This chatbot uses a Sentence Transformer model to encode each
    user query and classify it into two categories:
A - User seeking information related to mental health (informational query)
B - User seeking emotional support (emotional query)

For category A:
- A K-Nearest Neighbors (KNN) classifier is built to match the query
    with an appropriate informational response from the FAQ dataset.

For category B:
- A Linear Support Vector Classifier (LinearSVC) is trained to
    predict the user's emotional state (e.g., sadness, anger, etc.).
- Instead of canned responses, the chatbot uses a conversation graph
    derived from real counseling sessions to provide meaningful, multi-turn interactions.
- These conversation graphs help sustain an ongoing conversation,
    providing a more personalized and empathetic experience.

Report precision, recall, and F1 score for both classifiers
    (logistic classifier and emotion classifier).

Important Changes:
- The dependency on canned responses has been removed.
    Emotional responses are now generated based on a graph of
        real counseling sessions, enabling history-aware conversations.
- The chatbot maintains conversation state,
    allowing it to handle multi-turn conversations with users.

Future Enhancements:
1. Dynamic Response Generation:
    - Consider integrating advanced NLP models like
        GPT-3 or fine-tuned transformers to generate dynamic responses,
        further improving the chatbot's conversational capabilities.
2. Localization:
    - Support multiple languages by adding a language column to the FAQ
        and counseling conversation datasets,
        making the chatbot accessible to a wider audience.
3. Emotional Intensity:
    - Include an additional column to handle the intensity of emotions for
        more nuanced responses, improving the chatbot's ability to
        tailor support to users' emotional states.
4. Professional Help:
    - Enhance the chatbot's ethical considerations by ensuring it can recognize
    high-risk situations and direct users to professional help when necessary.
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from Emotional_Response import EmotionalResponse


class MentalHealthChatbot:
    """A chatbot for mental health support."""

    def __init__(self, faq_data_path, conversations_data_path, test_size=0.3, random_state=42, device='cpu'):
        """
        Initializes the MentalHealthChatbot with data paths and configuration parameters.

        Args:
            faq_data_path (str): Path to the FAQ dataset (CSV file).
            conversations_data_path (str): Path to the mental health counseling conversations dataset (CSV file).
            test_size (float): Fraction of the data to reserve for testing (0 < test_size < 1).
            random_state (int): Seed for random number generation to ensure reproducibility.

        Attributes:
            model (SentenceTransformer): Pretrained Sentence-BERT model for encoding sentences.
            logistic_classifier (LogisticRegression): Classifier for determining the type of query (informational vs. emotional).
            knn_classifier (NearestNeighbors): KNN classifier for retrieving informational responses (set later).
            df_faq (DataFrame): DataFrame containing FAQ questions and answers.
            df_counseling_conversations (DataFrame): DataFrame containing mental health counseling conversations.
            X (np.array): Encoded representations of both FAQ and emotional data.
            y (list): Labels corresponding to the FAQ and emotional data.
            emotional_response_handler (EmotionalResponse): Instance of EmotionalResponse class to handle emotional queries.
        """
        self.faq_data_path = faq_data_path
        self.conversations_data_path = conversations_data_path
        self.test_size = test_size
        self.random_state = random_state
        self.device = device  # Store the device

        # Initialize models and data placeholders
        self.encoder = None
        self.logistic_classifier = None
        self.knn_classifier = None
        self.emotional_response = None

    def load_data(self):
        """
        Loads the FAQ and mental health counseling conversations datasets into DataFrames.

        Returns:
            None
        """
        self.faq_df = pd.read_csv(self.faq_data_path)
        self.conversations_df = pd.read_json(
            self.conversations_data_path, lines=True)

    def preprocess_data(self):
        """
        Preprocesses and encodes the FAQ questions and emotional queries using Sentence-BERT.

        Returns:
            None
        """
        # Combine datasets and create labels
        faq_queries = self.faq_df['Question'].tolist()
        faq_labels = [0] * len(faq_queries)  # 0 for informational

        conv_contexts = self.conversations_df['Context'].tolist()
        conv_labels = [1] * len(conv_contexts)  # 1 for emotional

        all_queries = faq_queries + conv_contexts
        all_labels = faq_labels + conv_labels

        # Load the encoder model onto the specified device
        self.encoder = SentenceTransformer(
            'paraphrase-MiniLM-L6-v2', device=self.device)

        # Encode the queries
        self.encoded_queries = self.encoder.encode(
            all_queries, convert_to_tensor=True, device=self.device)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.encoded_queries.cpu().numpy(), all_labels,
            test_size=self.test_size, random_state=self.random_state
        )

    def train_logistic_classifier(self):
        """
        Trains a logistic regression classifier to categorize user 
            queries as either informational or emotional.

        Returns:
            None
        """
        self.logistic_classifier = LogisticRegression(
            random_state=self.random_state)
        self.logistic_classifier.fit(self.X_train, self.y_train)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     self.X, self.y, test_size=self.test_size, random_state=self.random_state
        # )

        # self.logistic_classifier.fit(X_train, y_train)
        # y_pred = self.logistic_classifier.predict(X_test)

        # print('\nClassification Report:\n',
        #       classification_report(y_test, y_pred))
        # precision, recall, f1, _ = precision_recall_fscore_support(
        #     y_test, y_pred, average='weighted')
        # print(
        #     f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

    def build_knn_classifier(self):
        """
        Builds a K-Nearest Neighbors (KNN) classifier for retrieving informational responses.

        Returns:
            None
        """
        faq_queries = self.faq_df['Question'].tolist()
        self.faq_answers = self.faq_df['Answer'].tolist()

        # Encode FAQ queries
        self.encoded_faq_queries = self.encoder.encode(
            faq_queries, convert_to_tensor=True, device=self.device)

        # Build KNN model
        self.knn_classifier = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.knn_classifier.fit(self.encoded_faq_queries.cpu().numpy())

        # Initialize EmotionalResponse with the same device
        self.emotional_response = EmotionalResponse(device=self.device)

    def get_informational_response(self, query):
        """
        Generates a response for informational queries.

        Args:
            query (str): The user's query to find an informational response.

        Returns:
            str: The informational response based on the closest match in the FAQ data.
        """
        query_embedding = self.model.encode([query])
        _, indices = self.knn_classifier.kneighbors(query_embedding)
        index = indices[0][0]
        response = self.df_faq['Answers'].iloc[index]
        return response

    def respond_to_query(self, query):
        """
        Responds to the user's query based on its classification.

        Args:
            query (str): The user's input query.

        Returns:
            str: The chatbot's response.
        """
        # Encode the query
        query_embedding = self.encoder.encode(
            [query], convert_to_tensor=True, device=self.device).cpu().numpy()

        # Classify the query
        prediction = self.logistic_classifier.predict(query_embedding)

        if prediction[0] == 0:
            # Informational query
            _, indices = self.knn_classifier.kneighbors(
                query_embedding)
            answer = self.faq_answers[indices[0][0]]
            return answer
        else:
            # Emotional query
            response = self.emotional_response.get_response(query)
            return response
