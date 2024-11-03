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
import random
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support


class MentalHealthChatbot:
    """A chatbot for mental health support."""

    def __init__(self, faq_data_path, empathetic_data_path, conversations_data_path, test_size, random_state):
        """
        Initializes the MentalHealthChatbot with data paths and configuration parameters.

        Args:
            faq_data_path (str): Path to the FAQ dataset (CSV file).
            empathetic_data_path (str): Path to the empathetic dialogues dataset (directory).
            conversations_data_path (str): Path to the 
                mental health counseling conversations dataset (CSV file).
            test_size (float): Fraction of the data to reserve for testing (0 < test_size < 1).
            random_state (int): Seed for random number generation to ensure reproducibility.

        Attributes:
            model (SentenceTransformer): Pretrained Sentence-BERT model for encoding sentences.
            logistic_classifier (LogisticRegression): Classifier for 
                determining the type of query (informational vs. emotional).
            knn_classifier (NearestNeighbors): KNN classifier for 
                retrieving informational responses (set later).
            emotion_classifier (LinearSVC): Classifier for 
                predicting emotions in emotional queries (set later).
            df_A (DataFrame): DataFrame containing FAQ questions and answers.
            df_B (DataFrame): DataFrame containing empathetic dialogues context and emotion.
            df_C (DataFrame): DataFrame containing mental health counseling conversations.
            X (np.array): Encoded representations of both FAQ and empathetic dialogue data.
            y (list): Labels corresponding to the FAQ and empathetic dialogue data.
            conversations_per_emotion (dict): Conversation graphs organized by emotion.
            current_conversation (list): Current conversation sequence.
            current_conversation_index (int): Index of the next 
                response in the current conversation.
            current_emotion (str): Emotion associated with the current conversation.
        """
        self.faq_data_path = faq_data_path
        self.empathetic_data_path = empathetic_data_path
        self.conversations_data_path = conversations_data_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.logistic_classifier = LogisticRegression()
        # Placeholder for additional classifiers
        self.knn_classifier = None
        self.emotion_classifier = None
        self.current_conversation = None
        self.current_conversation_index = 0
        self.current_emotion = None

    def load_data(self):
        """
        Loads the FAQ, empathetic dialogues, and 
            mental health counseling conversations datasets into DataFrames.

        Returns:
            None
        """
        self.df_faq = pd.read_csv(
            self.faq_data_path)  # Updated from df_A to df_faq
        print("FAQ Data Columns:", self.df_faq.columns)

        splits = {'train': 'train.csv',
                  'validation': 'valid.csv', 'test': 'test.csv'}
        # Updated from df_B to df_empathetic_dialogues
        self.df_empathetic_dialogues = pd.read_csv(
            self.empathetic_data_path + splits['train'])
        print("Empathetic Dialogues Columns:",
              self.df_empathetic_dialogues.columns)

        # Updated from df_C to df_counseling_conversations
        self.df_counseling_conversations = pd.read_csv(
            self.conversations_data_path)
        print("Mental Health Counseling Conversations Columns:",
              self.df_counseling_conversations.columns)

    def preprocess_data(self):
        """
        Preprocesses and encodes the FAQ questions and 
            empathetic dialogue contexts using Sentence-BERT.

        Also processes the mental health counseling conversations 
            to organize them per emotion.

        Returns:
            None
        """
        # Preprocess FAQ and empathetic dialogue data
        # Updated from class_A_questions to faq_questions
        faq_questions = self.df_faq['Questions']
        # Updated from class_A_labels to faq_labels
        faq_labels = [0] * len(faq_questions)

        # Updated from class_B_contexts to empathetic_contexts
        empathetic_contexts = self.df_empathetic_dialogues['context']
        # Updated from class_B_labels to empathetic_labels
        empathetic_labels = [1] * len(empathetic_contexts)

        all_data = pd.concat([faq_questions, empathetic_contexts],
                             ignore_index=True).to_numpy()
        self.X = self.model.encode(all_data)
        self.y = faq_labels + empathetic_labels

        # Process mental health counseling conversations
        self.conversations_per_emotion = {}

        for _, row in self.df_counseling_conversations.iterrows():
            emotion = row['emotion']
            # Assuming 'conversation' column exists
            conversation = row['conversation']
            conversation_turns = conversation.split('__eot__')
            conversation_turns = [turn.strip()
                                  for turn in conversation_turns if turn.strip()]

            if emotion not in self.conversations_per_emotion:
                self.conversations_per_emotion[emotion] = []

            self.conversations_per_emotion[emotion].append(conversation_turns)

    def train_logistic_classifier(self):
        """
        Trains a logistic regression classifier to categorize user 
            queries as either informational or emotional.

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

    def train_emotion_classifier(self):
        """
        Trains a Linear Support Vector Classifier (LinearSVC) to 
            predict emotions from emotional support queries.

        The classifier is trained on the encoded contexts from the 
            empathetic dialogues dataset and their 
        associated emotions, as well as the conversations from the 
            mental health counseling conversations dataset.

        Returns:
            None
        """
        # Extract emotions and contexts from the empathetic dialogues
        # Updated from df_B to df_empathetic_dialogues
        emotions_empathetic = self.df_empathetic_dialogues['emotion']
        # Updated from df_B to df_empathetic_dialogues
        contexts_empathetic = self.df_empathetic_dialogues['context']

        # Extract emotions and the initial client utterance from the counseling conversations
        # Updated from df_C to df_counseling_conversations
        emotions_counseling = self.df_counseling_conversations['emotion']
        contexts_counseling = self.df_counseling_conversations['conversation'].apply(
            lambda x: x.split('__eot__')[0])  # Assuming the first turn is the client's message

        # Combine the data from both sources
        emotions = pd.concat(
            [emotions_empathetic, emotions_counseling], ignore_index=True)
        contexts = pd.concat(
            [contexts_empathetic, contexts_counseling], ignore_index=True)

        # Encode contexts
        X_emotion = self.model.encode(contexts.tolist())
        y_emotion = emotions.tolist()

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_emotion, y_emotion, test_size=self.test_size, random_state=self.random_state
        )

        # Train the emotion classifier
        self.emotion_classifier = LinearSVC()
        self.emotion_classifier.fit(X_train, y_train)
        y_pred = self.emotion_classifier.predict(X_test)

        # Print the classification report
        print('\nEmotion Classification Report:\n',
              classification_report(y_test, y_pred))

    def build_knn_classifier(self):
        """
        Builds a K-Nearest Neighbors (KNN) 
            classifier for retrieving informational responses.

        Returns:
            None
        """
        faq_embeddings = self.model.encode(
            self.df_faq['Questions'].tolist())  # Updated from df_A to df_faq
        self.knn_classifier = NearestNeighbors(
            n_neighbors=1, algorithm='ball_tree').fit(faq_embeddings)

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
        # Updated from df_A to df_faq
        response = self.df_faq['Answers'].iloc[index]
        return response

    def get_emotional_response(self, query):
        """
        Generates an emotional support response based on the user's query and conversation state.

        Args:
            query (str): The user's emotional query.

        Returns:
            str: A response from the conversation graph based on the user's emotional state.
        """
        # If no current conversation, classify emotion and start new conversation
        if self.current_conversation is None:
            query_embedding = self.model.encode([query])
            predicted_emotion = self.emotion_classifier.predict(query_embedding)[
                0]

            # Select a conversation sequence for the predicted emotion
            if predicted_emotion in self.conversations_per_emotion:
                conversations = self.conversations_per_emotion[predicted_emotion]
                # Randomly select a conversation
                self.current_conversation = random.choice(conversations)
                self.current_conversation_index = 0
                self.current_emotion = predicted_emotion
            else:
                # If no conversation available for the emotion
                return "I'm here to listen. Please tell me more about how you're feeling."

        # Proceed to next turn in the conversation
        if self.current_conversation_index < len(self.current_conversation):
            response = self.current_conversation[self.current_conversation_index]
            self.current_conversation_index += 1
        else:
            # End of conversation
            response = "I hope our conversation has been helpful."
            self.current_conversation = None
            self.current_conversation_index = 0
            self.current_emotion = None

        return response

    def respond_to_query(self, query):
        """
        Responds to a user query by first categorizing it and 
            then generating an appropriate response.

        Args:
            query (str): The user's input query.

        Returns:
            str: The chatbot's response (informational or emotional) 
                based on the query type.
        """
        # If in the middle of an emotional conversation, continue it
        if self.current_conversation is not None:
            response = self.get_emotional_response(query)
            return response

        # Else, classify the query
        query_embedding = self.model.encode([query])
        category = self.logistic_classifier.predict(query_embedding)[0]

        if category == 0:
            response = self.get_informational_response(query)
        else:
            response = self.get_emotional_response(query)

        return response
