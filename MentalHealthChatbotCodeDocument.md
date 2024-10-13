# Mental Health Chatbot Code Documentation

## Mission Vision Goal

The Mental Health Chatbot is designed to provide users with emotional support and factual mental health information through conversational interactions. When a user inputs text into the chatbot, the process is as follows:

Query Classification: The input is first processed by the chatbot to determine whether the user is seeking information or emotional support. This classification is done using a logistic regression model that has been trained to recognize different types of queries.

**Response Generation**:

Informational Queries: If the query is classified as informational, the chatbot uses a K-Nearest Neighbors (KNN) model to search through a predefined FAQ dataset. The model finds the most relevant question and returns the corresponding factual answer.
Emotional Queries: If the query is classified as emotional, the chatbot processes the input to detect the user's emotional state using a Linear Support Vector Classifier (LinearSVC). The classifier predicts an emotion (e.g., sadness, anger, joy), and the chatbot responds with an empathetic message chosen from a predefined set of canned responses.
Text Encoding: Throughout these steps, the chatbot uses Sentence-BERT, a powerful natural language processing model, to convert the user's input into a numerical vector representation. This allows for efficient processing by machine learning models.

**Personalization and Adaptability**: Although responses are currently based on predefined templates, the chatbot can be expanded in future iterations to include more dynamic, personalized responses using generative models like GPT.

This document outlines how each part of the chatbot is implemented, including detailed explanations of its components, the machine learning models used, and how user input flows through the system to generate responses.

## Table of Contents

- [Class and Method Descriptions](#purpose)
  - [MentalHealthChatbot Class](#mentalhealthchatbot-class)
    - [\_\_init\_\_](#attributes)
    - [load\_data](#load_data)
    - [load\_canned\_responses](#load_canned_responses)
    - [preprocess\_data](#preprocess_data)
    - [train\_logistic\_classifier](#train_logistic_classifier)
    - [build\_knn\_classifier](#build_knn_classifier)
    - [train\_emotion\_classifier](#train_emotion_classifier)
    - [get\_informational\_response](#get_informational_response)
    - [get\_emotional\_response](#get_emotional_response)
    - [respond\_to\_query](#respond_to_query)
    - [run](#run)
- [How the Code Implements the Methodology](#how-the-code-implements-the-methodology)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Appendix](#appendix)
  - [Data File Formats](#data-file-formats)

## Purpose

Initializes the chatbot with data paths and configuration parameters.

### Arguments

- `faq_data_path`: Path to the FAQ dataset CSV file.
- `empathetic_data_path`: Path to the empathetic dialogues dataset directory.
- `canned_responses_path`: Path to the canned responses CSV file.
- `test_size`: Fraction of the data to reserve for testing.
- `random_state`: Seed for random number generation (ensures reproducibility).

### Attributes

- Initializes machine learning models.
- Loads canned responses into a dictionary.

---

## load_data

### Purpose

Loads the FAQ and empathetic dialogues datasets into Pandas DataFrames.

### Process

- Reads the FAQ CSV file containing questions and answers.
- Reads the empathetic dialogues CSV files (train, validation, test) containing conversational contexts and associated emotions.

---

## load_canned_responses

### Purpose

Loads canned emotional responses from a CSV file into a dictionary.

### Process

- Reads the `canned_responses.csv` file.
- Converts it into a dictionary with emotions as keys and responses as values.

### Error Handling

Includes a try-except block to handle exceptions during file loading.

---

## preprocess_data

### Purpose

Preprocesses and encodes text data for model training.

### Process

- Concatenates FAQ questions and empathetic dialogue contexts.
- Encodes text data into vector representations using Sentence-BERT.
- Creates labels: `0` for informational queries, `1` for emotional queries.

---

## train_logistic_classifier

### Purpose

Trains a logistic regression classifier to categorize queries as informational or emotional.

### Process

- Splits data into training and testing sets.
- Trains the classifier on encoded data and labels.
- Evaluates the classifier using precision, recall, and F1 score.

---

## build_knn_classifier

### Purpose

Builds a KNN model for retrieving informational responses.

### Process

- Encodes FAQ questions using Sentence-BERT.
- Fits the KNN model on the encoded questions.

---

## train_emotion_classifier

### Purpose

Trains a LinearSVC model to predict emotions from emotional queries.

### Process

- Encodes empathetic dialogue contexts.
- Splits data into training and testing sets.
- Trains the LinearSVC model on encoded data and emotion labels.
- Evaluates the classifier using precision, recall, and F1 score.

---

## get_informational_response

### Purpose

Generates responses for informational queries.

### Process

- Encodes the user's query.
- Uses the KNN model to find the closest matching question.
- Retrieves the corresponding answer from the FAQ dataset.

### Notes

The `distances` variable (not used in the current code) contains the distance metrics from the KNN model, which could be useful for future enhancements like confidence scoring.

---

## get_emotional_response

### Purpose

Generates empathetic responses based on the predicted emotion.

### Process

- Encodes the user's query.
- Predicts the emotion using the trained emotion classifier.
- Retrieves the corresponding canned response from the dictionary.
- Provides a default response if the emotion is not found.

---

## respond_to_query

### Purpose

Determines the type of query and generates an appropriate response.

### Process

- Encodes the user's query.
- Uses the logistic classifier to categorize the query.
- Routes the query to either `get_informational_response` or `get_emotional_response` based on the category.

---

## run

### Purpose

Executes the main flow of the chatbot and handles user interaction.

### Process

- Calls methods to load data, preprocess data, and train models.
- Initiates an interactive loop where the user can input queries.
- Handles exit conditions when the user types 'exit' or 'quit'.

---

# How the Code Implements the Methodology

The code closely follows the methodology outlined in the project proposal:

### Classification of User Queries

- **Methodology**: Classify queries as emotional or technical.
- **Implementation**: Uses a logistic regression classifier trained on encoded representations of FAQ questions (technical) and empathetic dialogues (emotional) to categorize queries.

### Handling Informational Queries

- **Methodology**: Use a KNN classifier to generate canned informational responses.
- **Implementation**: Encodes FAQ questions and uses a KNN model to find the closest match to the user's query, returning the associated answer.

### Handling Emotional Queries

- **Methodology**: Predict emotion using a classifier and generate canned responses.
- **Implementation**: Trains a LinearSVC model on emotional contexts to predict the user's emotional state and retrieves a corresponding canned response from a CSV file.

### Use of Pretrained Models

- **Methodology**: Leverage advanced NLP models for encoding and classification.
- **Implementation**: Uses Sentence-BERT (specifically the 'paraphrase-MiniLM-L6-v2' model) to encode sentences into vectors suitable for machine learning tasks.

### Evaluation Metrics

- **Methodology**: Report precision, recall, and F1 score for classifiers.
- **Implementation**: Uses Scikit-learn's evaluation metrics to assess model performance and prints classification reports during training.

### Canned Response Repository

- **Methodology**: Need a repository of canned responses for different emotions.
- **Implementation**: Loads canned responses from `canned_responses.csv` into a dictionary, allowing easy expansion and management of responses.

---

# Technologies Used

- **Python**: Main programming language for the project.
- **Sentence-BERT**: For encoding text into numerical vector representations.
- **Scikit-learn**: Provides machine learning algorithms and evaluation tools.
- **Pandas**: For data manipulation and handling CSV files.
- **absl-py**: For parsing command-line arguments and configuring the application.

---

# Future Improvements

### Dynamic Response Generation

Integrate language models like GPT-3 to generate more personalized and dynamic responses instead of relying solely on canned responses.

### Emotion Intensity Analysis

Enhance the emotion classifier to detect the intensity of emotions, allowing the chatbot to adjust its responses accordingly.

### Contextual Awareness

Implement a mechanism to maintain conversation context over multiple turns, providing a more coherent and engaging user experience.

### Multilingual Support

Expand the chatbot's capabilities to support multiple languages, making it accessible to a wider audience.

### Improved Error Handling

Enhance the robustness of the application by handling exceptions and edge cases more gracefully.

### User Feedback Loop

Incorporate a system for users to provide feedback on responses, enabling continuous improvement of the chatbot.

---

# Appendix

## Data File Formats

### Mental_Health_FAQ.csv

#### Columns

- `Questions`: The FAQ questions related to mental health topics.
- `Answers`: The corresponding answers to the questions.

### canned_responses.csv

#### Columns

- `emotion`: The emotion label (e.g., sadness, joy, anger).
- `response`: The canned response associated with the emotion.

### empathetic-dialogues-contexts/train.csv

#### Columns

- `context`: The conversational context or situation.
- `emotion`: The emotion associated with the context.
- Other columns may include dialogue history or metadata.
