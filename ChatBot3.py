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

# Read empathetic dialogues context
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

df_A = pd.read_csv('./data/Mental_Health_FAQ.csv')
print(df_A.columns)

splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
df_B = pd.read_csv("hf://datasets/bdotloh/empathetic-dialogues-contexts/" + splits["train"])
print(df_B.columns)

# Load pretrained Sentence-BERT model.
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create vector representation for each question/situation.
class_A_questions = df_A['Questions']
class_A_labels = [0]*len(class_A_questions)

class_B_contexts = df_B['situation']
class_B_labels = [1]*len(class_B_contexts)

all_data = pd.concat([class_A_questions, class_B_contexts], ignore_index=True).to_numpy()
X = model.encode(all_data)
print(X.shape)

y = class_A_labels + class_B_labels

# Split the data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a binary classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Use the test dataset to compute metrics
y_pred = classifier.predict(X_test)

print('\nClassification Report:\n ' + classification_report(y_test, y_pred))

# Build KNN classifier for class A questions.

# Build emotion classifier for class B questions.

# Generate responses



