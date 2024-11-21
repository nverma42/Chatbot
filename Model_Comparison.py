import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim import corpora
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import TfidfModel
import nltk
import pyLDAvis.gensim_models


def Compute_Coherence_BERTopic():
    df = pd.read_json(
        "hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)

    contexts = df['Context']

    # Create BERTopic model
    topic_model = BERTopic(top_n_words=25)
    topics, probs = topic_model.fit_transform(contexts)

    print(f'Number of topics = {len(set(topics))}')

    # Visualize the topics
    fig = topic_model.visualize_topics()
    fig.write_html("bertopic_visualization.html")

    # Prepare the documents for Gensim coherence calculation
    # Split your documents into tokens
    texts = [doc.split() for doc in contexts]

    # Create a Gensim Dictionary and Corpus
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Get topic words
    topic_words = topic_model.get_topics()
    topics = [[word for word, _ in topic_words[topic]]
              for topic in topic_words]

    coherence_model = CoherenceModel(
        topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')

    # Get the coherence score
    coherence_score = coherence_model.get_coherence()
    print("Coherence Score:", coherence_score)


def Compute_Coherence_LDA(from_n_topics, to_n_topics):
    # Load pretrained Sentence-BERT model.
    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    df = pd.read_json(
        "hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)
    contexts = df['Context'].tolist()

    nltk.download('stopwords')
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    custom_stop_words = stop_words

    # Preprocess questions
    processed_questions = []
    for context in contexts:
        tokens = word_tokenize(context.lower())
        base_words = [lemmatizer.lemmatize(token)
                      for token in tokens if token.isalpha()]
        filtered_base_words = [
            word for word in base_words if word not in custom_stop_words]
        processed_questions.append(filtered_base_words)
    # Create dictionary of processed questions
    vocabulary = corpora.Dictionary(processed_questions)

    # Represent each question as bag of words
    corpus_bow = [vocabulary.doc2bow(question)
                  for question in processed_questions]

    # Train the TF-IDF model
    tfidf = TfidfModel(corpus_bow)

    # Transform the BoW corpus to a TF-IDF corpus
    corpus_tfidf = tfidf[corpus_bow]

    # Model Tuning
    best_model = None
    best_coherence_score = -1
    score_dict = {}
    for n_topics in range(from_n_topics, to_n_topics):
        lda_model = LdaModel(corpus_tfidf,
                             num_topics=n_topics,
                             id2word=vocabulary,
                             alpha=0.01,
                             eta=0.01,
                             passes=20,
                             random_state=42
                             )

        perplexity_score = lda_model.log_perplexity(corpus_tfidf)

        coherence_model_lda = CoherenceModel(
            model=lda_model, texts=processed_questions, dictionary=vocabulary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        if (coherence_score > best_coherence_score):
            best_coherene_score = coherence_score
            best_model = lda_model
            score_dict[n_topics] = [perplexity_score, coherence_score]

        lda_model = best_model
        print(score_dict)

        # Visualize the topics
        vis = pyLDAvis.gensim_models.prepare(
            lda_model, corpus_tfidf, vocabulary)
        pyLDAvis.save_html(vis, 'lda_visualization.html')


if __name__ == '__main__':
    Compute_Coherence_BERTopic()

    # Change the from and to range based on requirements.
    # We already know the best number of topics are 7.
    Compute_Coherence_LDA(from_n_topics=7, to_n_topics=8)
