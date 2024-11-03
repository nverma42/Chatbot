import nltk
from nltk.corpus.reader import documents
import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel, TfidfModel
import sentence_transformers
from numpy.linalg import norm
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import random

class Emotional_Response(object):

    # Lemmatize and filter stop words
    def get_filtered_base_words(self, query):
        tokens = word_tokenize(query.lower())
        base_words = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        filtered_base_words = [word for word in base_words if word not in self.custom_stop_words]
        return filtered_base_words

    # Preprocess context data by lemmatizing and filtering out stop words.
    # This way, we can focus only on the relevant parts of the text.
    def preprocess_data(self, contexts):
        nltk.download('stopwords')
        nltk.download('wordnet')

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        custom_stop_words = stop_words.union(['wa',
                                              'ha',
                                              'doe',
                                              'know',
                                              'want',
                                              'something',
                                              'someone',
                                              'person',
                                              'process',
                                              'start',
                                              'nothing',
                                              'make',
                                              'really',
                                              'get',
                                              'like',
                                              'know',
                                              'many',
                                              'say',
                                              'walk',
                                              'one',
                                              'two',
                                              'ca',
                                              'would',
                                              'said',
                                              'still',
                                              'got',
                                              'go',
                                              'back',
                                              'lot',
                                              'u',
                                              'think',
                                              'feel',
                                              'need',
                                              'help',
                                              'issue',
                                              'told',
                                              'always',
                                              'stop',
                                              'even',
                                              'see',
                                              'ever',
                                              'thought',
                                              'thing',
                                              'everything',
                                              'year',
                                              'going',
                                              'around',
                                              'time',
                                              'keep',
                                              'also',
                                              'much',
                                              'day',
                                              'tell',
                                              'anything',
                                              'give',
                                              'normal',
                                              'never',
                                              'every',
                                              'stay',
                                              'month',
                                             ])

        # Preprocess questions
        processed_questions = []
        for context in contexts:
            filtered_base_words = self.get_filtered_base_words(context)
            processed_questions.append(filtered_base_words)
        
        return processed_questions

    # Apply Latent Dirichlet Allocation is a statistical generative model which uses
    # co-occurrence of various words to identify a certain topic. LDA is widely used for
    # topic modeling.
    def apply_lda_model(self, processed_data):
        self.dictionary = corpora.dictionary(processed_data)

        # Represent each data point by bag of words
        self.corpus_bow = [self.dictionary.doc2bow(doc) for doc in processed_data]

        # Initialize tf-idf model
        self.tfidf_model = TfidfModel(self.corpus_bow)

        # Transform the bag of words model to tf-idf model so that terms are weighted properly.
        self.corpus_tfidf = self.tfidf_model[self.corpus_bow]
        n_topics = 8
        self.lda_model = LdaModel(self.corpus_tfidf,
                                  num_topics=n_topics,
                                  id2word=self.dictionary,
                                  alpha=0.01,
                                  eta=0.01,
                                  passes=20,
                                  random_state=42)
        
        self.topics = self.lda_model.print_topics(num_words=25)

        for idx, topic in self.topics:
            print(f'Topic:{idx}')

            # Split the topic words into word, probability pairs
            word_probs = topic.split(' + ')
            for word_prob in word_probs:
                prob, word = word_prob.split('*')
                print(f'Word={word} Probability={prob}')

    def get_topic(self, query):
        filtered_base_words = self.get_filtered_base_words(query)
        doc_bow = self.dictionary.doc2bow(filtered_base_words)
        doc_tfidf = self.tfidf[doc_bow]
        topics = self.lda_model.get_document_topics(doc_tfidf)
        topic = max(topics, key=lambda x:x[1])[0]
        return topic

    def tag_documents(self):
        self.df['Topic'] = -1
        self.df['Conv_Id'] = -1

        for idx, row in self.df.iterrows():
            topic = self.get_topic(row['Context'])
            self.df.at[idx, 'Topic'] = topic
    
            # Each conversation in the present data is independent
            self.df.at[idx, 'Conv_Id'] = idx

    def add_new_conv_graph(self, row):
        conv_graph = nx.DiGraph()
        from_node = row['Context']
        to_node = row['Response']
        conv_graph.add_node(from_node, conv_id=row['Conv_Id'], type='Context')
        conv_graph.add_node(to_node, conv_id=row['Conv_Id'], type='Response')
        conv_graph.add_edge(from_node, to_node)
        topic = row['Topic']
        if topic not in self.graph_dict:
            self.graph_dict[topic] = {}
        self.graph_dict[topic][row['Conv_Id']] = conv_graph

    # Extend conversation graph by finding the leaf node
    # Conversations are linear by design presently.
    # In future, probabilistic conversations can be added.
    def extend_conv_graph(conv_graph, row):
        leaf_nodes = [node for node in conv_graph.nodes if conv_graph.out_degree(node) == 0]
    
        # Choose the first leaf node for insertion.
        new_from_node = row['Context']
        new_to_node = row['Response']
        conv_graph.add_edge(leaf_nodes[0], new_from_node)
        conv_graph.add_node(new_from_node, conv_id=row['Conv_Id'], type='Context')
        conv_graph.add_node(new_to_node, conv_id=row['Conv_Id'], type='Response')
        conv_graph.add_edge(new_from_node, new_to_node)

    def make_conversation_grah(self):
        self.graph_dict = {}
        conv_id = -1
        for idx, row in self.df.iterrows():
            topic = row['Topic']
            conv_id = row['Conv_Id']
            if (topic in self.graph_dict):
                if row['Conv_Id'] in self.graph_dict[topic]:
                    conv_graph = self.graph_dict[topic][conv_id]
                    self.extend_conv_graph(conv_graph, row)
            else:
                # It is a new conversation but check if duplicate context exists.
                found = False
                for conv_id, conv_graph in self.graph_dict[topic].items():
                    if (row['Context'] in conv_graph):
                        conv_graph.add_edge(row['Context'], row['Response'])
                        found = True
                        break
            
                if (found == False):
                    self.add_new_conv_graph(row)
                else:
                    self.add_new_conv_graph(row)

    def __init__(self):
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.df = pd.read_json("hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)
        contexts = self.df['Context'].tolist()
        processed_data = self.preprocess_data(contexts)
        self.apply_lda_model(processed_data)
        self.tag_documents()
        self.make_conversation_grah()


    # Work-horse: Get the emotional query response by navigating the graph.
    def get_response(self, query):
        topic = self.get_topic(query)

        # Find the best matching conversation by matching the root
        # node of conversation with the query.
        query_embedding = self.encoder.encode(query)
        sim_score = 0.0
        target_G = None
        target_node = None
        for topic, conv_graph in self.graph_dict[topic].items():
            root = None
            for node in conv_graph:
                if (conv_graph.in_degree(node) == 0):
                    root = node
                    break
            if (root):
                root_embedding = self.encoder.encode(root)
                score = np.dot(query_embedding, root_embedding) / (norm(query_embedding) * norm(root_embedding))
                if (score > sim_score):
                    target_G = conv_graph
                    target_node = root
                    sim_score = score


        # Get the neighbors of the target node
        neighbors = list(target_G.neighbors(target_node))

        if (neighbors):
            selected_neighbor = random.choice(neighbors)
            return selected_neighbor
        else:
            default_response = "Sorry, I do not have enough material on this topic to help you."
            return default_response




