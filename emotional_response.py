"""
This module provides the EmotionalResponse class, which generates emotional responses
based on input queries by utilizing NLP techniques such as LDA for topic modeling
and semantic similarity for response selection.
"""
import random
import logging
from typing import List, Dict
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import torch
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from networkx import DiGraph
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# Set up logging
logger = logging.getLogger(__name__)


class EmotionalResponse:
    """
    The EmotionalResponse class processes conversational data to generate appropriate
    emotional responses to user queries. It leverages topic modeling and graph-based
    conversation structures to find the best response.
    """

    def __init__(self, sentence_encoder="paraphrase-MiniLM-L6-v2", device='cpu', gpu_ids=None) -> None:
        """
        Initializes the EmotionalResponse instance by loading data, preprocessing it,
        and setting up the necessary models and graphs.

        Args:
            sentence_encoder (str): this is the string to represent the sentence transformer, default: paraphrase-MiniLM-L6-v2.
            device (str): Device to load models onto ('cpu', 'cuda:X').
            gpu_ids: help keep track of which gpu(s) are avaialbe for distributed work.
        """
        self.device = device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.gpu_ids = gpu_ids
        self.encoder = SentenceTransformer(sentence_encoder)
        self.encoder.to(self.device)

        try:
            self.df = pd.read_json(
                "hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json",
                lines=True
            )
        except Exception as e:
            raise FileNotFoundError(f"Data file could not be loaded: {e}")

        self.custom_stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        contexts = self.df['Context'].tolist()
        processed_data = self.preprocess_data(contexts)

        self.apply_lda_model(processed_data)
        self.tag_documents()
        self.make_conversation_graph()

    def preprocess_data(self, contexts: List[str]) -> List[List[str]]:
        """
        Preprocesses context data by lemmatizing and filtering out stop words.

        Args:
            contexts (List[str]): List of context strings.

        Returns:
            List[List[str]]: List of processed context tokens.
        """
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)

        processed_questions = []
        for context in contexts:
            filtered_base_words = self.get_filtered_base_words(context)
            processed_questions.append(filtered_base_words)

        return processed_questions

    def get_filtered_base_words(self, query: str) -> List[str]:
        """
        Tokenizes, lemmatizes, and filters stop words from the input query.

        Args:
            query (str): The input query string.

        Returns:
            List[str]: List of filtered base words.
        """
        tokens = word_tokenize(query.lower())
        base_words = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha()
        ]
        filtered_base_words = [
            word for word in base_words if word not in self.custom_stop_words
        ]
        return filtered_base_words

    def apply_lda_model(self, processed_data: List[List[str]]) -> None:
        """
        Applies LDA topic modeling on the processed data.

        Args:
            processed_data (List[List[str]]): List of processed context tokens.
        """
        self.dictionary = corpora.Dictionary(processed_data)

        # Represent each data point by bag of words
        self.corpus_bow = [self.dictionary.doc2bow(
            doc) for doc in processed_data]

        # Initialize tf-idf model
        self.tfidf_model = TfidfModel(self.corpus_bow)

        # Transform the bag of words model to tf-idf model
        self.corpus_tfidf = self.tfidf_model[self.corpus_bow]

        # Model Tuning
        # best_model = None
        # best_coherence_score = -1
        # score_dict = {}
        # for n_topics in range(2, 10):
        #     lda_model = LdaModel(
        #         self.corpus_tfidf,
        #         num_topics=n_topics,
        #         id2word=self.dictionary,
        #         alpha=0.01,
        #         eta=0.01,
        #         passes=20,
        #         random_state=42
        #         )

        #     perplexity_score = lda_model.log_perplexity(self.corpus_tfidf)

        #     coherence_model_lda = CoherenceModel(model= lda_model,
        #       texts=processed_data, dictionary=self.dictionary, coherence='c_v')
        #     coherence_score = coherence_model_lda.get_coherence()
        #     if (coherence_score > best_coherence_score):
        #         best_coherene_score = coherence_score
        #         best_model = lda_model

        #     score_dict[n_topics] = [perplexity_score, coherence_score]

        # self.lda_model = best_model
        # print(score_dict)

        n_topics = 7
        self.lda_model = LdaModel(
            self.corpus_tfidf,
            num_topics=n_topics,
            id2word=self.dictionary,
            alpha=0.01,
            eta=0.01,
            passes=20,
            random_state=42
        )
        self.topics = self.lda_model.print_topics(num_words=25)

        # for idx, topic in self.topics:
        #     print(f'Topic:{idx}')
        #     # Split the topic words into word, probability pairs
        #     word_probs = topic.split(' + ')
        #     for word_prob in word_probs:
        #         prob, word = word_prob.split('*')
        #         print(f'Word={word.strip()} Probability={prob}')

    def get_topic(self, query: str) -> int:
        """
        Determines the topic of a given query using the LDA model.

        Args:
            query (str): The input query string.

        Returns:
            int: The topic number with the highest probability.
        """
        # Ensure dictionary is initialized
        if not hasattr(self, 'dictionary'):
            raise AttributeError(
                "Dictionary is not initialized. Run apply_lda_model first.")

        filtered_base_words = self.get_filtered_base_words(query)
        doc_bow = self.dictionary.doc2bow(filtered_base_words)
        doc_tfidf = self.tfidf_model[doc_bow]
        topics = self.lda_model.get_document_topics(doc_tfidf)
        if topics:
            topic = max(topics, key=lambda x: x[1])[0]
            return topic
        else:
            return -1

    def tag_documents(self) -> None:
        """
        Tags each document in the DataFrame with its corresponding topic and conversation ID.
        """
        # Ensure dictionary is initialized
        if not hasattr(self, 'dictionary'):
            raise AttributeError(
                "Dictionary is not initialized. Run apply_lda_model first.")

        self.df['Topic'] = -1
        self.df['Conv_Id'] = -1

        for idx, row in self.df.iterrows():
            topic = self.get_topic(row['Context'])
            self.df.at[idx, 'Topic'] = topic

            # Each conversation in the present data is independent
            self.df.at[idx, 'Conv_Id'] = idx

    def make_conversation_graph(self) -> None:
        """
        Constructs graphs representing conversations for each topic.
        """
        self.graph_dict: Dict[int, Dict[int, DiGraph]] = {}

        for _, row in self.df.iterrows():
            topic = row['Topic']
            conv_id = row['Conv_Id']

            if topic not in self.graph_dict:
                self.graph_dict[topic] = {}

            if conv_id in self.graph_dict[topic]:
                conv_graph = self.graph_dict[topic][conv_id]
                self.extend_conv_graph(conv_graph, row)
            else:
                self.add_new_conv_graph(row)

    def add_new_conv_graph(self, row: pd.Series) -> None:
        """
        Adds a new conversation graph for a topic.

        Args:
            row (pd.Series): A row from the DataFrame containing conversation data.
        """
        conv_graph = nx.DiGraph()
        from_node = row['Context']
        to_node = row['Response']
        conv_graph.add_node(from_node, conv_id=row['Conv_Id'], type='Context')
        conv_graph.add_node(to_node, conv_id=row['Conv_Id'], type='Response')
        conv_graph.add_edge(from_node, to_node)

        topic = row['Topic']
        self.graph_dict[topic][row['Conv_Id']] = conv_graph

    def extend_conv_graph(self, conv_graph: DiGraph, row: pd.Series) -> None:
        """
        Extends an existing conversation graph by adding new nodes and edges.

        Args:
            conv_graph (DiGraph): The conversation graph to extend.
            row (pd.Series): A row from the DataFrame containing conversation data.
        """
        # Find leaf nodes (nodes with no outgoing edges)
        leaf_nodes = [
            node for node in conv_graph.nodes if conv_graph.out_degree(node) == 0]

        if leaf_nodes:
            new_from_node = row['Context']
            new_to_node = row['Response']
            conv_graph.add_node(
                new_from_node, conv_id=row['Conv_Id'], type='Context')
            conv_graph.add_node(
                new_to_node, conv_id=row['Conv_Id'], type='Response')
            conv_graph.add_edge(leaf_nodes[0], new_from_node)
            conv_graph.add_edge(new_from_node, new_to_node)

    def get_response(self, query: str) -> str:
        """
        Generates an emotional response to the input query by navigating the conversation graph.
        """
        try:
            topic = self.get_topic(query)

            # Find the best matching conversation
            query_embedding = self.encode_text(query)
            query_embedding = query_embedding.cpu().numpy()

            sim_score = 0.0
            target_graph = None
            target_node = None

            for conv_graph in self.graph_dict.get(topic, {}).values():
                root = next(
                    (node for node in conv_graph if conv_graph.in_degree(node) == 0),
                    None
                )
                if root:
                    root_embedding = self.encode_text(root)
                    root_embedding = root_embedding.cpu().numpy()

                    score = np.dot(query_embedding, root_embedding.T) / (
                        norm(query_embedding) * norm(root_embedding)
                    )
                    if score > sim_score:
                        target_graph = conv_graph
                        target_node = root
                        sim_score = score

            # Get the neighbors of the target node
            if target_graph and target_node:
                neighbors = list(target_graph.neighbors(target_node))
                if neighbors:
                    selected_neighbor = random.choice(neighbors)
                    return selected_neighbor

            return "I understand you're going through a difficult time. While I'm here to listen, it might be helpful to speak with a mental health professional who can provide more personalized support."

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble processing your message. Could you try expressing that in a different way?"

    def encode_text(self, texts, batch_size=32):
        """
        Wrapper method to handle encoding with batching and proper error handling.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            # Process in batches if needed
            if len(texts) > batch_size:
                encodings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_encoding = self.encoder.encode(
                        batch,
                        convert_to_tensor=True,
                        device=self.device
                    )
                    encodings.append(batch_encoding.cpu())
                return torch.cat(encodings, dim=0)
            else:
                return self.encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    device=self.device
                )

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Fall back to CPU
                self.device = torch.device('cpu')
                self.encoder.to(self.device)
                # Retry with new device
                return self.encode_text(texts, batch_size)
            raise e
