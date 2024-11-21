# Datasets used for Mental Health Chatbot

A. data/Mental_Health_FAQ.csv for informational response
B. Build a conversation graph by selecting some conversations. There is a lot in this dataset and we may need to curate this data.

<https://huggingface.co/datasets/Amod/mental_health_counseling_conversations> for mapping conversations appropriate to an emotion.

For example, for the emotion of sadness, we will pick some actual responses and create a conversation_graph.csv file.

User: I'm very depressed. How do I find someone to talk to?

Counseler: If there is no one in your life with whom you feel comfortable, I would recommend identifying someone on this site, psychology today or goodtherapy.org who is in your area, then give that therapist a call to see if you would be a good match.  

User: I have family issues, and my dad was both violent and a cheater.

Counseler: In general, our past is always somewhat alive and does inform our present experiences.  All past experiences, both positive and negative are what have brought you where you are today, physically and emotionally.   That being said, your father's behavior does not have to define you or your future.  You have power over your life and your future.  By learning more about yourself, you can learn to incorporate your past into your present life, without being defined by past traumas.  The best way to grow and learn how to do this is to talk with someone about your experiences.  By talking about your past and present struggles with a trusted confidante or helping professional you will hopefully learn how to be at peace with your past.

User: I've gone to a couple therapy sessions so far and still everytime I walk in I get nervous and shaky. Is this normal? Should I still be feeling like this?

Counseler: Certainly. It is ok to feel nervous. I wish you good luck!

This curated data will be added to the mental health counseling conversations as an example of multi-stage conversation.

So this conversation can be represented as a graph. If we do not have conversation data for specific situation of sadness, we will just use
canned_responses.csv to select the response and that would be the end of the conversation.

Pipeline of Mental Health Chatbot.

1. Load all datasets with pandas - df_A, df_B

2. Classify user's query into informational A or emotional B.

3. If user's query is informational, produce a response directly from the answers in the FAQ file.
   We already built, k nearest neighbor classifier for this purpose.

4. Generate distinct situation categories using Latent Dirichlet Allocation (LDA) generative statistical model for dataset B.

5. Build emotion graph based on dataset B. The graph is a network X graph which chat bot navigates.
   For example, let us say chatbot detects user wants to discuss their addiction issues.
   Then chatbot figures out the apppropriate topic and pick the best conversation for that topic.

6. We will use similarity based node matching to produce a response.

The canned response file is the mental health FAQ file.

A conversation graph will be created based on previous counseling sessions.
A conversation graph will enable us to have history and thus
chatbot will be able to have meaninful conversations.
We will use this data: <https://huggingface.co/datasets/Amod/mental_health_counseling_conversations>
So we need to add df_B dataframe to get this data.

Using conversation graph is better than producing canned responses.

## How Anchor works?

Anchor uses a model based on topic understanding and information rerieval.

1. The user submits a query.
2. Anchor first classifies a user query into informational and emotional.
3. If query is informational, Anchor extracts the response and provides it to the user. There is no need for conversation to continue in this case.
4. If the query is emotional, it identifies the specific situation a user may be in e.g. relationship issues, addiction issues etc.
5. It then maps user's query to a conversation graph.
6. Then the conversation flows according to the closest matching conversation graph in Anchor's database.
7. Each response will sent to a summarization engine which will produce a compact response.

## Summarization Engine

1. Divide the long response in sentences delimited by "." character.
2. Encode user query and each sentence with SentenceTransformer encoder. This step is called vectorization of sentences.
3. Compute the cosine similarity for each sentence and query using numpy dot and norm functions. An example is given here:
   similarity score = np.dot(query_embedding, sentence_embedding.T) / (
    norm(query_embedding) * norm(sentence_embedding)

4. Choose the relevance sentences
Initialization: Choose the sentence with the highest similarity to query. This is most relevant sentence in the summary.
Summary = {S0}

   Choose a parameter lambda = 0.7 # Weighting parameter
   k = 3 # Top k sentences

   * 4.1 Loop until top k sentences have been selected.
   * 4.2 Calculate the relevance of a sentence not currently in the summary set using, `Maximal Marginal Relevance` (**MMR**) with the formula:

   ```math
   MMR = lambda *Cosine Similarity(Q, S) - (1- lambda)* Highest Cosine Similarity to already existing 
   ```

   * 4.2.1 sentences in the summary.
      Choose the sentence with maximum MMR score and add to the summary.
   * 4.3 Go back to 4.1

5. Information Ordering: Order the sentences in the summary by their original ordering. For example, we get the following summary set:
   Summary = {S0, S3, S5}

But the original ordering in the text was S3, S0 and then S5, then our summary would in the original order. <=== Our novelty

## Model Performance

We evaluated two performance metrics, Model coherence and Model Perplexity. Model coherence indicates the coherence of topic words in the LDA model.
The best coherence metric of 0.45 was obtained when number of topics were set to 7.
The coherence metric of 0.45 indicates that topics are well formed, although there is a room for improvement.

The perplexity metric indicates how good the natural language model is in its predictions/ In the context of Latent Dirichlet Allocation (LDA), perplexity is a measure of how well the model fits the given set of documents, similar to how it's used in language models. Specifically, for LDA, perplexity quantifies how "surprised" the model is by the words in a held-out test set of documents, based on the topics it learned during training.
We obtained moderately good values for log perplexity in the range of [-8, -9]

For LDA, a lower (more negative) log perplexity generally suggests better model fit to the data. If the LDA model achieves a very negative log-perplexity score and good topic coherence, it may be better at representing specific topics or contextually relevant topics for the data.

Comparing the model with hugging face model : [mental-health-mistral-7b-instructv0.2-finetuned-V2] (<https://huggingface.co/GRMenon/mental-health-mistral-7b-instructv0.2-finetuned-V2>)
