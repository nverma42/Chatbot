# Datasets used for Mental Health Chatbot.

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
   
7. We will use similarity based node matching to produce a response.

```py
We do not neeed data/canned_response.csv.
The canned response file is the mental health FAQ file.

A conversation graph will be created based on previous counseling sessions.
A conversation graph will enable us to have history and thus
chatbot will be able to have meaninful conversations.
We will use this data: <https://huggingface.co/datasets/Amod/mental_health_counseling_conversations>
So we need to add df_B dataframe to get this data.

Using conversation graph is better than producing canned responses.

How will this process work?

This data has conversations with a counselor.
We will classify each of these conversations with a situation e.g. relationship issues because we have LDA classifier now.
Thus each situation will have a separate graph. Let us say when someone is
sad because of relationship issues, how conversation with a counseler proceeds is very predictable.
We will first classify user's situation.
The ChatBot will then map user's query to a graph.
Then the conversation will proceed accordingly.
Each response will sent to a summarization engine which will produce a compact response.
For the purpose of demo, we will show both response.
We do not have a lot of conversation data, so chatbot won't
be able to chat for long but for our project this should be fine.
```