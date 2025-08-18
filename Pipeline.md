# Mental Health Chatbot Pipeline

## Datasets

### Primary Data Sources

1. **Mental Health FAQ Dataset**
   - Source: [Kaggle Mental Health FAQ](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)
   - Purpose: Provides informational responses to mental health queries
   - Format: CSV with question-answer pairs

2. **Mental Health Counseling Conversations**
   - Source: [Hugging Face Dataset](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
   - Purpose: Builds conversation graphs for emotional support responses
   - Format: JSON with counseling session transcripts

## System Architecture

### Core Components

1. **Query Classification Module**
   - Binary classifier using Logistic Regression
   - Categories:
     - Type A: Informational queries (mental health facts, symptoms, treatments)
     - Type B: Emotional support queries (personal struggles, seeking empathy)

2. **Response Generation Systems**
   - **Informational Response System**: KNN classifier with FAQ dataset
   - **Emotional Response System**: Conversation graph navigation with LDA topic modeling

3. **Summarization Engine**
   - MMR-based extractive summarization
   - Preserves original sentence ordering for coherence

## Processing Pipeline

### Step-by-Step Workflow

1. **Data Loading**
   ```python
   - Load FAQ dataset (df_A)
   - Load counseling conversations (df_B)
   - Initialize Sentence Transformer model
   ```

2. **Query Classification**
   ```python
   Input: User query
   Process: Logistic Regression classification
   Output: Query type (A: Informational, B: Emotional)
   ```

3. **Response Generation**

   **For Informational Queries (Type A):**
   ```python
   - Encode query using Sentence Transformer
   - KNN search in FAQ embeddings
   - Return most similar FAQ answer
   ```

   **For Emotional Queries (Type B):**
   ```python
   - Apply LDA topic modeling (7 topics)
   - Map query to conversation graph
   - Navigate graph using similarity matching
   - Generate contextual response
   ```

4. **Response Optimization**
   ```python
   - Apply summarization if response > threshold
   - Use MMR algorithm for sentence selection
   - Preserve original sentence ordering
   ```

## Conversation Graph Structure

### Example Multi-turn Conversation

```
User: "I'm very depressed. How do I find someone to talk to?"
├── Counselor: "If there is no one in your life with whom you feel comfortable..."
│
User: "I have family issues, and my dad was both violent and a cheater."
├── Counselor: "In general, our past is always somewhat alive and does inform..."
│
User: "I've gone to therapy sessions but still get nervous. Is this normal?"
└── Counselor: "Certainly. It is ok to feel nervous. I wish you good luck!"
```

### Graph Navigation Algorithm

1. Extract user emotion/topic using LDA
2. Find closest matching node in conversation graph
3. Traverse edges based on similarity scores
4. Return appropriate counselor response

## Summarization Engine Details

### MMR (Maximal Marginal Relevance) Algorithm

1. **Sentence Segmentation**
   - Split response by sentence boundaries
   - Encode each sentence using Sentence Transformer

2. **Similarity Calculation**
   ```python
   similarity = np.dot(query_embedding, sentence_embedding.T) / 
                (norm(query_embedding) * norm(sentence_embedding))
   ```

3. **MMR Scoring**
   ```math
   MMR = λ × Sim(Q, S) - (1-λ) × max(Sim(S, Si))
   ```
   Where:
   - λ = 0.7 (relevance vs. diversity trade-off)
   - Q = query embedding
   - S = candidate sentence
   - Si = already selected sentences

4. **Selection Process**
   - Initialize with highest similarity sentence
   - Iteratively add sentences with maximum MMR score
   - Stop at k=3 sentences (configurable)

5. **Original Order Preservation** (Novel Contribution)
   - Reorder selected sentences by original position
   - Maintains narrative flow and coherence

## Model Performance Metrics

### LDA Topic Model
- **Coherence Score**: 0.45 (7 topics)
- **Log Perplexity**: [-8, -9] range
- Indicates well-formed topics with room for improvement

### Comparison with BERTopic
- BERTopic identified 148 fine-grained topics
- Similar coherence metric but higher granularity
- LDA chosen for computational efficiency

### Classification Performance
- Logistic Regression for query type classification
- KNN for FAQ matching (k=5 default)
- Evaluation metrics: Precision, Recall, F1-score

## Implementation Notes

### Device Management
- Automatic GPU detection and allocation
- Fallback to CPU if insufficient GPU memory (<4GB)
- Multi-GPU support for large-scale processing

### Session Management
- Conversation history tracking
- Session summarization on exit
- Interactive command handling (exit, summarize)

## Future Enhancements

1. **Dynamic Response Generation**
   - Integration with transformer-based language models
   - Fine-tuning on mental health domain

2. **Multi-lingual Support**
   - Language detection and translation
   - Culturally-aware responses

3. **Risk Assessment**
   - Crisis detection algorithms
   - Automatic referral to professional help

4. **Emotion Intensity Modeling**
   - Graduated response based on emotional severity
   - Adaptive conversation strategies