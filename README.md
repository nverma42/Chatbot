# Mental Health Chatbot

A sophisticated mental health support system that provides both informational responses and emotional support through machine learning-based query classification and conversation graph navigation. The chatbot uses advanced NLP techniques including Sentence Transformers, LDA topic modeling, and MMR-based summarization to deliver contextually appropriate and empathetic responses.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Using Conda](#using-conda)
  - [Using venv](#using-venv)
- [Running the Chatbot](#running-the-chatbot)
- [Project Structure](#project-structure)
- [Schema Definition](#schema-definition)
- [Usage](#usage)
- [Credits](#credits)

## Features

- **Dual Response Modes**: Informational (FAQ-based) and Emotional (conversation graph-based)
- **Advanced NLP**: Sentence Transformers for semantic understanding
- **Topic Modeling**: LDA for emotion/situation categorization
- **Smart Summarization**: MMR algorithm for concise, relevant responses
- **GPU Acceleration**: Automatic device management with multi-GPU support
- **Session Management**: Conversation history and summarization

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, falls back to CPU)
- Core Dependencies:
  - `sentence-transformers`: Semantic embeddings
  - `scikit-learn`: Classification algorithms
  - `pandas`: Data processing
  - `torch`: Deep learning backend
  - `networkx`: Conversation graph management
  - `gensim`: LDA topic modeling
  - `absl-py`: Command-line flags

## Installation

### Using Conda

1. **Install Conda**: Make sure Conda is installed on your machine. If not, download and install it from [Conda's website](https://docs.conda.io/en/latest/miniconda.html).

2. **Create a Conda environment**:

```bash
conda create -n chatbot_env python=3.8
```

3. **Activate the environment:**:

```bash
conda activate chatbot_env
```

4. **Install dependencies**: You can install the required dependencies using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

5. **Installing `sentence_transformers` Using Conda**

```bash
conda install -c conda-forge sentence-transformers
```

*if this fails try* [from anaconda.org](https://anaconda.org/conda-forge/sentence-transformers):

```bash
conda install conda-forge::sentence-transformers
```

### Using venv

1. **Install venv (if needed)**: If you don’t have venv installed, you can do so with the following command:

```bash
sudo apt install python3-venv   # On Ubuntu/Debian
```

2. **Create a virtual environment**:

```bash
python3 -m venv chatbot_env
```

3. **Activate the environment**:

    **On Linux/macOS**:

    ```bash
    source chatbot_env/bin/activate
    ```

    **On Windows**:

    ```bash
    chatbot_env\Scripts\activate
    ```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### Installation

### Option 1: Install as Package

```bash
pip install -e .
```

This will install the chatbot as a package, making it available from anywhere.

### Option 2: Direct Installation

Follow the conda or venv setup instructions above, then:

```bash
pip install -r requirements.txt
```

## Running the Chatbot

### If installed as package

```bash
mhchatbot --faq_data_path ./data/Mental_Health_FAQ.csv --conversations_data_path ./data/mental_health_counseling_conversations.csv
```

### Direct execution

```bash
python src/mhchatbot/main.py --faq_data_path ./data/Mental_Health_FAQ.csv --conversations_data_path ./data/mental_health_counseling_conversations.csv
```

### Optional Arguments

- `--faq_data_path`: Path to the FAQ dataset CSV file (default is ./data/Mental_Health_FAQ.csv).
- `--empathetic_data_path`: Path to the directory containing the empathetic dialogues datasets (default is ./data/empathetic-dialogues-contexts/).
- `--conversations_data_path`: Path to the CSV file containing mental health counseling conversations (default is ./data/mental_health_counseling_conversations.csv).
- `--test_size`: Fraction of data to use for testing (default is 0.3).
- `--random_state`: Seed for random number generation (default is 42).

The chatbot will start running, and you can interact with it via the command line. Type your query, and the chatbot will categorize it as either informational or emotional, providing a suitable response.
To exit the chatbot, you can type any of the following:

- `exit`
- `quit`
- `q`
- `x`
- `e`
- `Or press Ctrl+C (keyboard interrupt)`

## Project Structure

```
mhchatbot/
│
├── src/
│   └── mhchatbot/
│       ├── __init__.py
│       ├── main.py              # Entry point
│       ├── chatbot.py           # Core chatbot logic
│       ├── emotional_response.py # Emotional query handling
│       └── summarization.py     # Response summarization
├── tests/
│   ├── __init__.py
│   ├── test_emotional_response.py
│   └── test_summarization.py
├── scripts/
│   └── model_comparison.py      # Model evaluation scripts
├── docs/
│   └── visuals/                 # Visualizations
├── data/
│   └── Mental_Health_FAQ.csv
├── config/                      # Configuration files
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

### Schema Definition

The project structure is formally defined in `project-structure.graphql`:

```graphql
# Import from ./project-structure.graphql
type Project {
  name: String!
  rootDirectory: Directory!
}

# See complete schema in project-structure.graphql
```

### Key Components

- **Source Files**:
  - `main.py`: Entry point for the chatbot application
  - `chatbot.py`: Core chatbot implementation
  - `utils.py`: Helper functions and utilities

- **Data Directory**:
  - Mental health FAQ dataset
  - Canned responses
  - Empathetic dialogues training data

- **Configuration**:
  - `requirements.txt`: Project dependencies
  - `README.md`: Documentation and setup instructions

### Core Modules

#### src/mhchatbot/
- **main.py**: Application entry point
  - GPU/CPU device detection and management
  - Interactive session handling
  - Command-line argument processing
  
- **chatbot.py**: Core chatbot engine
  - Logistic Regression for query classification (A/B)
  - KNN classifier for FAQ matching
  - Integration with emotional response system
  
- **emotional_response.py**: Emotional support module
  - LDA topic modeling (7 topics)
  - Conversation graph navigation
  - NetworkX-based response generation
  
- **summarization.py**: Response optimization
  - MMR (Maximal Marginal Relevance) implementation
  - Sentence ordering preservation
  - Configurable summary length

#### Supporting Components
- **tests/**: Comprehensive unit test suite
- **scripts/**: Model evaluation and comparison tools
- **data/**: Dataset storage (CSV, JSON formats)
- **docs/visuals/**: Topic model visualizations
- **config/**: Environment-specific configurations

## Usage Guide

### Interactive Commands

- **Chat**: Type any mental health-related query
- **Summarize**: Type `summarize` to get conversation summary
- **Exit**: Type `exit`, `quit`, `q`, `x`, `e`, or press `Ctrl+C`

### Query Types and Responses

#### Informational Queries (Type A)
Factual questions about mental health, symptoms, treatments:
```
You: What are the symptoms of anxiety?
Chatbot: Common symptoms include nervousness, restlessness, rapid heartbeat, 
         difficulty concentrating, and a sense of impending danger.
```

#### Emotional Support Queries (Type B)
Personal struggles, seeking empathy and guidance:
```
You: I'm feeling really stressed and overwhelmed.
Chatbot: I understand you're going through a difficult time. It's completely 
         normal to feel overwhelmed. Would you like to talk about what's 
         causing your stress?
```

### Advanced Features

1. **Multi-turn Conversations**: The chatbot maintains context across exchanges
2. **Session Summaries**: Automatic summarization on exit
3. **GPU Acceleration**: Utilizes available CUDA devices for faster processing
4. **Adaptive Responses**: LDA-based topic matching for relevant support

## Technical Details

### Machine Learning Pipeline

1. **Query Encoding**: Sentence-BERT (paraphrase-MiniLM-L6-v2)
2. **Classification**: Logistic Regression (binary: A/B)
3. **Information Retrieval**: KNN (k=5) for FAQ matching
4. **Topic Modeling**: LDA with 7 topics for emotional categorization
5. **Graph Navigation**: NetworkX for conversation flow
6. **Summarization**: MMR algorithm (λ=0.7, k=3 sentences)

### Performance Metrics

- LDA Coherence Score: 0.45
- Log Perplexity: [-8, -9] range
- GPU Memory Requirement: 4GB minimum
- Response Time: <2 seconds (GPU), <5 seconds (CPU)

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns and conventions
- Tests are added for new functionality
- Documentation is updated accordingly

## License

This project is intended for educational and research purposes in mental health support systems.

## Acknowledgments

- **Sentence-BERT**: Semantic similarity computation
- **Scikit-learn**: Machine learning algorithms
- **Hugging Face**: Mental health conversation datasets
- **Kaggle**: Mental health FAQ dataset
