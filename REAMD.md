# Mental Health Chatbot

This project focuses on building a mental health chatbot that can provide empathetic responses and offer basic mental health information. The chatbot uses a logistic regression classifier to categorize user queries into two groups: information-seeking (class A) and emotional support-seeking (class B). The chatbot then generates appropriate canned responses based on these classifications.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Using Conda](#using-conda)
  - [Using venv](#using-venv)
- [Running the Chatbot](#running-the-chatbot)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Credits](#credits)

## Requirements

- Python 3.8 or higher
- Libraries:
  - absl-py
  - pandas
  - sentence-transformers
  - scikit-learn

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

### Running the Chatbot

Once you have installed the necessary dependencies and activated your environment, you can run the chatbot using the following command:

```bash
python main.py --faq_data_path ./data/Mental_Health_FAQ.csv --empathetic_data_path ./data/empathetic-dialogues-contexts --canned_responses_path ./data/canned_responses.csv
```

*alt view*:

```powershell
python main.py --faq_data_path ./data/Mental_Health_FAQ.csv \
               --empathetic_data_path ./data/empathetic-dialogues-contexts/ \
               --canned_responses_path ./data/canned_responses.csv
```

### Optional Arguments

- `--faq_data_path`: Path to the FAQ dataset CSV file (default is ./data/Mental_Health_FAQ.csv).
- `--empathetic_data_path`: Path to the directory containing the empathetic dialogues datasets (default is ./data/empathetic-dialogues-contexts/).
- `--canned_responses_path`: Path to the canned responses CSV file (default is `./data/canned_responses.csv`).
- `--test_size`: Fraction of data to use for testing (default is 0.3).
- `--random_state`: Seed for random number generation (default is 42).

The chatbot will start running, and you can interact with it via the command line. Type your query and receive a response. Type `exit` or `quit` to stop the chatbot.

### Usage

Once the chatbot is running, type any mental health-related query.
The chatbot will categorize your query as either informational or emotional.
It will respond with canned answers based on the classification.

**Example interaction**:

```bash
You: I'm feeling really stressed out lately.
Chatbot: I'm sorry to hear that you're feeling stressed. Would you like to talk about it?

You: What are the symptoms of anxiety?
Chatbot: Some common symptoms of anxiety include nervousness, restlessness, and a sense of impending danger.
```

### Credits

- **Sentence-BERT**: Used for encoding questions and contexts.
- **Scikit-learn**: Used for implementing classifiers.
- **Pandas**: For handling data loading and preprocessing.
