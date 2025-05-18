# Biomedical Tokenizer For FNLP 2025 Spring HW3

This project aims to train a WordPiece tokenizer on a biomedical corpus and utilize it to enhance a BERT model for classification tasks. The project includes the following components:

## Project Structure

- **data/**: Contains the datasets used in the project.
  - `pubmed_sampled_corpus.jsonline`: The biomedical corpus used for training the WordPiece tokenizer.
  - `HoC`: The HoC dataset used for training the classifier.

- **src/**: Contains the source code for the project.
  - **tokenizer/**: Implements the WordPiece tokenizer training.
    - `train_tokenizer.py`: Script for training the WordPiece tokenizer on the biomedical corpus.
  - **model/**: Contains the logic for adjusting the BERT model and training the classifier.
    - `train_classifier.py`: Trains a classifier using the modified BERT model.
    - `evaluate.py`: Evaluates the performance of the trained classifier.
  - **config/**: Configuration settings for the project.
    - `config.yaml`: Contains paths to datasets, model parameters, and training settings.

- `requirements.txt`: Lists the dependencies required for the project, including libraries for tokenization, model training, and evaluation.

- `README.md`: Documentation for the project, including setup instructions and usage.


## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd biomedical-tokenizer-project
   ```

2. Install the required dependencies:
   First get your torch ready!
   then:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the datasets:
   from course.pku.edu.cn
## Usage

1. **Train the WordPiece Tokenizer**:
   Run the following command to train the tokenizer on the biomedical corpus:
   ```
   python src/tokenizer/train_tokenizer.py --config src/config/expanded.yaml
   ```

2. **Train the Classifier**:
   Train the classifier using the original BERT model:
   ```
   python src/model/train_classifier.py --config src/config/ori.yaml
   ```
   Train the classifier using the modified BERT model:
   ```
   python src/model/train_classifier.py --config src/config/expanded.yaml
   ```

3. **Evaluate the Classifier**:
   Evaluate the performance of the trained classifier: (modify your ckpt path in .yaml)
   ```
   python src/model/evaluate.py --config src/config/config.yaml
   ```
