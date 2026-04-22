s-chatbot — Banking Intent Classifier & Chatbot
A machine learning project that classifies banking customer queries into 77 intent categories using the Banking77 dataset, and returns a scripted chatbot response based on the predicted intent.
Built as part of a University of Hertfordshire Data Science coursework project.

Project Structure
ds-chatbot/
├── data/
│   └── raw/
│       ├── train.csv          # 10,003 labelled training samples
│       └── test.csv           # 3,080 test samples
├── models/                    # Saved model files (if exported)
├── notebooks/
│   ├── 01_eda.ipynb           # Main notebook: EDA → models → chatbot demo
│   └── .ipynb_checkpoints/
├── reports/                   # Any exported figures or reports
├── src/                       # (Reserved for modular source code)
├── .venv/                     # Python virtual environment (not committed)
├── .gitignore
├── README.md
└── requirements.txt


Dataset
Banking77 — a single-domain intent detection dataset for online banking.
Split	Samples	Classes
Train	10,003	77
Test	3,080	77

·	Average query length: ~12 words
·	No missing values in either split
·	Reasonably balanced classes (min 35, max 187 samples per intent)

Models Trained
All models use an 80/20 stratified train/validation split of the training data.
Model	Vectorisation	Val Accuracy	Macro F1
Logistic Regression	TF-IDF (1-2 ngrams)	84.4%	0.84
LinearSVC (default)	TF-IDF (1-2 ngrams)	88.1%	0.88
LinearSVC (tuned, C=1)	TF-IDF (1-2 ngrams)	Best CV 85.5%	—
Simple NN (Embedding + Dense)	Tokenizer + Padding	77.6%	—
LSTM	Tokenizer + Padding	74.0%	—

LinearSVC with TF-IDF is the best performing model and is used as the chatbot backend (best_svm).

Requirements
See requirements.txt for the full dependency list.
Core dependencies:
·	pandas — data loading and manipulation
·	scikit-learn — TF-IDF vectorisation, LinearSVC, Logistic Regression, GridSearchCV, metrics
·	tensorflow / keras — Neural Network and LSTM models
·	matplotlib — EDA and training visualisations
·	numpy — numerical operations

Setup & Running
1. Clone the repository
git clone https://github.com/<your-username>/ds-chatbot.git
cd ds-chatbot

2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

3. Install dependencies
pip install -r requirements.txt

4. Add the data
Place train.csv and test.csv into data/raw/. The dataset can be downloaded from Hugging Face or Kaggle.
5. Run the notebook
jupyter notebook notebooks/01_eda.ipynb

Run all cells in order. The notebook is self-contained and covers:
1.	Data loading and EDA
2.	TF-IDF feature extraction
3.	Logistic Regression baseline
4.	LinearSVC training and hyperparameter tuning
5.	Neural Network (Embedding + Flatten + Dense)
6.	LSTM model
7.	Model comparison
8.	Chatbot demo using the best SVM model

Notebook Versions
The repository preserves iterative development through commit history and checkpoint files:
Version	Description
Initial	Data loading, EDA, class distribution plots
v2	Logistic Regression baseline + TF-IDF pipeline
v3	LinearSVC + confusion matrix + learning curve
v4	GridSearchCV hyperparameter tuning for SVM
v5	Neural Network (Simple NN + LSTM)
v6	Full model comparison chart + chatbot demo


Known Issues / Notes
·	The n_jobs=-1 argument in LogisticRegression raises a FutureWarning in scikit-learn ≥ 1.8 and should be removed.
·	Neural Network and LSTM models are trained for only 5 epochs as a baseline — performance can be improved with more epochs or pretrained embeddings (e.g. DistilBERT).

Possible Next Steps
·	Fine-tune a transformer model (e.g. distilbert-base-uncased) for accuracy above 90%
·	Export the best SVM model with joblib for use in a standalone app
·	Build a simple Flask or Gradio interface around the chatbot
·	Expand the responses dictionary to cover all 77 intents

Author
Rohith — University of Hertfordshire, Data Science
