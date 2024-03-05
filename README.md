# Tweet-Text-Classifier
Tweet Classification Project
Overview
This project focuses on classifying tweets as real or fake based on provided labels. Various machine learning algorithms were employed along with extensive preprocessing techniques to achieve the classification task. The project involved the following steps:

Data Preprocessing:

NLTK tweet tokenizer was used to tokenize emojis and URLs.
Emojis were converted to text using the Emoji module.
Punctuations and stopwords were removed.
Porter stemming was applied to the text.
TF-IDF vectorizer was utilized for feature extraction.
Modeling and Hyperparameter Tuning:

Several machine learning algorithms were explored:
Support Vector Machine (SVM)
Logistic Regression
K-Nearest Neighbors (KNN)
MLPClassifier (Neural Network)
KMeans
FastText
Hyperparameters were tuned using techniques like GridSearchCV and brute force.
Detailed hyperparameters tuning and best attributes obtained for each algorithm are provided in the project documentation.
Results:

Performance metrics such as precision, recall, f1-score, and accuracy were evaluated for each model.
Confusion matrices were generated to understand the model's behavior.
The best performing models were identified based on the evaluation metrics.
Usage
To utilize this project, follow these steps:

Clone the repository to your local machine:

git clone <project_repository_url.git>

Install the required dependencies:
Modules required -
pandas
numpy
matplotlib
nltk
preprocessor
ttp
emoji
pickle
sklearn
fasttext

Execute by running 'make'

Execute the provided Jupyter Notebook or Python scripts to preprocess the data, train the models, and evaluate their performance.

Optionally, fine-tune hyperparameters or experiment with different algorithms based on your specific requirements.

Project Structure
data/: Contains the dataset used for training and evaluation.
notebooks/: Contains Jupyter Notebooks demonstrating the data preprocessing, model training, and evaluation.
src/: Contains Python scripts for data preprocessing, model training, and evaluation.
README.md: This file providing an overview of the project.
requirements.txt: Lists all the dependencies required to run the project.
Contributors
Akash Bankar (23CS60R65)

Make sure you have all the packages installed.


