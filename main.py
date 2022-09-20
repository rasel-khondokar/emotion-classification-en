import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

from preprocessing import PreProcessor, EnglishPreprocessor

DIR_RESOURCES = 'RESOURCES'


dataset = pd.read_csv('DATASET/emotion_dataset_2-1.csv')
dataset.dropna(inplace=True, axis=0)
dataset = dataset.sample(100)
english_preprocessor = EnglishPreprocessor()
dataset['Text'] = english_preprocessor.preprocess(dataset['Text'])

preprocessor = PreProcessor()
name = 'emotion'
corpus = preprocessor.vectorize_tfidf(dataset.Text, (1, 1), name)
labels, class_names = preprocessor.encode_category(dataset.Emotion)
X_train, X_valid, y_train, y_valid = train_test_split(corpus, labels,
                                                      test_size=0.2, random_state=0)
# Fit into the models
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, config_dict="TPOT sparse")
tpot.fit(X_train, y_train)
print(tpot.score(X_valid, y_valid))
tpot.export('tpot_best_pipeline.py')