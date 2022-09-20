import pickle
import numpy as np
import dill
import re
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from spacy.lang.en import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

DIR_RESOURCES = 'RESOURCES'
nlp = spacy.load('en_core_web_sm')
CONSTRACTIONS = {
        "ain't": "am not / are not / is not / has not / have not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is / how does",
        "i'd": "i had / i would",
        "i'd've": "i would have",
        "i'll": "i shall / i will",
        "i'll've": "i shall have / i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
    }

class EnglishPreprocessor():

    def remove_stopwords(self, x_data):
        stopWords = STOP_WORDS
        return x_data.apply(
            lambda x: ' '.join([word for word in x.split() if word not in stopWords]))

    def remove_html(self, x_data):
        return x_data.apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

    def remove_urls(self, x_data):
        return x_data.apply(
            lambda x: re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&:/~+#-])?',
                             "",
                             str(x)))

    def con_to_expn(self, x_data, contractions):
        if type(x_data) is str:
            for key in contractions:
                value = contractions[key]
                x_data = x_data.replace(key, value)
            return x_data
        else:
            return x_data

    def remove_condtructions(self, x_data):
        return x_data.apply(lambda x: self.con_to_expn(x, CONSTRACTIONS))

    def remove_email(self, x_data):
        return x_data.apply(
            lambda x: re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9+._-]+\.[a-zA-Z0-9+_-]+)', '', x))

    def remove_space(self, x_data):
        return x_data.apply(lambda x: " ".join(x.split()))

    def remove_punctuations(self, x_data):
        x_data = x_data.apply(lambda x: re.sub('-', ' ', x))
        return x_data.apply(lambda x: re.sub('[^A-Z a-z 0-9]+', '', x))

    def to_lower(self, x_data):
        return x_data.apply(lambda x: x.lower())

    # Convert to base form
    def make_base_word(self, x):
        x_list = []
        doc = nlp(x)

        for token in doc:
            lemma = str(token.lemma_)
            if lemma == '-PRON-' or lemma == 'be':
                lemma = token.text
            x_list.append(lemma)
        return ' '.join(x_list)

    def to_lemma(self, x_data):
        return x_data.apply(lambda x: self.make_base_word(x))

    def convert_to_dataframe(self, x_data):
        if isinstance(x_data, list):
            x_data = pd.DataFrame(x_data)
        return x_data

    def preprocess(self, x_data):
        # Remove HTML
        x_data = self.remove_html(x_data)
        # Remove URLS
        x_data = self.remove_urls(x_data)
        # Remove Emails
        x_data = self.remove_email(x_data)
        # Remove Contruction
        x_data = self.remove_condtructions(x_data)
        # Remove Multiple Space
        x_data = self.remove_space(x_data)
        # Remove Punctuation and Special Character
        x_data = self.remove_punctuations(x_data)
        # REMOVE Stopwords
        x_data = self.remove_stopwords(x_data)
        # Lemmatiazation
        x_data = self.to_lemma(x_data)
        # To_lower
        x_data = self.to_lower(x_data)

        return x_data

class PreProcessor:

    def vectorize_tfidf(self, article, gram, name):
        tfidf = TfidfVectorizer(ngram_range=gram, use_idf=True, tokenizer=lambda x: x.split())
        x = tfidf.fit_transform(article)
        # save the label encoder into a pickle file
        # with open(DIR_RESOURCES + '/label_encoder.pickle', 'wb') as handle:
        with open(f'{DIR_RESOURCES}/{name}_tfidf_encoder.pickle', 'wb') as handle:
            dill.dump(tfidf, handle)
        return x

    def encode_category(self, category_col, is_test=False, name=''):
        if is_test:
            with open(DIR_RESOURCES+f'/{name}label_encoder.pickle', 'rb') as handle:
                le = pickle.load(handle)
            encoded_labels = le.transform(category_col)
            labels = np.array(encoded_labels)
            class_names = le.classes_
        else:
            le = LabelEncoder()
            le.fit(category_col)
            encoded_labels = le.transform(category_col)
            labels = np.array(encoded_labels)
            class_names = le.classes_

            # save the label encoder into a pickle file
            with open(DIR_RESOURCES + f'/{name}label_encoder.pickle', 'wb') as handle:
                pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return labels, class_names
