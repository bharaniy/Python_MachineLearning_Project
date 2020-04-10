from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
import string
from nltk import pos_tag

import nltk

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

df = pd.read_csv("/content/drive/My Drive/fake_job_postings.csv")
df.fillna(" ", inplace=True)

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df[
    'description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df[
                 'required_education'] + ' ' + df['industry'] + ' ' + df['function']
print(df.text)

lemmatizer = WordNetLemmatizer()

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def lemmatize_words(words):
    final_text = []
    for t in df.text[:50]:
        word = lemmatizer.lemmatize(t, pos="n")
        final_text.append(words.lower())
    return " ".join(final_text)


df.text = df.text.apply(lemmatize_words)

train_text, test_text, train_category, test_category = train_test_split(df.text, df.fraudulent, test_size=0.2,
                                                                        random_state=0)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(df['text'].apply(lambda x: np.str_(x)))
clf = KNeighborsClassifier()
clf.fit(X_train_tfidf, df.fraudulent)

X_test_tfidf = tfidf_Vect.transform(df['fraudulent'].apply(lambda x: np.str_(x)))

predicted = clf.predict(X_test_tfidf)

scoreAccuracy = metrics.accuracy_score(df['fraudulent'], predicted)
print(scoreAccuracy)