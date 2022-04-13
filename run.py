import nltk
import pandas as pd
import re
import string

from keras import layers
from keras import losses
from keras import metrics
from keras import models
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('english')


def filter_for_actual_words(word_list):
    return [word for word in word_list if re.search(r'^[a-zA-Z\d]+$', re.sub(r'^[#@]', '', word))]
    # return [word for word in word_list if not re.search(r'^(?!rt)\w{1,2}$', word, flags=re.IGNORECASE)]

def stem_words(word_list):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    return [stemmer.stem(word) for word in word_list]

def find_useful_words(input):
    words = re.sub(r'(?<=\d),(?=\d)', '', input)
    words = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', words)
    word_list = word_tokenize(words)
    word_list = filter_for_actual_words(word_list)
    stemmed_words = stem_words(word_list)
    return ' '.join([word for word in stemmed_words if word not in stop_words])

if __name__ == '__main__':
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    train_df['final_text'] = train_df['text'].apply(find_useful_words)
    test_df['final_text'] = test_df['text'].apply(find_useful_words)
    print(train_df.tail())

    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(train_df['final_text'])
    print(vectorizer.get_feature_names_out())

    x_test = vectorizer.transform(test_df['final_text'])
    print(vectorizer.get_feature_names_out())

    x_train, x_val, y_train, y_val = train_test_split(x_train.toarray(), train_df['target'], test_size=0.2, random_state=None)
    input_shape = x_train.shape[1]
    print(f"x_train.shape: {input_shape}")

    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.15),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    print(model.summary())

    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

    history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=16,
                    validation_data=(x_val, y_val))
