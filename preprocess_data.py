import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import re

def read_data(data_path, data_encoding="latin-1"):
    
    spam_data = pd.read_csv(data_path, encoding=data_encoding)

    return spam_data


def preprocess_data(dataframe, target_cols = ['v1', 'v2']):

    spam_data = dataframe[target_cols]

    spam_data.rename(columns = {'v1': 'target', 
                            'v2': 'text'}, inplace = True)

    spam_data['target'] = spam_data['target'].map({'ham': 0, 
                                           'spam': 1})

    return spam_data


def clean_data(dataframe, max_len=2):

    # 1. Remove punctuations
    dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # 2. Convert to lowercase
    dataframe['text'] = dataframe['text'].apply(lambda x: x.lower())

    # 3. Remove stopwords
    stop_words = set(stopwords.words('english'))

    dataframe['text'] = dataframe['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # 4. Remove numbers
    dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'\d+', '', x))

    # 5. Remove words less than 2 letters
    dataframe['text'] = dataframe['text'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > max_len]))

    return dataframe

if __name__ == "__main__":

    DATA_PATH = "./data/"
    DATA_FILE = "spam.csv"

    FINAL_FILE = "cleaned_data.csv"

    # 1. Read data
    spam_data = read_data(DATA_PATH+DATA_FILE)

    # 2. Preprocess the data
    preprocessed_data = preprocess_data(spam_data)

    # 3. Clean the data
    cleaned_data = clean_data(preprocessed_data)

    # 4. Save the cleaned data for training
    cleaned_data.to_csv(DATA_PATH+FINAL_FILE, index=False)