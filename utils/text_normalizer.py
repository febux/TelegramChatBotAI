import re

from nltk import pos_tag, data, download, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from conf.constants import BASE_DIR

lemmatizer = WordNetLemmatizer()


data.path.append(f'{BASE_DIR}\\nltk_package\\')

download('popular', download_dir="../nltk_package/")
download('stopwords', download_dir="../nltk_package/")
download('wordnet', download_dir="../nltk_package/")
download('averaged_perceptron_tagger', download_dir="../nltk_package/")


STOP_WORDS = set(stopwords.words('russian'))


def text_normalization(text):
    text = str(text).lower()  # text to lower case
    formatted_text = re.sub(r'[^ a-z]', '', text)  # removing special characters
    tokenized_text = word_tokenize(formatted_text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in tokenized_text if not word.lower() in STOP_WORDS]
    tags_list = pos_tag(lemmatized_text)  # parts of speech
    lema_words = []  # empty list
    for token, pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val = 'v'
        elif pos_token.startswith('J'):  # Adjective
            pos_val = 'a'
        elif pos_token.startswith('R'):  # Adverb
            pos_val = 'r'
        else:
            pos_val = 'n'  # Noun
        lema_token = lemmatizer.lemmatize(token, pos_val)  # performing lemmatization
        lema_words.append(lema_token)  # appending the lemmatized token into a list

    return " ".join(lema_words)  # returns the lemmatized tokens as a sentence
