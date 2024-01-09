import re


def text_normalization(text):
    text = str(text).lower()  # text to lower case
    spl_char_text = re.sub(r'[^ a-z]', '', text)  # removing special characters
    return spl_char_text
