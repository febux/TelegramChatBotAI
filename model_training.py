import os
import pickle
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from numpy import nan
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from utils.context_loader import Loader
from text_normalizer import text_normalization


BASE_DIR = Path(__file__).resolve().parent
data_agent_path = os.path.join(BASE_DIR, "data_agent.pickle")
model_path = os.path.join(BASE_DIR, "model.pickle")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pickle")

train_df = pd.read_excel("dialog_talk_agent.xlsx")
train_df = train_df.replace(nan, None)

test_df = pd.read_excel("test_data.xlsx")
test_df = test_df.replace(nan, None)

vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore')

model = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=300,
    subsample=1.0,
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=9,
    min_impurity_decrease=0.0,
    init='zero',
    random_state=None,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=1e-4,
    ccp_alpha=0.0,
)


def prepare_data(data_df, desc_text):
    data = {}
    examples = []
    responses = []
    cat_header_flag = False
    cat_header = None

    for index, line in tqdm(data_df.iterrows(), desc=desc_text, total=train_df.shape[0]):
        if cat_header_flag:
            cat_header = line['Context']
            data[line['Context']] = {'examples': examples, 'responses': responses}
            cat_header_flag = False

        if line['Context'] == ':::' and line['Text Response'] == ':::':
            examples = []
            responses = []
            cat_header_flag = True
            cat_header = None
        else:
            if line['Text Response']:
                data[cat_header]['responses'].append(line['Text Response'])

            if line['Context']:
                data[cat_header]['examples'].append(text_normalization(line['Context']))
    return data


def vector_normalization(df, desc_text):
    X_vector_str = []
    y_vector_str = []
    prepared_data = prepare_data(df, desc_text)
    for key in tqdm(prepared_data.keys(), desc='Making vectors', total=len(prepared_data.keys())):
        X_vector_str += [*prepared_data[key]['examples']]
        y_vector_str += [key for ex in prepared_data[key]['examples']]

    return prepared_data, X_vector_str, y_vector_str


data, train_X, train_y = vector_normalization(train_df, 'Reading train DF')

tfidf = vectorizer.fit(train_X)
vec_train_data = vectorizer.fit_transform(train_X)

start = time.time()
with Loader(desc="Model is training ", end="Model was trained"):
    model.fit(vec_train_data, train_y)
stop = time.time()
print(f"training time: {timedelta(seconds=(stop - start))}")

test_data, test_X, test_y = vector_normalization(train_df, 'Reading test DF')

vec_test_data = vectorizer.fit_transform(test_X)

predicted = model.predict(vec_test_data)

print(f"accuracy score: {round(accuracy_score(test_y, predicted) * 100, 2)}%")
print(f"F1 score: {round(f1_score(test_y, predicted, average='weighted') * 100, 2)}%")

pickle.dump(data, open(data_agent_path, "wb"))
pickle.dump(tfidf, open(vectorizer_path, "wb"))
pickle.dump(model, open(model_path, 'wb'))
