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

from context_loader import Loader
from text_normalizer import text_normalization


BASE_DIR = Path(__file__).resolve().parent
data_agent_path = os.path.join(BASE_DIR, "data_agent.pickle")
model_path = os.path.join(BASE_DIR, "model.pickle")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pickle")

train_df = pd.read_excel("dialog_talk_agent.xlsx")
train_df = train_df.replace(nan, None)

data = {}
examples = []
responses = []
cat_header_flag = False
cat_header = None

for index, line in tqdm(train_df.iterrows(), desc=f'Reading DF    ', total=train_df.shape[0]):
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

vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore')
examples_X_vector_str = []
examples_y_vector_str = []
for key in tqdm(data.keys(), desc='Making vectors', total=len(data.keys())):
    examples_X_vector_str += [*data[key]['examples']]
    examples_y_vector_str += [key for ex in data[key]['examples']]

tfidf = vectorizer.fit(examples_X_vector_str)
vec_train_data = vectorizer.fit_transform(examples_X_vector_str)

model = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=600,
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
start = time.time()
with Loader(desc="Model is training ", end="Model was trained"):
    model.fit(vec_train_data, examples_y_vector_str)
stop = time.time()
predicted = model.predict(vec_train_data)
print(f"training time: {timedelta(seconds=(stop - start))}")
print(f"accuracy score: {round(accuracy_score(examples_y_vector_str, predicted) * 100, 2)}%")
print(f"F1 score: {round(f1_score(examples_y_vector_str, predicted, average='weighted') * 100, 2)}%")

pickle.dump(data, open(data_agent_path, "wb"))
pickle.dump(tfidf, open(vectorizer_path, "wb"))
pickle.dump(model, open(model_path, 'wb'))
