import os
import pickle
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from numpy import nan
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from utils.context_loader import Loader
from utils.text_normalizer import text_normalization

CLASSIFIERS = {
    # Canonical classifiers
    "Logistic Regression": LogisticRegression(C=1, max_iter=10, n_jobs=1),
    "Ridge Classifier": RidgeClassifier(alpha=1.0, solver="sparse_cg"),
    "kNN": KNeighborsClassifier(2),
    # "Random Forest": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=600,
        subsample=1.0,
        criterion="friedman_mse",
        max_depth=9,
        min_impurity_decrease=0.0,
        init='zero',
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=3,
        tol=1e-4,
        ccp_alpha=0.0,
    ),

    # L2 penalty Linear SVC
    "Linear SVC": LinearSVC(C=0.1, dual=False, max_iter=1000, tol=10**(-2)),

    # L2 penalty Linear SGD
    "log-loss SGD": SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-3, random_state=42,
                                  max_iter=1000, tol=None),

    # NearestCentroid (aka Rocchio classifier)
    "NearestCentroid": NearestCentroid(),

    # Sparse naive Bayes classifier
    "Complement naive Bayes": ComplementNB(alpha=0.1),
}


BASE_DIR = Path(__file__).resolve().parent
data_agent_path = os.path.join(BASE_DIR, "data_agent.pickle")
model_path = os.path.join(BASE_DIR, "model.pickle")

train_df = pd.read_excel("dialog_data/dialog_talk_agent.xlsx")
train_df = train_df.replace(nan, None)

test_df = pd.read_excel("dialog_data/test_data.xlsx")
test_df = test_df.replace(nan, None)

count_vect = CountVectorizer()
tfidf = TfidfTransformer()


def prepare_data(data_df, desc_text):
    data = {}
    examples = []
    responses = []
    cat_header_flag = False
    cat_header = None

    for index, line in tqdm(data_df.iterrows(), desc=desc_text.ljust(30), total=data_df.shape[0]):
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
    for key in tqdm(prepared_data.keys(), desc='Making vectors'.ljust(30), total=len(prepared_data.keys())):
        X_vector_str += [*prepared_data[key]['examples']]
        y_vector_str += [key for _ in prepared_data[key]['examples']]

    return prepared_data, X_vector_str, y_vector_str


def model_training(clf_name, clf):
    model_classifier = Pipeline([
        ('vectorizer', count_vect),
        ('tfidf', tfidf),
        ('classifier', clf),
    ])

    start = time.time()
    with Loader(desc=f"Model with base {clf_name} is training ",
                end=f"Model with base {clf_name} was trained"):
        model_classifier.fit(train_X, train_y)
    stop = time.time()
    print(f"training time: {timedelta(seconds=(stop - start))}")

    return model_classifier


data, train_X, train_y = vector_normalization(train_df, 'Reading train DF')
test_data, test_X, test_y = vector_normalization(test_df, 'Reading test DF')


accuracy_dict = {}
f1_dict = {}
for classifier_name, classifier in CLASSIFIERS.items():
    text_clf = model_training(classifier_name, classifier)

    predicted = text_clf.predict(test_X)
    accuracy = round(accuracy_score(test_y, predicted) * 100, 2)
    accuracy_dict.update({classifier_name: (text_clf, accuracy)})
    f1 = round(f1_score(test_y, predicted, average='weighted') * 100, 2)
    f1_dict.update({classifier_name: f1})
    print(f"{classifier_name} accuracy score: {accuracy}%")
    print(f"{classifier_name} F1 score: {f1}%")
    print("================================================")


best_accuracy = max(accuracy_dict, key=lambda x: accuracy_dict[x][1])
best_f1 = max(f1_dict, key=lambda x: f1_dict[x])
print(f"The Best accuracy : {best_accuracy} - {accuracy_dict[best_accuracy][1]}%")
print(f"The Best accuracy : {best_f1} - {f1_dict[best_f1]}%")

pickle.dump(data, open(data_agent_path, "wb"))
pickle.dump(accuracy_dict[best_accuracy][0], open(model_path, 'wb'))
