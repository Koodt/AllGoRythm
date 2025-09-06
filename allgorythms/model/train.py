import os
import json
import pickle
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(THIS_DIR, "../datasets"))
DATASET_JSON = os.path.join(DATASET_DIR, "train.json")
MODEL_PKL = os.path.join(THIS_DIR, "algo_recognizer.pkl")
LANGUAGES = ["go", "cpp", "python", "unknown"]


def fit_with_dynamic_iter(
    clf,
    X,
    y,
    classes=None,
    initial_iter=1000,
    max_total_iter=5000,
):
    iter_step = initial_iter
    total_iter = 0
    while True:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                if hasattr(clf, "partial_fit") and classes is not None:
                    clf.partial_fit(X, y, classes=classes)
                else:
                    clf.fit(X, y)
            break
        except ConvergenceWarning:
            total_iter += iter_step
            if total_iter >= max_total_iter:
                print(
                    f"Warning: model did not converge after {total_iter} "
                    "iterations"
                )
                break
            print(
                "ConvergenceWarning caught, increasing max_iter to "
                f"{total_iter + iter_step}"
            )
            clf.max_iter = total_iter + iter_step


with open(DATASET_JSON) as f:
    data = json.load(f)

codes = []
y_lang = []
y_tags = []

for d in data:
    code = d["code"]
    tags = d["tags"]

    lang = next((t.lower() for t in tags if t.lower() in LANGUAGES), "unknown")
    other_tags = [t for t in tags if t not in LANGUAGES]

    codes.append(code)
    y_lang.append(lang)
    y_tags.append(other_tags)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(codes)

mlb = MultiLabelBinarizer()
Y_tags = mlb.fit_transform(y_tags)

if os.path.exists(MODEL_PKL):
    with open(MODEL_PKL, "rb") as f:
        model_data = pickle.load(f)
    clf_lang = model_data["clf_lang"]
    clf_tags = model_data["clf_tags"]
    mlb_existing = model_data["mlb"]
    mlb = mlb_existing
else:
    clf_lang = SGDClassifier(loss="log_loss", max_iter=1000)
    clf_tags = OneVsRestClassifier(
        SGDClassifier(loss="log_loss", max_iter=1000),
    )

clf_lang.partial_fit(X, y_lang, classes=LANGUAGES)

clf_tags.fit(X, Y_tags)

with open(MODEL_PKL, "wb") as f:
    pickle.dump(
        {
            "vectorizer": vectorizer,
            "clf_lang": clf_lang,
            "clf_tags": clf_tags,
            "mlb": mlb,
        },
        f,
    )

print(f"Model trained on {len(codes)} examples. Saved to {MODEL_PKL}")
