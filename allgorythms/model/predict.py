import os

import pickle
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(THIS_DIR, "algo_recognizer.pkl")

with open(MODEL_PKL, "rb") as f:
    model_data = pickle.load(f)

vectorizer = model_data["vectorizer"]
clf_lang = model_data["clf_lang"]
clf_tags = model_data["clf_tags"]
mlb = model_data["mlb"]

code_file = sys.argv[1]
with open(code_file) as f:
    code = f.read()

X = vectorizer.transform([code])

pred_lang = clf_lang.predict(X)[0]
pred_tags_binary = clf_tags.predict(X)
pred_tags = mlb.inverse_transform(pred_tags_binary)[0]

print(f"Language: {pred_lang}")
print(f"Tags: {pred_tags}")
