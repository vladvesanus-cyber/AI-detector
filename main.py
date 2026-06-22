import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

data_full = pd.read_csv('Data/data.csv')
data = data_full.sample(n=30000, random_state=42)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])

y = data["generated"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearSVC()
model.fit(X_train, y_train)

pickle.dump({
    "model" : model,
    "vectorizer" : vectorizer
}, open("artifacts.pkl", "wb"))

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(f"Accuracy: {accuracy}")