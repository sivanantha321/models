import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report

# Load the 20 Newsgroups dataset (using all data, and removing headers/footers for cleaner text)
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a text processing pipeline:
# 1. TfidfVectorizer: converts raw text into TF-IDF features, ignoring common English stop words.
# 2. TruncatedSVD: reduces dimensionality (also known as LSA) to capture the most important topics.
tfidf = TfidfVectorizer(stop_words='english', max_df=0.5)
svd = TruncatedSVD(n_components=100, random_state=42)

# Define base classifiers
lr = LogisticRegression(max_iter=1000, random_state=42)
svc = LinearSVC(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Build a stacking classifier that ensembles the base classifiers.
# The meta-classifier is also a Logistic Regression model.
estimators = [
    ('lr', lr),
    ('svc', svc),
    ('rf', rf)
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)

# Create the full pipeline by chaining the steps
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('svd', svd),
    ('stack', stacking_clf)
])

# Optionally, set up a grid search to tune hyperparameters for better performance.
param_grid = {
    'tfidf__max_df': [0.5, 0.75],
    'svd__n_components': [100, 150],
    'stack__final_estimator__C': [0.1, 1.0, 10],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("Best parameters found:")
print(grid.best_params_)

# Evaluate the best model on the test set
y_pred = grid.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the best model to disk
joblib.dump(grid.best_estimator_, 'news_paper_model.joblib')