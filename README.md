# Mastering-the-AI-Toolkit
Task 1: Classical ML with Scikit-learn (Iris Dataset)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target).map(dict(enumerate(iris.target_names)))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))


---

Task 2: Deep Learning with TensorFlow (MNIST Dataset)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Visualize predictions
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {predictions[i].argmax()}, True: {y_test[i]}")
    plt.axis('off')
    plt.show()


---

Task 3: NLP with spaCy (Amazon Reviews)

import spacy
nlp = spacy.load("en_core_web_sm")

review = "I love the Sony WH-1000XM4 headphones. The noise cancellation is incredible!"
doc = nlp(review)

print("Named Entities:")
for ent in doc.ents:
    print(ent.text, "-", ent.label_)

positive_words = ['love', 'amazing', 'incredible', 'great', 'fantastic']
negative_words = ['bad', 'terrible', 'hate', 'poor', 'worst']

sentiment = "Positive" if any(word in review.lower() for word in positive_words) else \
            "Negative" if any(word in review.lower() for word in negative_words) else "Neutral"

print("Sentiment:", sentiment)

1. Ethical Considerations

MNIST Model Bias: May underperform on digits written in culturally distinct handwriting.

Amazon Reviews Bias: Reviews may reflect demographic or product bias.

Mitigation:

Use TensorFlow Fairness Indicators to visualize model disparities.

Use spaCy's rule-based patterns to neutralize identity terms or flagged sentiments.



2. Troubleshooting Challenge

Common TensorFlow issues and fixes:

Mismatch in input shape: Ensure .reshape() or input_shape matches the model.

Incorrect loss function: Use sparse_categorical_crossentropy for integer labels, categorical_crossentropy for one-hot labels.

Wrong activation in last layer: Use softmax for multi-class classification.
