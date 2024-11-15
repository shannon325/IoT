import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models, layers, utils
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
train_data = pd.read_csv('TrainingData.txt', header=None)
X_full = train_data.iloc[:, :-1].values
y_full = train_data.iloc[:, -1].values

# Load the testing data
test_data = pd.read_csv('TestingData.txt', header=None)
X_test = test_data.values
X_test_origin = X_test

# Normalize the features
scaler = StandardScaler()
X_full = scaler.fit_transform(X_full)
X_test = scaler.transform(X_test)

# Split the full data into training and testing sets
X_train, X_test_split, y_train, y_test_split = train_test_split(X_full, y_full, test_size=0.25, random_state=42)

# Further split the training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Build the ANN model
model = models.Sequential()
model.add(layers.Dense(units=64, activation='relu', input_shape=(24,)))  # Hidden layer with 64 neurons
# model.add(layers.Dense(16, activation='relu'))  # Second hidden layer
# model.add(layers.Dense(8, activation='relu'))   # Third hidden layer
model.add(layers.Dense(units=2, activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_split, y_train_split, epochs=300, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test split
y_pred_split = model.predict(X_test_split)
y_pred_split_labels = np.argmax(y_pred_split, axis=1)

# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_test_split, y_pred_split_labels)
precision = precision_score(y_test_split, y_pred_split_labels)
recall = recall_score(y_test_split, y_pred_split_labels)
f1 = f1_score(y_test_split, y_pred_split_labels)
accuracy = accuracy_score(y_test_split, y_pred_split_labels)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.png')
plt.close()

print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Predict on the provided testing data
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Output the predicted labels
print("Predicted Labels for Testing Data:")
print(predicted_labels)

# Save the predicted labels to file
try:
    predicted_results = np.column_stack((X_test_origin, predicted_labels))
    np.savetxt('TestingResults.txt', predicted_results)
    print("Predicted results saved successfully.")
except Exception as e:
    print(f"Error saving predicted results: {e}")