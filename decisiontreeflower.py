import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
file_path = "path/to/data4_19.csv"  # Replace with the actual file path
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_data = pd.read_csv(file_path, names=column_names)

# Display the first few rows of the dataset
print(iris_data.head())

# Separate features and target variable
X = iris_data.iloc[:, :4]
y = iris_data["species"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")

# Predict the species of a new flower
new_flower_attributes = [5.2, 3.1, 1.4, 0.2]
new_flower_species = clf.predict([new_flower_attributes])[0]
print(f"Predicted species for the new flower: {new_flower_species}")

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=column_names[:-1], class_names=clf.classes_, filled=True, rounded=True)
plt.show()
