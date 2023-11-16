from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import _tree  # Add this line

def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    stack = [(0, -1)]  # (node, parent_depth)
    rules = []

    while stack:
        node, parent_depth = stack.pop()

        # Add a rule
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            condition = f"{name} <= {threshold}"
            rule = f"if {condition}: "
        else:
            value = np.argmax(tree_.value[node])
            rule = f"return '{tree.classes_[value]}',"

        # Add the rule to the list
        rules.append("  " * parent_depth + rule)

        # Add children to the stack
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            stack.append((tree_.children_right[node], parent_depth + 1))
            stack.append((tree_.children_left[node], parent_depth + 1))

    return rules

# Your data
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'Play Golf': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])

# Create and fit the decision tree model
X = df.drop('Play Golf', axis=1)
y = df['Play Golf']

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Print the decision tree rules
tree_rules = tree_to_rules(clf, X.columns)
for rule in tree_rules:
    print(rule)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True)
plt.show()
