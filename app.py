# Import Libraries
from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Flask app initialization
app = Flask(__name__)

# 1: Load Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Check for basic info and missing values
print(df.info())

# Preprocessing
x = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
x_ref = scaler.fit_transform(x)

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x_ref, y, test_size=0.15, random_state=42)

# 2: Decision tree classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Training
model.fit(x_train, y_train)

# 3: Prediction
y_pred = model.predict(x_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'ACCURACY OF THE DECISION TREE MODEL : {accuracy * 100:.2f}%')

# Logistic Regression
model_log = LogisticRegression(max_iter=200, random_state=42)
model_log.fit(x_train, y_train)
pred_log = model_log.predict(x_test)
accuracy_log = accuracy_score(y_test, pred_log) * 100
print("ACCURACY OF LOGISTIC REGRESSION MODEL: {:.2f}%".format(accuracy_log))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)

#confusion matrix regression
cm_log = confusion_matrix(y_test, pred_log)

#data
models = ['Decision Tree', 'Logistic Regression']
values = [accuracy, accuracy_log]

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Visualization and result route
@app.route('/predict', methods=['POST'])
def predict():
    # Randomly select a sample from the test set
    random_index = np.random.randint(0, x_test.shape[0])
    input_data = x_test[random_index].reshape(1, -1)
    true_class = y_test.iloc[random_index]

    # Predictions
    pred = model.predict(input_data)[0]
    pred_log = model_log.predict(input_data)[0]

    # Create bar plot
    plt.figure(figsize=(7, 5))
    plt.bar(models, values, color=['red', 'brown'])
    plt.title("Comparison of Classifier Accuracies")
    plt.ylabel("Accuracy")
    plt.ylim(0, 110)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)

    # Convert bar plot to PNG and encode it
    bar_img = io.BytesIO()
    plt.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_plot_url = base64.b64encode(bar_img.getvalue()).decode()

    # 4: Visualization (Confusion Matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Convert confusion matrix to PNG and encode it
    cm_img = io.BytesIO()
    plt.savefig(cm_img, format='png')
    cm_img.seek(0)
    cm_plot_url = base64.b64encode(cm_img.getvalue()).decode()

    # 4: Visualization (Confusion Matrix)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Convert confusion matrix to PNG and encode it
    cm_log_img = io.BytesIO()
    plt.savefig(cm_log_img, format='png')
    cm_log_img.seek(0)
    cm_log_plot_url = base64.b64encode(cm_log_img.getvalue()).decode()

    return render_template(
        'prediction.html',
        input_data=input_data[0],
        true_class=true_class,
        pred=pred,
        pred_log=pred_log,
        accuracy=accuracy,
        accuracy_log=accuracy_log,
        bar_plot_url=bar_plot_url,
        cm_plot_url=cm_plot_url,
        cm_log_plot_url=cm_log_plot_url
    )

if __name__ == '__main__':
    app.run(debug=True)