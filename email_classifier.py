import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load and preprocess data
df = pd.read_csv('mail_data.csv')
data = df.where((pd.notnull(df)), '')  # Replace NaN with empty strings

# Encode spam as 0 and ham as 1
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

x = data['Message']
y = data['Category']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Transform text data to feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# Convert target variables to integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Train the model
model = LogisticRegression()
model.fit(x_train_features, y_train)

# Evaluate the model
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print("Accuracy on training data:", accuracy_on_training_data)

prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print("Accuracy on test data:", accuracy_on_test_data)

# Tkinter for web application
def classify_email():
    input_email = email_input.get("1.0", "end-1c")  # Get the input email text
    if input_email.strip() == "":
        messagebox.showwarning("Input Error", "Please enter an email text!")
        return
    
    # Transform the input email text to feature vectors
    input_features = feature_extraction.transform([input_email])
    
    # Predict using the trained model
    prediction = model.predict(input_features)
    
    if prediction[0] == 1:
        messagebox.showinfo("Result", "Ham mail")
    else:
        messagebox.showinfo("Result", "Spam mail")

# Create the main window
root = tk.Tk()
root.title("Spam/Ham Classifier")

# Create the input text area
email_input_label = tk.Label(root, text="Enter your email text:")
email_input_label.pack()

email_input = tk.Text(root, height=10, width=50)
email_input.pack()

# Create the classify button
classify_button = tk.Button(root, text="Classify", command=classify_email)
classify_button.pack()

# Start the GUI loop
root.mainloop()