# Step 1: Import Libraries
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Data
data = pd.read_csv('events.csv')

# Step 3: Basic Data Cleaning
data = data.drop_duplicates()
data['transactionid'] = data['transactionid'].fillna(0)

# Step 4: Define the events order (view, addtocart, transaction)
Events = ['view', 'addtocart', 'transaction']

# Step 5: Encode Event Variable
enc = OrdinalEncoder(categories=[Events])
data['eventencode'] = enc.fit_transform(data[['event']])
Event_Encodings = pd.DataFrame(data['eventencode'], columns=['eventencode'])

# Step 6: Set X variables (features)
X = data[['timestamp', 'itemid', 'visitorid', 'eventencode']]

# Step 7: Standardize numerical features in X
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Step 8: Define Y variable (target)
# Y will be 1 if event is 'transaction', 0 otherwise
data['is_transaction'] = data['event'].apply(lambda x: 1 if x == 'transaction' else 0)
Y = data['is_transaction']

# Step 9: View the final X and Y
print(X_scaled.head())
print(Y.head())

# Step 10: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Step 11: Initialize the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Step 12: Train the Decision Tree on the training data
decision_tree.fit(X_train, Y_train)

# Step 13: Predict on the testing set
Y_pred = decision_tree.predict(X_test)

# Step 14: Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)

# Display results
print(f"Accuracy of Decision Tree: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Step 15: Function to Recommend Item ID Based on Features
def recommend_item(visitor_id, timestamp):
    # Filter data for the visitor and event before the given timestamp
    visitor_data = data[(data['visitorid'] == visitor_id) & (data['timestamp'] <= timestamp)]
    
    if visitor_data.empty:
        print(f"No data available for visitor {visitor_id} before timestamp {timestamp}")
        return
    
    # Ensure X_visitor has the same columns as X_train
    X_visitor = visitor_data[['timestamp', 'itemid', 'visitorid', 'eventencode']]  # Make sure to include 'eventencode'
    
    # Standardize the features of the visitor data
    X_visitor_scaled = pd.DataFrame(scaler.transform(X_visitor), columns=X.columns)
    
    # Predict the probability of a transaction
    transaction_prob = decision_tree.predict_proba(X_visitor_scaled)[:, 1]  # Get the probability for the 'transaction' class

    # Recommend the item with the highest transaction probability
    visitor_data['transaction_prob'] = transaction_prob
    recommended_item = visitor_data.loc[visitor_data['transaction_prob'].idxmax(), 'itemid']
    
    print(f"Recommended Item for Visitor {visitor_id}: {recommended_item}")

# Step 16: Get User Input and Recommend an Item
visitor_id_input = int(input("Enter Visitor ID: "))
timestamp_input = int(input("Enter Timestamp: "))

recommend_item(visitor_id_input, timestamp_input)
