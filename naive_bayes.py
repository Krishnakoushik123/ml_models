import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
#defining the data with the labels
email=[
    "Hey, are we still meeting for lunch today?",
    "CONGRATULATIONS! You have won a free $1000 Walmart gift card! Click here",
    "Please review the attached project proposal by EOD.",
    "URGENT: Your account has been compromised. Log in immediately to claim reward",
    "Dear friend, I hope this email finds you well.",
    "Get rich quick! Missing out on thousands of dollars weekly! Free entry."
]
label=[0,1,1,0,1,0]
#dividing the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(email,label,test_size=0.33,random_state=42)
#converting the text data into neumarical format vectors
vectorizer=CountVectorizer()
x_train_count=vectorizer.fit_transform(x_train)
x_test_count=vectorizer.transform(x_test)
#defining the multinomialNb model and training with the data
model=MultinomialNB()
model.fit(x_train_count,y_train)
#predicting the output labels 
preds=model.predict(x_test_count)
#evaluating the model with the metrics
accuracy=accuracy_score(y_test,preds)
print(f"accuracy of the model:{accuracy*100:2f}%")
print("classification report")
print(classification_report(y_test,preds,target_names=['ham','spam']))

#evaluating the model performance with sample input
new_email = ["Free cash prizes waiting for you now!"]
new_email_counts = vectorizer.transform(new_email)
prediction = model.predict(new_email_counts)
print(f"sample_email:{new_email[0]}")
print(f"sample:{'spam' if new_email[0]==1 else 'ham'}")