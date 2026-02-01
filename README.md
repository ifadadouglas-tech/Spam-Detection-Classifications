# Spam-Detection-Classifications
This project is accurate and special

My Spam Mail Project with streamlit deployment.

Topic: 
The Protector: How AI Fights Spam & Junk Mail.
''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#import pickle

#data = pd.read_csv("spam email project.csv", encoding= 'ISO-8859-1', encoding_errors = 'strict')

#data.head()

#data.shape

#data.info()

#data.tail()

#data['v1'].value_counts()

#> * We drop the last three columns as they are unnecessary and have a lot of null values

#data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
#data.head()

#> * Encoding ham and spam using label encoder
#encoder = LabelEncoder()
#data['v1'] = encoder.fit_transform(data['v1'])
#data.head()

#> * Checking and removing duplicates
#data.duplicated().sum()

# Drop the duplicated
#data = data.drop_duplicates(keep='first')
#data.duplicated().sum()
#> * Finding the total number of words using tokenization (nltk)
# nltk.download('punkt')
# data['words'] = data['v2'].apply(lambda x:len(nltk.word_tokenize(x)))
# data.head()
# data[data['v1']==0][['words']].describe()

#data[data['v1']==1][['words']].describe()

# Checking its distributions 
# plt.figure(figsize=(15,8))
#sns.histplot(data[data['v1'] == 0]['words'])
#sns.histplot(data[data['v1'] == 1]['words'],color='red')

## We can now see that Spam has usually more numbers of words even though Ham has a larger distributions

#So we need to do the following:

#1. Lower Case Transformation.
#2. Tokenization.
#3. Removal of stop words.
#4. Punctuation and special characters removal.
#5. Stemming 

#ps = PorterStemmer()

#def transform_text(text):
    #text = text.lower()
    #text = nltk.word_tokenize(text)

    #y = []
    #for i in text:
       # if i.isalnum():
          #  y.append(i)

    #text = y[:]
    #y.clear()

    #for i in text:
       # if i not in stopwords.words('english') and i not in string.punctuation:
            #y.append(i)

    #text = y[:]
    #y.clear()

    #for i in text:
       # y.append(ps.stem(i))


    #return " ".join(y)

    ## Using the Function Above we can Transform The Test

    #data['after_transformation'] = data['v2'].apply(transform_text)
    #data.head()

    # Showing WordCloud of Ham and Spam Data
    
#wc = WordCloud(min_font_size=8)
#spam_wc = wc.generate(data[data['v1']==1]['after_transformation'].str.cat(sep=" "))
#plt.figure(figsize=(15, 8))
#plt.imshow(spam_wc)

#wc = WordCloud(min_font_size=8)
#am_wc = wc.generate(data[data['v1']==0]['after_transformation'].str.cat(sep=" "))
#plt.figure(figsize=(15, 8))
#plt.imshow(ham_wc)

# # Building A Model
# Importing the neccessary librabies.

#from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#cv = CountVectorizer()
#tfidf = TfidfVectorizer(max_features=3000)

# Splitting Dataset into X and Y

#X = tfidf.fit_transform(data['after_transformation']).toarray()
#X

#y = data['v1'].values
#y

## Splitting the Data into train and Test Data

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Accuracy Report and Evaluation Method

#def acc_report(actual,predicted):
    #acc_score=accuracy_score(actual,predicted)
    #cm_matrix=confusion_matrix(actual,predicted)
    #class_rep=classification_report(actual,predicted)
    #print('the accuracy of tha model is ',acc_score)
    #print(cm_matrix)
    #print(class_rep)


## Applying different  Machine Learning classification Models

#mnb = MultinomialNB()
#dtree = DecisionTreeClassifier(max_depth=5)
#ada = AdaBoostClassifier(n_estimators=50, random_state=2)
#rf = RandomForestClassifier(n_estimators=50, random_state=2)

## Fitting the Model to the train data

mnb.fit(X_train, y_train)
dtree.fit(X_train, y_train)
ada.fit(X_train, y_train)
rf.fit(X_train, y_train)

## Predictions

mnbtrain = mnb.predict(X_train)
mnbtest = mnb.predict(X_test)

dtreetrain = dtree.predict(X_train)
dtreetest = dtree.predict(X_test)

adatrain = ada.predict(X_train)
adatest = ada.predict(X_test)

rftrain = rf.predict(X_train)
rftest = rf.predict(X_test)

# #I DECIDED TO COMPARE THE ACCURACIES OF ALL THE MODELS.

# MultinomialNB
#print(acc_report(y_train, mnbtrain))
#print(acc_report(y_test, mnbtest))

##I GOT THESE RESULTS
## For MultinomialNB
##1. Train Accuracy_score = 97.87%.
##2. Test Accuracy_score = 97.29%
'''''''''''''''''''''''''''''''''''''''
# DecisionTreeClassifier
#print(acc_report(y_train, dtreetrain))
#print(acc_report(y_test, dtreetest))

## For DecisionTreeClassifier
#1. Train Accuracy_score = 94.93%.
#2. Test Accuracy_score = 93.32%
''''''''''''''''''''''''''''''''''
# AdaBoostClassifier
#print(acc_report(y_train, adatrain))
#print(acc_report(y_test, adatest))

## For AdaBoostClassifier

#1. Train Accuracy_score = 97.19%.
#2. Test Accuracy_score = 96.42%
'''''''''''''''''''''''''''''''''''
#RandomForest

#print(acc_report(y_train, rftrain))
#print(acc_report(y_test, rftest))
# For RandomForestClassifier

#1. Train Accuracy_score = 99.97%.
#2. Test Accuracy_score = 97.19%

## In Conclusions,i decided to choose MultinomialNB as our model as it has a higher test accuracy score, a low bias and low varience.

## Importing and saving the Model and the vectorizer

#import pickle
#pickle.dump(tfidf,open('e_vectorizer.pkl','wb'))
#pickle.dump(mnb,open('e_model.pkl','wb'))
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Streamlit.


import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from PIL import Image
import joblib
from io import BytesIO
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import warnings




def main():



    tfidf = pickle.load(open("C:\\Users\DOUGLAS IFADA\Desktop\My Data Science Files\e_vectorizer.pkl", 'rb'))
    model = pickle.load(open(r"C:\\Users\DOUGLAS IFADA\Desktop\My Data Science Files\e_model.pkl", 'rb'))

     ## set page configuration
    st.set_page_config(page_title = "SPAM EMAIL CLASSIFIER", layout = 'centered')

    
    image = Image.open(" ")
    st.write(image, use_column_width=None)

   


    input_text = st.text_area("Enter Email Content Here for Classification:", height=200)

    
    with st.sidebar:
        st.header("DASHBOARD")
        st.write("ℹ️ About This App")
    
        st.markdown("""
        This is a **Spam and Junk Detector and Prevention Web App** built using **Streamlit** and a ** MultinomialNB Naive Bayes Classifier** and **RandomForest Classifier**.

        The model is trained on the **spam project emails Dataset** 

        ---
        ### 🧠 Model Used:
        - MultinomialNB Naive Bayes Classifier
        - Accuracy: ~97%

        ---
        Built by: **Douglas Aseharianegbe Ifada**
                
        Year Built: ** 2025**
        """)
    


    ps = PorterStemmer()

    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))


        return " ".join(y)

    if st.button('Clasify your  Email'):
        if input_text:
            transformed_input = transform_text(input_text)
            vectorizedd = tfidf.transform([transformed_input])
            prediction = model.predict(vectorizedd)[0]

            if prediction ==1:   # Assuming 1 for spam, 0 for ham
                st.error("WRANING  ALERT! This email is likely classified as SPAM!")
            else:
                st.success("This email is likely Not Spam.")


            st.info("Classification logic placeholder. Replace with actual model prediction.")
        else:
            st.warning("You haven't entered your email yet! Please enter Your  email content to classify.")




    st.subheader("Test if your email is Spam or not")
    st.balloons()


    st.title("Spam and Nonspam Detector Web App")
    st.subheader("Predict whether you are likely to have Spam Mail or Not.")
    st.markdown("Enter the Email information to get an instant predictions.")

            
if __name__ == '__main__':
    main()
