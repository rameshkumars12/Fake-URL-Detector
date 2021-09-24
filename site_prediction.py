import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


#Custom Vectorizer
def customtkns(t):
    tkns_byslash = str(t.encode("utf-8")).split("/")
    total_tokens = []
    for i in tkns_byslash:
        tokens = str(i).split("-")
        tkns_bydot = []
        for j in range(0,len(tokens)):
            temp_tkns = str(tokens[j]).split(".")
            tkns_bydot = tkns_bydot + temp_tkns
        total_tokens = total_tokens + tokens + tkns_bydot
    total_tokens = list(set(total_tokens))
    if "com" in total_tokens:
        total_tokens.remove("com")
    elif "http:" in total_tokens:
        total_tokens.remove("http:")
    return total_tokens

#MODEL
def FakeSiteDetection(link):
    #Getting the dataset
    df = pd.read_csv("urls.csv")
    df[df["label"] == "good"] = df[df["label"] == "good"].sample(n = 5000, random_state=33)
    df[df["label"] == "bad"] = df[df["label"] == "bad"].sample(n = 5000, random_state=33)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


    #Shuffling the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    #Shuffling
    df = np.array(df)
    random.shuffle(df)

    #Getting urls and labels
    x = [d[0] for d in df]
    y = [d[1] for d in df]

    #Setting Customtokens
    vectorizer = TfidfVectorizer(tokenizer=customtkns)

    #Vectorizing the column
    X = vectorizer.fit_transform(x)

    #Train Test Split
    xtrain, xtest,ytrain,ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    #Train Model
    dt = DecisionTreeClassifier()
    dt.fit(xtrain, ytrain)

    get_url = [link]
    predict_url = vectorizer.transform(get_url)
    value = dt.predict(predict_url)
    if value == "good":
        return f'{link}: The Entered URL is Legit'
    elif value == "bad":
        return f'{link}: The Entered URL is Fake'
    else:
        return "Enter a valid url"

