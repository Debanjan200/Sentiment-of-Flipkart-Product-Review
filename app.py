import re,pickle
from nltk.stem.porter import PorterStemmer
from flask import Flask,render_template,request


with open("stopwords.txt","r") as f:
    stopwords=f.read().split("\n")
cv=pickle.load(open("vectorizer.pkl","rb"))
log_classifier=pickle.load(open("log_classifier.pkl","rb"))
sentiment_encoder=pickle.load(open("sentiment_encoder.pkl","rb"))
app=Flask(__name__)

def preprocessing(str):
    corpus=[]
    ps=PorterStemmer()
    emo=re.sub("[^a-zA-Z]",' ',str)
    emo=emo.lower()
    emo=emo.split()
    emo=[ps.stem(word) for word in emo if word not in set(stopwords)]
    emo=' '.join(emo)
    corpus.append(emo)
    data=cv.transform(corpus)
    return prediction(data)[0]

def prediction(data):
    return log_classifier.predict(data)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        review=request.form["review"]
        review=preprocessing(review)

    if review==0:
        sentiment="The review is Negative"

    elif review==1:
        sentiment="The review is Neutral"

    else:
        sentiment="The review is Positive"

    return render_template("index.html",prediction=review,st=sentiment)

if __name__=="__main__":
    app.run(debug=True)