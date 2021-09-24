from flask import Flask, render_template, request
import site_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/",methods = ["GET","POST"])
def predict():
    if request.method == "POST":
        urls = request.form["link"]
        url_predict = site_prediction.FakeSiteDetection(urls)
        output = url_predict

    return render_template("index.html",site_url = output )

if __name__ =='__main__':
    app.run(debug = True)