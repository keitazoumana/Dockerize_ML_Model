from flask import Flask,request, render_template, jsonify
import joblib

MODEL_FILE_NAME = "svm_best_model.joblib"
MODEL_FOLDER = "./models/"

VECTOR_FILE_NAME = "data_vectorizer.joblib"
VECTOR_FOLDER = "./vectors/"

# 2. Load the model
loaded_svm_model = joblib.load(open(MODEL_FOLDER+MODEL_FILE_NAME, 'rb'))
loaded_data_vector = joblib.load(open(VECTOR_FOLDER+VECTOR_FILE_NAME, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    
    input_text = request.form['message']

    preprocessed_text = loaded_data_vector.transform([input_text])

    prediction = loaded_svm_model.predict(preprocessed_text)

    label_info = '' 

    if(prediction[0] == 0):
        label_info = "Ham ✅"

    else:
        label_info = "Spam 🚨" 

    return render_template('home.html', prediction='This message is a: {}'.format(label_info))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=9696)
