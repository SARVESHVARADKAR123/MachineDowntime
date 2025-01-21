from flask import Flask
from utils import preprocess_data, train_model, make_prediction

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    return preprocess_data()

@app.route('/train', methods=['POST'])
def train():
    return train_model()

@app.route('/predict', methods=['POST'])
def predict():
    return make_prediction()

if __name__ == '__main__':
    app.run(debug=True)
