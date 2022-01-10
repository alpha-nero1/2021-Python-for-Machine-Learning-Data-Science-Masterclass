from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Just convention I like these declared at top even though it is not necessary.
model = None
col_names = None

# Create flask app.
app = Flask(__name__)

# Connect post api call to predict func.
@app.route('/predict', methods=['POST'])
def predict():
    # Get json request.
    feat_data = request.json

    # convert into pandas df and ensure matches with col names.
    df = pd.DataFrame(feat_data)
    df = df.reindex(columns=col_names)

    # predict and return
    prediction = list(model.predict(df))
    return jsonify({ 'prediction': str(prediction) });

# load model and setup column names.
if __name__ == '__main__':
    # Load model and feature models.
    model = joblib.load('final_model.pkl')
    col_names = joblib.load('column_names.pkl')

    app.run(debug=True)

# Woo and that's it folks!!