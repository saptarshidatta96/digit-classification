from flask import Flask, request, jsonify
import joblib


app = Flask(__name__)

@app.route('/predict', methods=['POST', 'GET'])


def predict():
    if digit_recognizer_model:
        try:
            if request.get_json() is not None:
                json_ = request.json
                prediction1 = list(digit_recognizer_model.predict([json_['image1']]))
                prediction2 = list(digit_recognizer_model.predict([json_['image2']]))
                
                return jsonify({'prediction1': str(prediction1),
                                'prediction2': str(prediction2)})
                                

        except Exception as e:

            return jsonify({'error': e})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':

    digit_recognizer_model = joblib.load("digit_recognizer_model.pkl")
    print ('Model loaded')
    app.run(host='0.0.0.0', port=12345, debug=True)