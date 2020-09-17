# Make a flask API for our DL Model

from flask_restplus import Api, Resource, fields
# Import the WSGI application library
from werkzeug.utils import cached_property
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

# Import FLASK
from flask import Flask, request, jsonify

import numpy as np
import pickle

from firebase import firebase

import pyrebase

config = {
  "apiKey": "",
  "authDomain": "",
  "databaseURL": "",
  "storageBucket": ""
}


firebase = pyrebase.initialize_app(config)
userEmail = ''
userPass = ''

auth = firebase.auth()

# Log the user in
user = auth.sign_in_with_email_and_password(userEmail, userPass)

# Get a reference to the database service
db = firebase.database()

# data to save
data = {
            "name": "Mortimer Williams"
            }

# Pass the user's idToken to the push method
results = db.child("test").push(data, user['idToken'])
print results
results = db.child("test").get(user['idToken'])
print results.val()



# Define app and api object
application = Flask(__name__)
api = Api(application, version='1.0', title='Logistic Regression', description='Logistic Regression')
ns = api.namespace('DS2_3_docker_and_aws', description='Methods')

# Define arguments for our API, in this case, it is going to be just a comma separated string
single_parser = api.parser()
single_parser.add_argument('input', type=str, required=True, help='input CSV')

# Load objects from pickle files
labelEncoder1 = pickle.load(open('pickle_files/labelEncoder1.pickle', 'rb'))
labelEncoder2 = pickle.load(open('pickle_files/labelEncoder2.pickle', 'rb'))
standardScaler = pickle.load(open('pickle_files/standardScaler.pickle', 'rb'))
model = pickle.load(open('pickle_files/log_reg_model.pickle', 'rb'))



@ns.route('/prediction')
class LogRegPrediction(Resource):
    """Applies pre-trained Logistic Regression model to input data"""
    @api.doc(parser=single_parser, description='Upload input data')
    def post(self):
        # Parse arguments
        args = single_parser.parse_args()

        # Get input data in string format
        input_data = args.input

        # Convert data to numpy array
        dataset = np.array(input_data.split(','))

        # Get only the data that we need
        X = dataset[3:]

        # Apply label encoders to categorical data
        X[1] = int(labelEncoder1.transform([X[1]]))
        X[2] = int(labelEncoder2.transform([X[2]]))

        # Scale the data using our standardScaler
        X = standardScaler.transform([X])

        # Make the prediction
        prediction = model.predict(X)

        # Return prediction
        return {'prediction': str(prediction)}


if __name__ == '__main__':
    application.run()