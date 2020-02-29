import pickle

import numpy as np

from flask_restful import Resource, reqparse

with open('core/models/random-forest-0.8251.pkl', 'rb') as file:
    model = pickle.load(file)


class ModelPredict(Resource):
    post_parser = reqparse.RequestParser()
    post_parser.add_argument('Pclass', type=int, required=True)
    post_parser.add_argument('Sex', type=int, required=True)
    post_parser.add_argument('Age', type=int, required=True)
    post_parser.add_argument('Fare', type=int, required=True)
    post_parser.add_argument('Embarked', type=float, required=True)
    post_parser.add_argument('Relatives', type=int, required=True)
    post_parser.add_argument('Alone', type=int, required=True)

    def post(self):
        request = self.post_parser.parse_args()

        x_test = np.array([request['Pclass'], request['Sex'],
                           request['Age'], request['Fare'],
                           request['Embarked'], request['Relatives'],
                           request['Alone']])
        y_pred = model.predict([x_test])
        return {'Survived': int(y_pred[0])}
