from flask_restful import Api
from apis.random_forest_api import ModelPredict

api = Api()
api.add_resource(ModelPredict, '/model')
