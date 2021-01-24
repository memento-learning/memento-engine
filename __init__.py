from flask import Flask
from flask_cors import CORS
from .routes import routes_blueprint 

def create_app():
    app = Flask(__name__)
    app.debug = True
    CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})
    app.register_blueprint(routes_blueprint)
    return app

