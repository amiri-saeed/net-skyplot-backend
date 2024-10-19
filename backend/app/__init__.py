# app/__init__.py
from flask import Flask
from app.routes import main

def create_app():
    app = Flask(__name__)

    # Register the blueprint
    app.register_blueprint(main)

    return app
