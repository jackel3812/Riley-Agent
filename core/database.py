import os
from pathlib import Path
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.String(500))
    response = db.Column(db.String(1000))

def init_db(app):
    # Ensure the chat directory exists
    chat_dir = os.path.dirname(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
    Path(chat_dir).mkdir(parents=True, exist_ok=True)
    
    db.init_app(app)
    with app.app_context():
        db.create_all()

def save_conversation(user_input, response):
    entry = Conversation(user_input=user_input, response=response)
    db.session.add(entry)
    db.session.commit()
