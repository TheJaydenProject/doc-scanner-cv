from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class ScanRecord(db.Model):
    __tablename__ = "scan_records"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    char_count = db.Column(db.Integer, nullable=False)
    word_count = db.Column(db.Integer, nullable=False)
    processing_time_ms = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
