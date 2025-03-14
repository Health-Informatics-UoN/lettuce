from os import environ
from urllib.parse import quote_plus

from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

DB_HOST = environ["DB_HOST"]
DB_USER = environ["DB_USER"]
DB_PASSWORD = quote_plus(environ["DB_PASSWORD"])
DB_NAME = environ["DB_NAME"]
DB_PORT = environ["DB_PORT"]
DB_SCHEMA = environ["DB_SCHEMA"]

connection_uri = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(
    connection_uri,
)
print(f"Real engine created: {engine.url}")

db_session = sessionmaker(engine)
