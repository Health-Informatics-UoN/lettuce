from os import environ
from urllib.parse import quote_plus

from sqlalchemy.engine import URL as db_URL, create_engine
from sqlalchemy.orm import sessionmaker

DB_HOST = environ["DB_HOST"]
DB_USER = environ["DB_USER"]
DB_PASSWORD = quote_plus(environ["DB_PASSWORD"])
DB_NAME = environ["DB_NAME"]
DB_PORT = environ["DB_PORT"]
DB_SCHEMA = environ["DB_SCHEMA"]

url = db_URL.create(
    drivername="psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
)
engine = create_engine(url)

Session = sessionmaker
