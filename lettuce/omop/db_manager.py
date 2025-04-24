from os import environ
from contextlib import contextmanager 
from urllib.parse import quote_plus

from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker


def get_db_connection(): 
    """Get database connection parameters."""
    try: 
        db_host = environ["DB_HOST"]
        db_user = environ["DB_USER"]
        db_password = quote_plus(environ["DB_PASSWORD"])
        db_name = environ["DB_NAME"]
        db_port = environ["DB_PORT"]
        db_schema = environ["DB_SCHEMA"]

        connection_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        return {
            "uri": connection_uri,
            "schema": db_schema
        }
    except Exception as e:
        raise ValueError(f"Database configuration error: {e}")


db_config = get_db_connection()
engine = create_engine(db_config["uri"])
DB_SCHEMA = db_config["schema"]
db_session = sessionmaker(engine)


@contextmanager 
def get_session(): 
    """Get a session that will be properly closed after use."""
    session = db_session()
    try: 
        yield session 
    finally: 
        session.close() 
