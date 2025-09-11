from contextlib import contextmanager 

from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from options.base_options import BaseOptions

settings = BaseOptions()

def get_db_connection(): 
    """Get database connection parameters."""
    try: 
        connection_uri = settings.connection_url()
        
        return {
            "uri": connection_uri,
            "schema": settings.db_schema
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
        
