import secrets 
import hashlib 
from typing import Set 
from fastapi import FastAPI, Depends, HTTPException, status  
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from routers import pipeline_routes, search_routes
from options.base_options import BaseOptions

settings = BaseOptions()

security = HTTPBearer()


def hash_api_key(api_key: str): 
    """Hash an API key for secure storage comparison."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def load_valid_api_keys() -> Set[str]: 
    """Load and return hashed API key from the environment."""
    api_key = settings.auth_api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API_KEY not set"
        )
    
    valid_key = {hash_api_key(api_key)}  # can include logic for handling additional keys

    return valid_key


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    valid_api_keys = load_valid_api_keys()

    # Hash the provided API key
    provided_key_hash = hash_api_key(credentials.credentials)

    # Use constant-time comparison to prevent timing attacks
    is_valid = any(
        secrets.compare_digest(provided_key_hash, valid_key)
        for valid_key in valid_api_keys
    )
      
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return credentials.credentials


app = FastAPI(
    title="OMOP concept Assistant",
    description="The API to assist in identifying OMOP concepts",
    version="0.1.0",
    contact={
        "name": "BRC, University of Nottingham",
        "email": "james.mitchell-white1@nottingham.ac.uk",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    router=pipeline_routes.router,
    prefix="/pipeline",
    dependencies=[Depends(verify_api_key)]  
)

app.include_router(
    router=search_routes.router,
    prefix="/search",
    dependencies=[Depends(verify_api_key)]  
)


def main():
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__": 
    main()
    
