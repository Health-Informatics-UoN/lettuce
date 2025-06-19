import os 
from fastapi import FastAPI, Depends, HTTPException, status  
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from routers import pipeline_routes, search_routes


security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = os.getenv("AUTH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API_KEY not set"
        )
    valid_api_keys = {api_key}
    if credentials.credentials not in valid_api_keys:
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
    
