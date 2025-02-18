from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import pipeline_routes

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
)


def main():
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
