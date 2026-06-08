from fastapi import FastAPI
from src.api.v1.routes import multimodel_rag_route

app = FastAPI()


@app.get("/")
def read_root():
    return {"Message": "Hello World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(multimodel_rag_route.router, prefix="/api/v1")
