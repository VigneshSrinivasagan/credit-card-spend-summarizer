from fastapi import FastAPI
from src.api.v1.routes import multimodel_rag_route

app = FastAPI()


@app.get("/")
def read_root():
    return {"Message": "The creditcard summarization bot is up and running. please check .../docs for more"}


app.include_router(multimodel_rag_route.router, prefix="/api/v1")
