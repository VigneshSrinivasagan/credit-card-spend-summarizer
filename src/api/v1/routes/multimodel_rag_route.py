from fastapi import APIRouter, File, UploadFile
from api.v1.services import add_document
from src.api.v1.services.multimodel_rag_service import query_documents
from src.api.v1.schemas.query_schema import QueryRequest

router = APIRouter(
    prefix="/api/v1",
    tags=["creditcards","spendsummary","creditcardsummary"]
)


@router.post("/embed/multimodel/document")
async def upload_document(file: UploadFile = File(...)):
    return await add_document(file)


@router.post("/query")
def query_endpoint(request: QueryRequest):
   docs = query_documents(request.query)
   return docs