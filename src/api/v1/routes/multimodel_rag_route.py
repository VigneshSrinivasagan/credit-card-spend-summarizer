from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from src.api.v1.services.multimodel_rag_service import add_document,query_documents,query_documents_static
from src.api.v1.schemas.query_schema import QueryRequest

router = APIRouter(
    prefix="/multimodelrag", tags=["creditcardsummary"]
)


@router.post("/embed/multimodel/document")
async def upload_document(file: UploadFile = File(...)):
    return await add_document(file)

@router.post("/query")
def query_endpoint_static(request: QueryRequest):
    docs = query_documents_static(request.query)
    return docs


@router.post("/query/stream")
async def query_endpoint(request: QueryRequest):
    print("Inside the query route...")
    answer = await query_documents(request.query)
    return StreamingResponse(answer, media_type="text/event-stream")
