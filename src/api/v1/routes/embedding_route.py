from fastapi import APIRouter, File, UploadFile
from api.v1.services import add_document

router = APIRouter(
    prefix="/api/v1",
    tags=["creditcards","spendsummary","creditcardsummary"]
)

@router.post("/embed/multimodel/document")
async def upload_document(file: UploadFile = File(...)):
    return await add_document(file)
