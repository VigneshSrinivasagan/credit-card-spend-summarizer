from pydantic import BaseModel, Field

from typing import List, Optional, Literal


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., example="Which merchants contributed to it?")

    chat_history: Optional[List[ChatHistoryMessage]] = Field(
        default_factory=list,
        example=[
            {"role": "user", "content": "What is my highest spend category?"},
            {"role": "assistant", "content": "Your highest spend category is Travel."},
        ],
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    policy_citations: str
    page_no: str
    document_name: str


class AIResponse(BaseModel):
    query: str = Field(description="The Given query by user must be present here")
    answer: str = Field(description="The generated response")
    policy_citations: str = Field(
        description="Give the Policy Citation (for document queries)"
    )
    page_no: str = Field(description="The page number in the metadata")
    document_name: str = Field(description="Name of the document used")
    sql_query_executed: Optional[str] = Field(
        default=None,
        description="The SQL query executed (for product/database queries)",
    )
