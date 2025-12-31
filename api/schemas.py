"""
Pydantic schemas for API requests/responses
"""

from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request schema for queries"""
    query: str
    top_k: Optional[int] = 5


class DocumentResponse(BaseModel):
    """Response schema for documents"""
    content: str
    score: float
    metadata: Optional[dict] = None
