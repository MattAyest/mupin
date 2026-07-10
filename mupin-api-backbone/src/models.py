from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class JobSubmit(BaseModel):
    job_type: str = Field(..., description="Job type discriminator, e.g. 'coding'")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Worker-specific payload")


class JobProgress(BaseModel):
    current_node: Optional[str] = None
    sandbox_loop_count: int = 0
    thoughts: list[str] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    worker_id: Optional[str] = None
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    cancel_requested: bool = False
    created_at: datetime
    started_at: Optional[datetime] = None
    updated_at: datetime
