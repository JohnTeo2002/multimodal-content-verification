from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import reconcile, trust_and_impact, verify_claim

app = FastAPI(
    title="Content Verification API",
    description="Backend API for claim verification and human response scoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=500)
    evidence: List[str] = Field(default_factory=list)
    scores: Optional[List[float]] = Field(None, min_length=3, max_length=3)


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/verify")
def verify_content(payload: VerifyRequest):
    try:
        check = verify_claim(payload.claim, payload.evidence)
        result = {"claim_check": check}
        if payload.scores:
            if len(payload.scores) != 3:
                raise ValueError("scores must contain exactly 3 values")
            response = trust_and_impact(*payload.scores)
            result["human_response"] = response
            result["reconciliation"] = reconcile(check, response)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
