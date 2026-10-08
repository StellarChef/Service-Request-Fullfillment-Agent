from typing import Literal

from pydantic import BaseModel


class ClientInput(BaseModel):
    textclient: str


class Client(BaseModel):
    name: str
    email: str
    phone: str | None


class InputClassificationModel(BaseModel):
    category: Literal["COMPLAINT", "INQUIRY", "SPAM"]
    confidence: float
    language: str
    flagged: bool

class Policychunk(BaseModel):
    id: str
    text: str
    similarity: float
    category: Literal["COMPLAINT", "INQUIRY", "SPAM"]

class ServiceResponse(BaseModel):
  classification: InputClassificationModel
  matched_policies: list[Policychunk]
  reply: str
  flagged: bool

