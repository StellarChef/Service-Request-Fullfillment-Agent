from typing import Literal

from pydantic import BaseModel


class ClientInput(BaseModel):
    textclient: str


class Client(BaseModel):
    name: str
    email: str
    phone: str | None


class Request(BaseModel):
    id: str
    title: str
    client_input: ClientInput

class Complaint(BaseModel):
    request: Request
    client: Client

class Inquiry(BaseModel):
    request: Request
    client: Client
    
class Spam(BaseModel):
    request: Request
    client: Client


class InputClassificationModel(BaseModel):
    category: Literal["COMPLAINT", "INQUIRY", "SPAM"]
    confidence: float
    language: str
    flagged: bool

