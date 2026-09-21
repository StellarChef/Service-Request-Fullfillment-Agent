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
