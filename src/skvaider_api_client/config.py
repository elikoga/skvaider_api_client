from pydantic import BaseModel


class Config(BaseModel):
    skvaider_token: str
    skvaider_url: str
