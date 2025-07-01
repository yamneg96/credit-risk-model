from pydantic import BaseModel

class RiskRequest(BaseModel):
    Value: float
    Frequency: int
    Monetary: float
    ProductCategory: str
    ChannelId: str
