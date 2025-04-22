from fastapi import FastAPI
from pydantic import BaseModel
from poem_classifier_model import predict_genre  

from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React Vite runs on this port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PoemRequest(BaseModel):
    poem: str

@app.get("/")
def doit():
    return {"hello" : "hello"}

@app.post("/predict")
def predict(request: PoemRequest):
    genre = predict_genre(request.poem)
    return {"genre": genre}
