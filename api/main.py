# api/main.py
from fastapi import FastAPI
import torch
from pydantic import BaseModel
import torch.nn as nn

class MySimpleModel(nn.Module):
    def __init__(self):
        super(MySimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MySimpleModel()
model.load_state_dict(torch.load("../model/model.pth"))
model.eval()

app = FastAPI()

class InputData(BaseModel):
    input: list

@app.post("/predict")
def predict(data: InputData):
    input_tensor = torch.tensor(data.input).float()
    with torch.no_grad():
        output = model(input_tensor)
    return {"prediction": output.tolist()}
