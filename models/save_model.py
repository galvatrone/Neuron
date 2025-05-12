# model/save_model.py
import torch
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
torch.save(model.state_dict(), "model.pth")

dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])
