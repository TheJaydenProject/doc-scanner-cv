import os
import torch
import torchvision.models as models

os.makedirs("models", exist_ok=True)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
model.eval()

dummy_input = torch.zeros(1, 3, 224, 224)

try:
    torch.onnx.export(
        model,
        dummy_input,
        "models/doc_classifier.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
    print("Exported models/doc_classifier.onnx")
except Exception as e:
    print(f"Export failed: {e}")
