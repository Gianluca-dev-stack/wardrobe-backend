import torch
import torchvision.models as models

OUTPUT_PATH = "../models/mobilenetv2.onnx"


def export_mobilenetv2():
    print("📦 Loading MobileNetV2 from torchvision...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print("📝 Exporting to ONNX using the OLD exporter...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        # 👇 THIS is the magic line that avoids onnxscript
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )

    print(f"✅ Export complete! Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    export_mobilenetv2()
