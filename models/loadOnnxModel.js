import path from "path";
import { fileURLToPath } from "url";
import { InferenceSession } from "onnxruntime-node";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let session = null;

export async function loadOnnxModel() {
    if (session) return session;

    const modelPath = path.join(process.cwd(), "models", "mobilenetv2.onnx");

    console.log("📦 Loading ONNX model from:", modelPath);

    session = await InferenceSession.create(modelPath, {
        executionProviders: ["cpu"]
    });

    console.log("✓ ONNX model loaded successfully");
    return session;
}