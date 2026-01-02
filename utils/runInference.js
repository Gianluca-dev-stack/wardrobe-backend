import sharp from "sharp";
import { preprocessImage } from "./preprocessImage.js";

export async function runInference(session, imageBase64) {
    // Convert base64 → tensor
    const inputTensor = await preprocessImage(imageBase64);

    // Run ONNX inference
    const output = await session.run({ input: inputTensor });

    // ONNX model outputs a tensor named "output"
    const scores = output.output.data;

    // Find the index of the highest score
    let maxIndex = 0;
    for (let i = 1; i < scores.length; i++) {
        if (scores[i] > scores[maxIndex]) {
            maxIndex = i;
        }
    }

    return maxIndex; // or map to label if you have labels
}