import sharp from "sharp";
import ort from "onnxruntime-node";

export async function preprocessImage(imageBase64) {
    const buffer = Buffer.from(imageBase64, "base64");

    // Resize + normalize
    const image = await sharp(buffer)
        .resize(224, 224)
        .toFormat("png")
        .raw()
        .toBuffer();

    const floatArray = new Float32Array(224 * 224 * 3);

    for (let i = 0; i < image.length; i++) {
        floatArray[i] = image[i] / 255.0;
    }

    return new ort.Tensor("float32", floatArray, [1, 3, 224, 224]);
}