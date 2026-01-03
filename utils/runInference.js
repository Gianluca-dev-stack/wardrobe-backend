import sharp from "sharp";

export async function runInference(session, imageBuffer) {
    // 1. Preprocess image → 224x224 RGB float32
    const resized = await sharp(imageBuffer)
        .resize(224, 224)
        .removeAlpha()
        .raw()
        .toBuffer();

    const floatArray = new Float32Array(224 * 224 * 3);
    for (let i = 0; i < resized.length; i++) {
        floatArray[i] = resized[i] / 255; // normalize to [0,1]
    }

    // 2. Create tensor: [1, 3, 224, 224]
    const inputTensor = new session.constructor.Tensor(
        "float32",
        floatArray,
        [1, 3, 224, 224]
    );

    // 3. Run inference
    const output = await session.run({ input: inputTensor });

    // 4. Extract logits
    const scores = output.output.data;

    // 5. Find top‑1 index
    let maxIndex = 0;
    for (let i = 1; i < scores.length; i++) {
        if (scores[i] > scores[maxIndex]) {
            maxIndex = i;
        }
    }

    return maxIndex;
}