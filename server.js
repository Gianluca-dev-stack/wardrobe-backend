import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";

import { loadOnnxModel } from "./models/loadOnnxModel.js";
import { runInference } from "./utils/runInference.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// -----------------------------
// Health check (Render uses this)
// -----------------------------
app.get("/", (req, res) => {
    res.send("Wardrobe backend is running");
});

// -----------------------------
// Load label mappings
// -----------------------------
const clothingLabels = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "models", "clothing_labels.json"))
);

const imagenetSubset = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "models", "imagenet_clothing_subset.json"))
);

// -----------------------------
// Load ONNX model at startup
// -----------------------------
let session;
(async () => {
    try {
        session = await loadOnnxModel();
        console.log("✓ ONNX model loaded");
    } catch (err) {
        console.error("❌ Failed to load ONNX model:", err);
    }
})();

// -----------------------------
// Auto-tagging route
// -----------------------------
app.post("/analyze-image", async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        console.log("🔥 /analyze-image hit");

        // 1. Validate presence
        if (!imageBase64) {
            console.log("❌ Missing imageBase64");
            return res.status(400).json({ error: "Missing imageBase64" });
        }

        // 2. Validate format
        if (
            !imageBase64.startsWith("data:image/jpeg") &&
            !imageBase64.startsWith("data:image/png")
        ) {
            console.log("❌ Unsupported image format");
            return res.status(400).json({ error: "Unsupported image format" });
        }

        // 3. Strip prefix → raw base64
        const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, "");
        const buffer = Buffer.from(base64Data, "base64");

        // 4. Validate buffer size
        if (buffer.length < 5000) {
            console.log("❌ Image too small or corrupted");
            return res.status(400).json({ error: "Image too small or corrupted" });
        }

        // 5. Run ONNX inference
        const predictionIndex = await runInference(session, base64Data);

        // 6. Convert index → raw ImageNet label
        const rawLabel = Object.keys(imagenetSubset).find(
            key => imagenetSubset[key] === predictionIndex
        );

        // 7. Convert raw label → wardrobe category
        const category = clothingLabels[rawLabel] || "Unknown";

        console.log(`✓ Prediction: ${category} (${rawLabel})`);

        // 8. Respond
        res.json({
            success: true,
            category,
            rawLabel,
            index: predictionIndex
        });

    } catch (err) {
        console.error("❌ analyze-image error:", err);
        res.status(500).json({ success: false, error: err.message });
    }
});

// -----------------------------
// Start server
// -----------------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});