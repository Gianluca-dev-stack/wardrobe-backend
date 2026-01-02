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

// Health check route (Render uses this)
app.get("/", (req, res) => {
    res.send("Wardrobe backend is running");
});

// Load clothing label mappings
const clothingLabels = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "models", "clothing_labels.json"))
);

const imagenetSubset = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "models", "imagenet_clothing_subset.json"))
);

// Load ONNX model at startup
let session;
(async () => {
    try {
        session = await loadOnnxModel();
        console.log("✓ ONNX model loaded");
    } catch (err) {
        console.error("❌ Failed to load ONNX model:", err);
    }
})();

// Auto-tagging route (your core feature)
app.post("/analyze-image", async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        // -----------------------------
        // 1. Validate presence
        // -----------------------------
        if (!imageBase64) {
            return res.status(400).json({ error: "Missing imageBase64" });
        }

        // -----------------------------
        // 2. Validate format (JPEG/PNG)
        // -----------------------------
        if (
            !imageBase64.startsWith("data:image/jpeg") &&
            !imageBase64.startsWith("data:image/png")
        ) {
            return res.status(400).json({ error: "Unsupported image format" });
        }

        // -----------------------------
        // 3. Extract base64 → buffer
        // -----------------------------
        const base64Data = imageBase64.split(",")[1];
        const buffer = Buffer.from(base64Data, "base64");

        // -----------------------------
        // 4. Validate buffer size
        // -----------------------------
        if (buffer.length < 5000) {
            return res.status(400).json({ error: "Image too small or corrupted" });
        }

        // -----------------------------
        // 5. Run ONNX inference
        // -----------------------------
        const predictionIndex = await runInference(session, imageBase64);

        // Convert index → raw ImageNet label
        const rawLabel = Object.keys(imagenetSubset).find(
            key => imagenetSubset[key] === predictionIndex
        );

        // Convert raw label → wardrobe category
        const category = clothingLabels[rawLabel] || "Unknown";

        // -----------------------------
        // 6. Respond
        // -----------------------------
        res.json({
            success: true,
            category,
            rawLabel,
            index: predictionIndex
        });

    } catch (err) {
        console.error("analyze-image error:", err);
        res.status(500).json({ success: false, error: err.message });
    }
});

// Start server (Render requires process.env.PORT)
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});