import fs from "fs";
import path from "path";

export function loadClothingLabels() {
    const filePath = path.join(process.cwd(), "models", "clothing_labels.json");
    const raw = fs.readFileSync(filePath, "utf8");
    return JSON.parse(raw);
}