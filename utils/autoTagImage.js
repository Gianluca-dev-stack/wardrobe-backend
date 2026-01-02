export async function autoTagImage(imageBase64) {
    try {
        const response = await fetch("http://192.168.1.205:3000/analyze-image", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ imageBase64 }),
        });

        const data = await response.json();
        return data.category || "Unknown";
    } catch (err) {
        console.error("Auto-tagging error:", err);
        return "Unknown";
    }
}