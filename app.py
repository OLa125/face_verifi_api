import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from io import BytesIO

app = Flask(__name__)

# تحميل النموذج مرة واحدة
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/", methods=["GET"])
def home():
    return "PromptDresser API is running!"

@app.route("/generate", methods=["POST"])
def generate_image():
    person_img = request.files.get("person")
    cloth_img = request.files.get("cloth")
    prompt = request.form.get("prompt", "A fashionable outfit")

    if not person_img or not cloth_img:
        return jsonify({"error": "Both 'person' and 'cloth' images are required."}), 400

    person_image = Image.open(person_img).convert("RGB")
    cloth_image = Image.open(cloth_img).convert("RGB")

    # التوليد باستخدام النموذج
    result = pipe(prompt=prompt, image=person_image, mask_image=cloth_image).images[0]

    # تحويل الصورة إلى ملف لإرسالها
    buffer = BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
