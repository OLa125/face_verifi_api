from flask import Flask, request, jsonify
import cv2
import numpy as np
from numpy.linalg import norm
import insightface
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# تهيئة نموذج InsightFace
face_model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

@app.route('/verify', methods=['POST'])
def verify_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'يرجى رفع صورتين باسم image1 و image2'}), 400

    img1_file = request.files['image1']
    img2_file = request.files['image2']

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], img1_file.filename)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], img2_file.filename)
    img1_file.save(path1)
    img2_file.save(path2)

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        return jsonify({'error': 'فشل في تحميل الصور'}), 400

    faces1 = face_model.get(img1)
    faces2 = face_model.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return jsonify({'error': 'وجه غير مكتشف في إحدى الصور'}), 400

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    similarity = cosine_similarity(emb1, emb2)
    result = similarity > 0.5

    return jsonify({
        'similarity': float(similarity),
        'same_person': result
    })

if __name__ == '__main__':
    app.run(debug=True)