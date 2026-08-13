# Deepfake Video Detector — Hugging Face Spaces version
# Model file (stage1_efficientnet_v2.pth) must be uploaded alongside this file.

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import gradio as gr
import os

# ---- CONFIG ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# On Spaces, the model file sits in the same folder as this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stage1_efficientnet_v2.pth')
FRAME_SAMPLE_COUNT = 10

# ---- LOAD MODEL (once, at startup) ----
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=DEVICE)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)


def analyze_video(video_path):
    if video_path is None:
        return "No video uploaded.", None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return "Could not read video.", None

    sample_indices = np.linspace(0, total_frames - 1, FRAME_SAMPLE_COUNT, dtype=int)

    probs = []
    best_face_tensor = None
    best_prob_extremity = -1

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face = mtcnn(rgb_frame)
        except Exception:
            face = None
        if face is None:
            continue

        input_tensor = transform(
            transforms.ToPILImage()(face.mul(0.5).add(0.5))
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
        probs.append(prob)

        extremity = abs(prob - 0.5)
        if extremity > best_prob_extremity:
            best_prob_extremity = extremity
            best_face_tensor = face

    cap.release()

    if len(probs) == 0:
        return "No face detected in this video. Try a clearer, front-facing clip.", None

    avg_prob = sum(probs) / len(probs)
    label = "FAKE" if avg_prob > 0.5 else "REAL"
    confidence_pct = avg_prob * 100 if label == "FAKE" else (1 - avg_prob) * 100

    result_text = (
        f"### Prediction: {label}\n"
        f"**Confidence:** {confidence_pct:.1f}%\n\n"
        f"(Average fake-probability across {len(probs)} sampled frames: {avg_prob:.3f})"
    )

    heatmap_image = None
    if best_face_tensor is not None:
        face_np = best_face_tensor.permute(1, 2, 0).mul(0.5).add(0.5).clamp(0, 1).numpy()
        input_tensor = transform(
            transforms.ToPILImage()(best_face_tensor.mul(0.5).add(0.5))
        ).unsqueeze(0).to(DEVICE)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        heatmap_image = show_cam_on_image(face_np, grayscale_cam, use_rgb=True)

    return result_text, heatmap_image


demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload a video clip"),
    outputs=[
        gr.Markdown(label="Result"),
        gr.Image(label="Grad-CAM: where the model focused"),
    ],
    title="Deepfake Video Detector",
    description=(
        "Upload a short video clip. The model samples frames, detects the face, "
        "and predicts whether the video is real or a deepfake. The heatmap shows "
        "which facial regions most influenced the model's decision — trained on "
        "FaceForensics++ (Deepfakes, Face2Face, FaceSwap), evaluated cross-dataset "
        "on Celeb-DF (83.3% accuracy)."
    ),
)

# Render provides the port via the PORT environment variable
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
