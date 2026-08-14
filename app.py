# Deepfake Video Detector — Polished UI version (Gradio Blocks)
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
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stage1_efficientnet_v2.pth')
FRAME_SAMPLE_COUNT = 6  # reduced from 10 for faster response on limited CPU

# ---- LOAD MODEL ----
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


def analyze_video(video_path, progress=gr.Progress()):
    if video_path is None:
        return "Please upload a video first.", None, None

    progress(0, desc="Reading video...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return "Could not read this video file.", None, None

    sample_indices = np.linspace(0, total_frames - 1, FRAME_SAMPLE_COUNT, dtype=int)

    probs = []
    best_face_tensor = None
    best_prob_extremity = -1

    for i, idx in enumerate(sample_indices):
        progress((i + 1) / len(sample_indices), desc=f"Analyzing frame {i+1}/{len(sample_indices)}...")
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
        return (
            "**No face detected.** Try a clearer, front-facing, well-lit clip.",
            None,
            None,
        )

    avg_prob = sum(probs) / len(probs)
    is_fake = avg_prob > 0.5
    label = "FAKE" if is_fake else "REAL"
    confidence_pct = avg_prob * 100 if is_fake else (1 - avg_prob) * 100

    color = "#e74c3c" if is_fake else "#2ecc71"
    icon = "⚠️" if is_fake else "✅"

    result_html = f"""
    <div style="text-align:center; padding: 20px; border-radius: 12px; background: {color}22; border: 2px solid {color};">
        <div style="font-size: 42px;">{icon}</div>
        <div style="font-size: 28px; font-weight: 700; color: {color}; margin-top: 8px;">{label}</div>
        <div style="font-size: 16px; color: #ccc; margin-top: 6px;">Confidence: {confidence_pct:.1f}%</div>
        <div style="font-size: 12px; color: #888; margin-top: 10px;">
            Based on {len(probs)} of {FRAME_SAMPLE_COUNT} sampled frames &middot; avg fake-probability {avg_prob:.3f}
        </div>
    </div>
    """

    heatmap_image = None
    if best_face_tensor is not None:
        face_np = best_face_tensor.permute(1, 2, 0).mul(0.5).add(0.5).clamp(0, 1).numpy()
        input_tensor = transform(
            transforms.ToPILImage()(best_face_tensor.mul(0.5).add(0.5))
        ).unsqueeze(0).to(DEVICE)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        heatmap_image = show_cam_on_image(face_np, grayscale_cam, use_rgb=True)

    return result_html, heatmap_image, gr.update(visible=True)


# ---- CUSTOM THEME ----
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
)

CUSTOM_CSS = """
#title { text-align: center; margin-bottom: 0px; }
#subtitle { text-align: center; color: #999; margin-top: 4px; margin-bottom: 24px; }
.stat-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    background: #6366f122; border: 1px solid #6366f1; color: #a5b4fc;
    font-size: 13px; margin: 0 4px;
}
"""

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Deepfake Video Detector") as demo:
    gr.HTML("""
        <h1 id="title">🎭 Deepfake Video Detector</h1>
        <p id="subtitle">Upload a video &mdash; AI analyzes facial regions and shows you exactly what it sees</p>
        <div style="text-align:center; margin-bottom: 20px;">
            <span class="stat-badge">📊 83.3% cross-dataset accuracy</span>
            <span class="stat-badge">🧠 EfficientNet-B0</span>
            <span class="stat-badge">🔍 Grad-CAM explainability</span>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload a video clip")
            analyze_btn = gr.Button("🔎 Analyze Video", variant="primary", size="lg")
            gr.Markdown(
                "*Tip: short, front-facing, well-lit clips work best. "
                "First run may take up to a minute on the free server.*"
            )

        with gr.Column(scale=1):
            result_output = gr.HTML(label="Result")
            heatmap_output = gr.Image(label="Grad-CAM: what the model focused on", show_label=True)

    with gr.Accordion("ℹ️ How this works", open=False):
        gr.Markdown("""
        This model was trained on **FaceForensics++** (Deepfakes, Face2Face, and FaceSwap
        manipulation methods) and evaluated cross-dataset on **Celeb-DF** to test real-world
        generalization — achieving **83.3% accuracy** on completely unseen fake-generation techniques.

        The model samples several frames from your video, detects the face in each, and predicts
        real vs. fake per frame. The final result is the average across all frames.

        The **Grad-CAM heatmap** highlights which facial regions most influenced the model's decision
        — warm colors (red/orange) show where the model is "looking" when making its call.
        """)

    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[result_output, heatmap_output, video_input],
    )

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
