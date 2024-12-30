import os
import cv2
from transformers import CLIPModel, CLIPProcessor

def compute_clip_score(video_path, model, processor, prompt, device, num_frames=24):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    for i in range(num_frames):
        frame_idx = int(i * total_frames / num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Frame at index {frame_idx} could not be read.")

    cap.release()

    if model is not None:
        inputs = processor(text=prompt, images=frames, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        average_score = logits_per_image.mean().item()

    return average_score