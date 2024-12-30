import os
import sys

import cv2
import numpy
import numpy as np
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
from torchvision import models, transforms 
from .curricularface import get_model 


def load_image(image):
    img = image.convert('RGB')
    img = transforms.Resize((299, 299))(img)  # Resize to Inception input size
    img = transforms.ToTensor()(img)
    return img.unsqueeze(0)  # Add batch dimension

def get_face_keypoints(face_model, image_bgr):
    face_info = face_model.get(image_bgr)
    if len(face_info) > 0:
        return sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    return None

def sample_video_frames(video_path, num_frames=24):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def process_image(face_model, image_path):
    if isinstance(image_path, str):
        np_faceid_image = np.array(Image.open(image_path).convert("RGB"))
    elif isinstance(image_path, numpy.ndarray):
        np_faceid_image = image_path
    else:
        raise TypeError("image_path should be a string or PIL.Image.Image object")

    image_bgr = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)

    face_info = get_face_keypoints(face_model, image_bgr)
    if face_info is None:
        padded_image, sub_coord = pad_np_bgr_image(image_bgr)
        face_info = get_face_keypoints(face_model, padded_image)
        if face_info is None:
            print("Warning: No face detected in the image. Continuing processing...")
            return None, None
        face_kps = face_info['kps']
        face_kps -= np.array(sub_coord)
    else:
        face_kps = face_info['kps']
    arcface_embedding = face_info['embedding']

    norm_face = face_align.norm_crop(image_bgr, landmark=face_kps, image_size=224)
    align_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB)

    return align_face, arcface_embedding


def get_face_keypoints2(face_model, image_bgr):
    face_info = face_model.get(image_bgr)
    if len(face_info) > 0:
        return sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-2:]
    return None


def matrix_sqrt(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
    sqrt_matrix = (eigenvectors * sqrt_eigenvalues).mm(eigenvectors.T)
    return sqrt_matrix


def calculate_fid(real_activations, fake_activations, device="cuda"):
    real_activations_tensor = torch.tensor(real_activations).to(device)
    fake_activations_tensor = torch.tensor(fake_activations).to(device)

    mu1 = real_activations_tensor.mean(dim=0)
    sigma1 = torch.cov(real_activations_tensor.T)
    mu2 = fake_activations_tensor.mean(dim=0)
    sigma2 = torch.cov(fake_activations_tensor.T)

    ssdiff = torch.sum((mu1 - mu2) ** 2)
    covmean = matrix_sqrt(sigma1.mm(sigma2))
    if torch.is_complex(covmean):
        covmean = covmean.real
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.item()



def process_image2(face_model, image_path):
    if isinstance(image_path, str):
        np_faceid_image = np.array(Image.open(image_path).convert("RGB"))
    elif isinstance(image_path, numpy.ndarray):
        np_faceid_image = image_path
    else:
        raise TypeError("image_path should be a string or PIL.Image.Image object")

    image_bgr = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)

    face_info = get_face_keypoints2(face_model, image_bgr)
    print(len(face_info)) 
    align_face_list = []
    arcface_embedding_list = []
    for f in face_info: 
        face_kps = f['kps']
        arcface_embedding = f['embedding']
        norm_face = face_align.norm_crop(image_bgr, landmark=face_kps, image_size=224)
        align_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB)
        align_face_list.append(align_face)
        arcface_embedding_list.append(arcface_embedding)
    return align_face_list, arcface_embedding_list


@torch.no_grad()
def inference(face_model, img, device):
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    embedding = face_model(img).detach().cpu().numpy()[0]
    return embedding / np.linalg.norm(embedding)


def get_activations(images, model, batch_size=16):
    model.eval()
    activations = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pred = model(batch)
            activations.append(pred)
    activations = torch.cat(activations, dim=0).cpu().numpy()
    if activations.shape[0] == 1:
        activations = np.repeat(activations, 2, axis=0)
    return activations


import math

def cosine_similarity(list_1, list_2): 
    cos_list = []
    for list1 in list_1:
        for list2 in list_2:
            dot_product = sum(a * b for a, b in zip(list1, list2))
            magnitude1 = math.sqrt(sum(a ** 2 for a in list1))
            magnitude2 = math.sqrt(sum(b ** 2 for b in list2))
            cos_list.append(dot_product / (magnitude1 * magnitude2))
    return max(cos_list)


def get_face_sim_and_fid(
    image_path_list,
    video_path,
    model_path,
    device="cuda",
): 
    face_arc_path = os.path.join(model_path, "face_encoder")
    face_cur_path = os.path.join(face_arc_path, "glint360k_curricular_face_r101_backbone.bin")

    # Initialize FaceEncoder model for face detection and embedding extraction
    face_arc_model = FaceAnalysis(root=face_arc_path, providers=['CUDAExecutionProvider'])
    face_arc_model.prepare(ctx_id=0, det_size=(320, 320))

    # Load face recognition model
    face_cur_model = get_model('IR_101')([112, 112])
    face_cur_model.load_state_dict(torch.load(face_cur_path, map_location="cpu"))
    face_cur_model = face_cur_model.to(device)
    face_cur_model.eval()

    # Load InceptionV3 model for FID calculation
    fid_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    fid_model.fc = torch.nn.Identity()  # Remove final classification layer
    fid_model.eval()
    fid_model = fid_model.to(device)

    align_face_image_list = []
    arcface_image_embedding_list = []
    real_activations_list = []
    cur_image_embedding_list = []
    for image_path in image_path_list: 
        align_face_image, arcface_image_embedding = process_image(face_arc_model, image_path)
        if align_face_image is None:
            print(f"Error processing image at {image_path}")
            return 
        #print(len(arcface_image_embedding)) #512
        align_face_image_list.append(align_face_image)
        arcface_image_embedding_list.append(arcface_image_embedding)

        cur_image_embedding = inference(face_cur_model, align_face_image, device)
        cur_image_embedding_list.append(cur_image_embedding)
        align_face_image_pil = Image.fromarray(align_face_image)
        real_image = load_image(align_face_image_pil).to(device)
        real_activations = get_activations(real_image, fid_model) 
        #print(len(real_activations[0])) # 2048
        real_activations_list.append(real_activations)

    video_frames = sample_video_frames(video_path, num_frames=24)
    print("video frames: ", len(video_frames))

    cur_scores = []
    arc_scores = []
    fid_face = []

    for frame in video_frames:
        # Convert to RGB once at the beginning
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for ArcFace embeddings
        align_face_frame_list, arcface_frame_embedding_list = process_image2(face_arc_model, frame_rgb)
        print(len(arcface_frame_embedding_list))
        
        cos = cosine_similarity(arcface_image_embedding_list, arcface_frame_embedding_list)
        print("cos: ", cos)
        arc_scores.append(cos)

        # Process FID score 
        f_list = []
        for align_face_frame in align_face_frame_list: 
            align_face_frame_pil = Image.fromarray(align_face_frame)
            fake_image = load_image(align_face_frame_pil).to(device)
            fake_activations = get_activations(fake_image, fid_model)
            for real_activations in real_activations_list:
                fid_score = calculate_fid(real_activations, fake_activations, device)
                f_list.append(fid_score)
        print("fid: ", min(f_list)) 
        fid_face.append(min(f_list))
    
    # Aggregate results with default values for empty lists
    avg_arc_score = np.mean(arc_scores) if arc_scores else 0.0
    avg_fid_score = np.mean(fid_face) if fid_face else 0.0
    return avg_arc_score, avg_fid_score
        

