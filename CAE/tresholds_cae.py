import os
import torch
import numpy as np
import cv2
import math
import pandas as pd

from models.CAE import ConvolutionalAutoencoder
from data.preprocessing import VideoPreprocessor
from data.dataset import VideoDataset
from data.preprocessing import VideoPreprocessor
from metrics import Evaluation
import matplotlib.pyplot as plt


os.environ["ALBUMENTATIONS_DISABLE_UPDATE_CHECK"] = "1"

img_size = 128
max_num_layers = int(math.log2(img_size // 4))

TEST_DIR = "data/my_data/Test"
LOG_PATH = "outputs/log_tr/threshold_comparison.csv"
PATH_MODEL = "outputs/models/latent_2/"


labels_map = {
    "output_video_pnp_0_1.avi": "labels_pnp_0_1.csv",
    "output_video_pnp_0_2.avi": "labels_pnp_0_2.csv",
    "output_video_pnp_0_3.avi": "labels_pnp_0_3.csv",
    "output_video_pnp_0_4.avi": "labels_pnp_0_4.csv",
}

COMBINATIONS = [
   {"exp_id": 2,"kernel_size": 7, "channels_out_1_layer": 32, "latent_dim":  316,"num_layers":5},
    {"exp_id":5 ,"kernel_size": 4, "channels_out_1_layer": 64, "latent_dim":  328,"num_layers":5},
    {"exp_id": 8,"kernel_size": 3, "channels_out_1_layer": 32, "latent_dim": 121,"num_layers":5},
    {"exp_id": 10,"kernel_size": 4, "channels_out_1_layer": 32, "latent_dim": 599,"num_layers":5},
    {"exp_id":15 ,"kernel_size": 3, "channels_out_1_layer": 32, "latent_dim":  92,"num_layers":4},
    {"exp_id": 18,"kernel_size": 5, "channels_out_1_layer": 32, "latent_dim":   126,"num_layers":5},
    {"exp_id": 19,"kernel_size": 4, "channels_out_1_layer": 32, "latent_dim": 687,"num_layers":4},
   
]


def log_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame([data])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


def load_model(params, model_path, device):
    model = ConvolutionalAutoencoder(
        input_channels=1,
        latent_dim=params['latent_dim'],
        kernel_size=params['kernel_size'],
        channels_out_1_layer=params['channels_out_1_layer'],
        num_layers=params['num_layers'],
        in_size=img_size,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_frame_losses(model, video_path, preprocessor, device):
    cap = cv2.VideoCapture(video_path)
    losses = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            p_frame = preprocessor.process_frame(frame, validate_quality=False)
            tensor = torch.from_numpy(p_frame).view(1, 1, img_size, img_size).to(device).float()
            
            with torch.no_grad():
                recon = model(tensor)
                loss  = torch.mean((tensor - recon) ** 2).item()
                losses.append(loss)
            frame_idx += 1
        except Exception as e:
            print(f"  [DEBUG] {os.path.basename(video_path)}: {e}")
            break
    cap.release()
    return np.array(losses)


def evaluate(params, model, device):
 
    preprocessor = VideoPreprocessor(
        target_size=(img_size, img_size),
        convert_to_grayscale=True,
        quality_threshold=0.0
    )

    all_metr_losses = []
    all_labels = []



    for video_file, label_file in labels_map.items():
        video_path = os.path.join(TEST_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"  Не найдено: {video_path}")
            continue

        losses_arr = get_frame_losses(model, video_path, preprocessor, device)
        true_label = np.genfromtxt(label_file, delimiter=',', usecols=(1), skip_header=1)
        if len(losses_arr) != len(true_label):
            print(f"  [WARN] Длины не совпадают для {video_file}: losses ({len(losses_arr)}), labels ({len(true_label)})")
            continue

        normal_losses  = losses_arr[true_label == 0]
        anomaly_losses = losses_arr[true_label == 1]

        if len(normal_losses) == 0 or len(anomaly_losses) == 0:
            print(f"  [{video_file}] пропуск — нет нормы или аномалий")
            continue
        
        all_labels.extend(true_label)
        all_metr_losses.extend(losses_arr)


        evaluator = Evaluation()
    
    threshold_rows = evaluator.treshold_eval(

                np.array(all_metr_losses),
                np.array(all_labels), 
                exp_id=f"exp_{params['exp_id']}"
            )


    return threshold_rows
    

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    
    experiments = []


    for combo in COMBINATIONS:
  
            experiments.append({
                **combo,
 

            })



    print(f"Всего экспериментов: {len(experiments)}")

    for params in experiments:
        try:
            model_path = f"{PATH_MODEL}cae_exp_{params['exp_id']}.pth"
            model = load_model(params, model_path, device)
            threshold_rows = evaluate(params, model, device)
            
            for row in threshold_rows:
                log_to_csv(row, LOG_PATH)

            print(f"  Лог обновлён (exp_id={params['exp_id']})\n")
            
        except Exception as e:
            print(f"  Ошибка в эксп {params['exp_id']}: {e}")