import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.cae_without_latent import ConvolutionalAutoencoder
from data.dataset import VideoDataset
from data.preprocessing import VideoPreprocessor


DATA_PATH      = "/home/liza/Desktop/diplom/code/code_python/data/my_data/Train"
IMG_SIZE       = 128
BATCH_SIZE     = 16
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLDS     = [0.80, 0.90, 0.95, 0.99]


def build_dataloader() -> DataLoader:
    preprocessor = VideoPreprocessor(
        target_size=(IMG_SIZE, IMG_SIZE),
        quality_threshold=0.0,
    )
    train_files = []
    for root, _, files in os.walk(DATA_PATH):
        for file in sorted(files):
            if file.endswith((".tif", ".png", ".jpg")):
                train_files.append(os.path.join(root, file))

    print(f"Найдено файлов: {len(train_files)}")
    dataset = VideoDataset(
        train_files,
        frame_size=(IMG_SIZE, IMG_SIZE),
        preprocessor=preprocessor,
        mode="frame",
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

@torch.no_grad()
def _extract_features_from_model(model: ConvolutionalAutoencoder,
                                 loader: DataLoader) -> np.ndarray:
    model.eval()
    feats = []
    for batch in loader:
        x = batch[0].to(DEVICE) if isinstance(batch, (list, tuple)) else batch.to(DEVICE)
        h = model.encoder(x)
        feats.append(h.view(h.size(0), -1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def get_features(weights,features_cache,model) -> np.ndarray:
    """
    Возвращает матрицу признаков.
    Если кэш существует — грузит из файла (модель и данные не нужны).
    Иначе — извлекает через модель и сохраняет кэш.
    
    if os.path.exists(FEATURES_CACHE):
        features = np.load(FEATURES_CACHE)
        print(f"Признаки загружены из кэша: {FEATURES_CACHE}  {features.shape}")
        return features

    print("Кэш не найден — извлекаем признаки через модель...")
    """
    loader = build_dataloader()


    
    if weights and os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=DEVICE))
        print(f"Загружены веса: {weights}")
    else:
        print("Веса не найдены — используются случайные")

    features = _extract_features_from_model(model, loader)
    print(f"Матрица признаков: {features.shape}")

    os.makedirs(os.path.dirname(features_cache), exist_ok=True)
    np.save(features_cache, features)
    print(f"Признаки сохранены в кэш: {features_cache}")

    return features

def run_pca(features: np.ndarray, thresholds: list):

    X = features
    n_comp = min(X.shape[0], X.shape[1])

    pca = PCA(n_components=n_comp)
    pca.fit(X)

    explained  = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    recommendations = {}
    print("\nРекомендации по latent_dim:")
    for thr in thresholds:
        n = int(np.searchsorted(cumulative, thr) + 1)
        recommendations[thr] = n
        print(f"  {int(thr*100)}%  ->  latent_dim >= {n}")

    return explained, cumulative, recommendations



def plot_results(explained, cumulative, recommendations, save_path):
    thr_colors = {0.80: "green", 0.90: "steelblue", 0.95: "orange", 0.99: "red"}
    x_vals = np.arange(1, len(explained) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PCA-анализ латентного пространства автокодировщика", fontsize=14)

    # ── График 1: Scree plot (линейные оси) ──────────────────────────────────
    ax = axes[0]
    top = min(260, len(explained))
    bar_x = np.arange(200, top + 1)
    ax.bar(bar_x, explained[200:top + 1] * 100, color="steelblue", alpha=0.8)

    # Локоть: минимум второй производной (самый крутой перегиб)
    elbow = int(np.argmin(np.diff(explained[200:top + 1]))) + 200
    ax.axvline(elbow, color="orange", linestyle="--", linewidth=1.5,
               label=f"Локоть ≈ {elbow}")

    ax.set_title("Scree Plot (компоненты 200–260)")
    ax.set_xlabel("Компонента PCA")
    ax.set_ylabel("Дисперсия, %")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── График 2: Дисперсия в лог-лог шкале ─────────────────────────────────
    ax = axes[1]
    ax.loglog(x_vals, explained * 100, color="mediumpurple", linewidth=2)

    # Пороговые линии (по кумулятивной дисперсии → вертикальные метки)
    for thr, n in recommendations.items():
        c = thr_colors[thr]
        ax.axvline(n, color=c, linestyle="--", linewidth=1.2, alpha=0.85,
                   label=f"{int(thr * 100)}% → d={n}")
        # Подпись значения дисперсии в точке
        if n <= len(explained):
            ax.scatter([n], [explained[n - 1] * 100], color=c, s=50, zorder=5)

    ax.set_title("Дисперсия (лог-лог шкала)")
    ax.set_xlabel("Компонента PCA  (лог)")
    ax.set_ylabel("Дисперсия, % (лог)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nГрафик сохранён: {save_path}")
    plt.show()


def main():
    """
    id = [5,10,15,20,25,30]
    kernel_sizes = [3,5,4,7,5,4]
    num_layers = [5,5,5,5,5,5]
    channels_layers =[64,64,32,32,32,16]
    """
    id = [5,4,14]
    kernel_sizes = [3,3,4]
    num_layers = [5,4,4]
    channels_layers =[32,32,32]

    for i in range(0, len(id)):

        weights = f"outputs/models/num2/cae_exp_{id[i]}.pth"  
        save_path = f"outputs/pca/pca{id[i]}_analysis.png"
        features_cache = f"outputs/pca/pca{id[i]}_features_cache.npy"
        print(f"Устройство: {DEVICE}\n")
        model = ConvolutionalAutoencoder(input_channels=1,
                                     kernel_size=kernel_sizes[i],
                                     channels_out_1_layer = channels_layers[i],
                                     num_layers = num_layers[i],
                                     in_size =128).to(DEVICE)
        features = get_features(weights,features_cache,model)
        print("\nPCA...")
        explained, cumulative, recommendations = run_pca(features, THRESHOLDS)
        plot_results(explained, cumulative, recommendations, save_path)

if __name__ == "__main__":
    main()