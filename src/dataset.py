import os
from torch.utils.data import Dataset
import numpy as np
import cv2

from utils import generate_heatmaps


class HandPointDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, split='train', transform=None, sigma=2, downsample=4):
        # downsample=4, так как выход модели обычно в 4 раза меньше входа (stride 4)
        self.img_dir = os.path.join(img_dir, split)
        self.lbl_dir = os.path.join(lbl_dir, split)
        self.sigma = sigma
        self.downsample = downsample
        self.transform = transform

        self.image_files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Читаем картинку
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig, _ = image.shape

        # Читаем метки
        base_name, _ = os.path.splitext(img_name)
        lbl_path = os.path.join(self.lbl_dir, base_name + ".txt")

        with open(lbl_path, "r") as f:
            # формат: class x c y w h p1x p1y v1 ...
            data = list(map(float, f.read().strip().split()))
            # Пропускаем первые 5 чисел (box), берем точки
            keypoints_data = data[5:]

        # Превращаем в N x 3 (x, y, visibility)
        keypoints = np.array(keypoints_data, dtype=np.float32).reshape(-1, 3)

        # Денормализация: (0..1) -> (0..Width)
        if keypoints[:, :2].max() <= 1.0:
            keypoints[:, 0] *= w_orig
            keypoints[:, 1] *= h_orig

        # Аугментация
        if self.transform:
            # Albumentations требует список списков для keypoints
            transformed = self.transform(image=image, keypoints=keypoints.tolist())
            image = transformed['image']
            # Возвращаем трансформированные точки обратно в numpy
            keypoints = np.array(transformed['keypoints'], dtype=np.float32)

        # Генерируем хитмапы
        _, h_new, w_new = image.shape

        heatmaps = generate_heatmaps(keypoints, h_new, w_new, sigma=self.sigma, downsample=self.downsample)

        return image, heatmaps