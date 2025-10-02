import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

def generate_heatmaps(keypoints, height, width, sigma=2, downsample=1):
    num_points = keypoints.shape[0]
    h, w = height // downsample, width // downsample
    heatmaps = np.zeros((num_points, h, w), dtype=np.float32)

    keypoints[keypoints[:, 0] < 0] = 0
    keypoints[keypoints[:, 1] < 0] = 0
    keypoints[keypoints[:, 2] == 0] = 0

    #keypoints[:, 0] /= width  # width
    #keypoints[:, 1] /= height  # height

    for i, (x, y, v) in enumerate(keypoints.numpy()):
        if v == 0:
            continue

        # Исправляем масштабирование координат
        x = x * (w - 1)
        y = y * (h - 1)

        ul = int(np.floor(x - 3 * sigma)), int(np.floor(y - 3 * sigma))
        br = int(np.ceil(x + 3 * sigma)), int(np.ceil(y + 3 * sigma))

        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
            continue

        x_range = np.arange(ul[0], br[0] + 1)
        y_range = np.arange(ul[1], br[1] + 1)
        xx, yy = np.meshgrid(x_range, y_range)

        d = (xx - x) ** 2 + (yy - y) ** 2
        g = np.exp(-d / (2 * sigma ** 2))

        xx_idx = np.clip(x_range, 0, w - 1)
        yy_idx = np.clip(y_range, 0, h - 1)
        heatmaps[i][yy_idx[:, None], xx_idx] = np.maximum(
            heatmaps[i][yy_idx[:, None], xx_idx], g[:len(yy_idx), :len(xx_idx)]
        )

    return torch.from_numpy(heatmaps)


def visualize_heatmap(image, heatmaps):
    cols = 8
    rows = 2
    fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
    k = 0

    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    threshold = 0.001  # порог для отсутствия ключевой точки

    for j in range(rows):
        for i in range(cols):
            if k >= len(image):
                break
            ax = fig.add_subplot(rows, cols, j * cols + i + 1)
            ax.axis('off')

            # Берём k-е изображение и heatmaps
            img_tensor = image[k]
            heatmap_tensor = heatmaps[k]

            # Преобразуем изображение в numpy
            img_np = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))

            # Убираем нормализацию (если нужно)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            ax.imshow(img_np)

            # Правильно берём максимум по ключевым точкам
            heatmap_sum = heatmap_tensor.max(dim=0)[0].cpu().numpy()

            # Масштабируем heatmap к размеру изображения
            heatmap_sum_resized = np.array(
                Image.fromarray(heatmap_sum).resize((img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR)
            )

            ax.imshow(heatmap_sum_resized, cmap='jet', alpha=0.5)

            # Извлекаем координаты ключевых точек из heatmap
            kp_from_heatmap = []
            scale_x = img_np.shape[1] / heatmap_tensor.shape[2]
            scale_y = img_np.shape[0] / heatmap_tensor.shape[1]

            for h in heatmap_tensor:
                h_np = h.cpu().numpy()
                max_val = h_np.max()

                if max_val < threshold:
                    kp_from_heatmap.append((0, 0))  # точка отсутствует
                else:
                    y, x = np.unravel_index(np.argmax(h_np), h_np.shape)
                    kp_from_heatmap.append((x * scale_x, y * scale_y))

            kp_from_heatmap = np.array(kp_from_heatmap)

            # Рисуем скелет
            for start, end in skeleton:
                if (start < len(kp_from_heatmap) and end < len(kp_from_heatmap) and
                        not (kp_from_heatmap[start] == (0, 0)).all() and
                        not (kp_from_heatmap[end] == (0, 0)).all()):
                    x_vals = [kp_from_heatmap[start][0], kp_from_heatmap[end][0]]
                    y_vals = [kp_from_heatmap[start][1], kp_from_heatmap[end][1]]
                    ax.plot(x_vals, y_vals, c='lime', linewidth=1.5)

            # heatmap точки
            for x, y in kp_from_heatmap:
                if (x, y) != (0, 0):
                    ax.scatter(x, y, s=10, c='cyan', marker='o')

            k += 1

    plt.tight_layout()
    plt.show()