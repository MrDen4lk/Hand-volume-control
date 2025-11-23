import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch


def generate_heatmaps(keypoints, height, width, sigma=2, downsample=1):
    """
    keypoints: [N, 3] np.array (x, y, v) в пикселях
    height, width: размеры входного изображения
    downsample: коэффициент уменьшения
    """
    # Размеры хитмапа
    h_hm = height // downsample
    w_hm = width // downsample

    num_points = keypoints.shape[0]
    heatmaps = np.zeros((num_points, h_hm, w_hm), dtype=np.float32)

    for i, (x, y, v) in enumerate(keypoints):
        if v < 0.5:
            continue

        # Координаты на хитмапе
        x_hm = x / downsample
        y_hm = y / downsample

        # 1. Вычисляем границы квадрата гауссианы (ROI - Region of Interest)
        # 3-sigma правило покрывает 99.7% гауссианы
        tmp_size = sigma * 3
        ul_x = int(np.floor(x_hm - tmp_size))
        ul_y = int(np.floor(y_hm - tmp_size))
        br_x = int(np.ceil(x_hm + tmp_size))
        br_y = int(np.ceil(y_hm + tmp_size))

        # 2. Находим ПЕРЕСЕЧЕНИЕ квадрата гауссианы и границ картинки
        # Координаты начала и конца на ХИТМАПЕ (ограничены размерами картинки)
        start_x = max(0, ul_x)
        start_y = max(0, ul_y)
        end_x = min(w_hm, br_x + 1)  # +1 т.к. slicing не включает правую границу
        end_y = min(h_hm, br_y + 1)

        # Если пересечения нет (точка далеко за краем) - пропускаем
        if start_x >= end_x or start_y >= end_y:
            continue

        # 3. Вычисляем размеры пересечения
        box_w = end_x - start_x
        box_h = end_y - start_y

        # 4. Вычисляем координаты внутри самой ГАУССИАНЫ
        # Если ul_x < 0, значит мы отрезали левую часть гауссианы,
        # поэтому начинаем генерировать её не с 0, а со смещением.
        g_x_start = max(0, -ul_x)
        g_y_start = max(0, -ul_y)

        # 5. Генерируем сетку для Гауссианы
        # Нам нужны координаты относительно центра точки (x_hm, y_hm)
        # Создаем диапазон координат, соответствующих вырезанной области
        range_x = np.arange(start_x, end_x)
        range_y = np.arange(start_y, end_y)
        xx, yy = np.meshgrid(range_x, range_y)

        d2 = (xx - x_hm) ** 2 + (yy - y_hm) ** 2
        g = np.exp(-d2 / (2 * sigma ** 2))

        # 6. Наложение (Максимум, чтобы пятна не складывались яркостью, а перекрывали)
        # Размеры g и heatmaps[slice] теперь гарантированно совпадают (box_h, box_w)
        heatmaps[i, start_y:end_y, start_x:end_x] = np.maximum(
            heatmaps[i, start_y:end_y, start_x:end_x],
            g
        )

    return torch.from_numpy(heatmaps)


def visualize_heatmap(image, heatmaps):
    """
    Функцию визуализации можно оставить почти без изменений,
    она выглядит рабочей, но добавил пару проверок на всякий случай.
    """
    cols = 4  # Уменьшил кол-во колонок, чтобы картинки были крупнее
    rows = 2
    fig = plt.figure(figsize=(15, 8))
    k = 0

    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    threshold = 0.001

    for j in range(rows):
        for i in range(cols):
            if k >= len(image):
                break
            ax = fig.add_subplot(rows, cols, j * cols + i + 1)
            ax.axis('off')

            img_tensor = image[k]
            heatmap_tensor = heatmaps[k]

            # [3, H, W] -> [H, W, 3]
            img_np = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))

            # Денормализация для красоты
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            ax.imshow(img_np)

            # Сумма хитмапов для отображения всех точек сразу
            heatmap_sum = heatmap_tensor.max(dim=0)[0].cpu().numpy()

            # Ресайз хитмапа обратно к размеру картинки для наложения
            heatmap_sum_resized = np.array(
                Image.fromarray(heatmap_sum).resize((img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR)
            )
            ax.imshow(heatmap_sum_resized, cmap='jet', alpha=0.5)

            # Восстановление координат
            kp_from_heatmap = []

            # Важно: scale считаем от реальных размеров тензоров
            scale_x = img_tensor.shape[2] / heatmap_tensor.shape[2]
            scale_y = img_tensor.shape[1] / heatmap_tensor.shape[1]

            for h in heatmap_tensor:
                h_np = h.cpu().numpy()
                max_val = h_np.max()

                if max_val < threshold:
                    kp_from_heatmap.append((0, 0))
                else:
                    y, x = np.unravel_index(np.argmax(h_np), h_np.shape)
                    # Умножаем координаты на downsample (scale)
                    kp_from_heatmap.append((x * scale_x, y * scale_y))

            kp_from_heatmap = np.array(kp_from_heatmap)

            # Скелет
            for start, end in skeleton:
                if (start < len(kp_from_heatmap) and end < len(kp_from_heatmap)):
                    p1 = kp_from_heatmap[start]
                    p2 = kp_from_heatmap[end]
                    # Рисуем только если обе точки найдены (не 0,0)
                    if np.sum(p1) > 0 and np.sum(p2) > 0:
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='lime', linewidth=1.5)

            # Точки
            for x, y in kp_from_heatmap:
                if x > 0 and y > 0:
                    ax.scatter(x, y, s=10, c='cyan', marker='o')

            k += 1

    plt.tight_layout()
    plt.show()