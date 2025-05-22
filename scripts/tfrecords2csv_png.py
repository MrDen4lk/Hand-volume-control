import tensorflow as tf
import os
import csv
import cv2
from pathlib import Path

IMG_SIZE = 256

def unnormalize_and_scale_labels(keypoints, max_val):
    keypoints = tf.math.scalar_mul(max_val, tf.math.add(keypoints, 0.5))
    return keypoints

# get params from tfrecords
def parse_tfrecord(example_proto):
    features = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([42], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    image = tf.io.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
    labels = tf.reshape(parsed['label'], [21, 2])
    labels = unnormalize_and_scale_labels(labels, IMG_SIZE)
    return image.numpy(), labels.numpy()

def tfrecords_to_csv(tfrecord_files, csv_output_path, output_img_dir):
    header = []
    for i in range(21):
        header.append(f'x{i}')
        header.append(f'y{i}')

    with open(csv_output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        idx = 0
        for tf_file in tfrecord_files:
            dataset = tf.data.TFRecordDataset(tf_file)
            for raw_record in dataset:
                image, labels = parse_tfrecord(raw_record)
                # save image
                img_path = os.path.join(output_img_dir, f"img_{idx}.png")
                cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # save points
                writer.writerow(labels.flatten())
                idx += 1

    print(f"CSV&PNG сохранён в {csv_output_path}")
    
# Путь к текущему скрипту
current_file = os.getcwd()

# positions of points from tfrecords to csv
tfrecord_files = [f"{current_file}/data/archive/train/training_data_{i}.tfrecords" for i in range(30)]
csv_output = f"{current_file}/data/archive/train/keypoints_labels_train.csv"
img_output = f"{current_file}/data/archive/train/train_images"

tfrecords_to_csv(tfrecord_files, csv_output, img_output)

tfrecord_files = [f"{current_file}/data/archive/test/testing_data_{i}.tfrecords" for i in range(10)]
csv_output = f"{current_file}/data/archive/test/keypoints_labels_test.csv"
img_output = f"{current_file}/data/archive/test/test_images"

tfrecords_to_csv(tfrecord_files, csv_output, img_output)

tfrecord_files = [f"{current_file}/data/archive/valid/validation_data_{i}.tfrecords" for i in range(10)]
csv_output = f"{current_file}/data/archive/valid/keypoints_labels_valid.csv"
img_output = f"{current_file}/data/archive/valid/valid_images"

tfrecords_to_csv(tfrecord_files, csv_output, img_output)