import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import cv2

PROTOTXT_PATH = 'cascade_file/deploy.prototxt.txt'
WEIGHTS_PATH  = 'cascade_file/res10_300x300_ssd_iter_140000.caffemodel'

INPUT_PATH    = 'training/dataset/dev/train/without_mask/'
OUTPUT_PATH   = 'faces/without_mask/'

DEVICE_ID     = 0
ESC_KEY       = 27
INTERVAL      = 33
CONFIDENCE    = 0.5

if __name__ == '__main__':

    img_folder = Path(INPUT_PATH)

    # 顔検出モデル
    detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)

    for img_path in tqdm(img_folder.iterdir()):

        img = Image.open(img_path)
        # PNG形式の場合，アルファチャネルを削除
        img = img.convert('RGB')
        # numpy形式に変換
        img = np.asarray(img)

        h, w, c = img.shape

        # 300x300に画像をリサイズ、画素値を調整
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 400)), 1.0,
                                     (300, 400), (104.0, 177.0, 123.0))
        # 顔検出の実行
        detector.setInput(blob)
        detections = detector.forward()

        # 検出結果の可視化
        img_copy = img.copy()
        face = None

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = img_copy[startY-10:endY+10, startX-20:endX+20]
        
        if face is not None and face.shape[0]*face.shape[1]*face.shape[2] != 0:
            pil_img = Image.fromarray(face)
            pil_img.save(OUTPUT_PATH+img_path.stem+'.jpg')