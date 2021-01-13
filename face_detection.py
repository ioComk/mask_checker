import cv2

import numpy as np
import torch as t
import torchvision.transforms as transforms
from PIL import Image
from net import Net
# from pyzbar.pyzbar import decode

MODEL_PATH    = 'model.pth'
WINDOW_NAME   = 'Window'
PROTOTXT_PATH = 'face_detection_model/deploy.prototxt.txt'
WEIGHTS_PATH  = 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'

DEVICE_ID     = 0
ESC_KEY       = 27   
INTERVAL      = 33 
CONFIDENCE    = 0.5

IN_C = 3
W = 256
H = 256
MID_C = 14
OUT_C = 32
HIDDEN_UNITS = 2929
OUT_UNITS = 228

if __name__ == '__main__':

    # device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    # DNNモデル読み込み
    model = Net(IN_C, H, W, MID_C, OUT_C, HIDDEN_UNITS, OUT_UNITS).to(device)
    model.load_state_dict(t.load(MODEL_PATH, map_location=device))
    model.eval()

    # 画像下処理
    transform = transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
                ])

    # 顔検出モデル
    detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:
        
        # d = decode(c_frame)
        # if d:
        #     print(d[0].data.decode('utf-8'))

        # 画像の取得と顔の検出
        img = c_frame

        # 300x300に画像をリサイズ、画素値を調整
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # 顔検出の実行
        detector.setInput(blob)
        detections = detector.forward()

        trim = None

        # 検出結果の可視化
        img_copy = img.copy()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
    
            if confidence > CONFIDENCE:
                
                color_red   = (0, 0, 225)
                color_green = (0, 255, 0)

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                margin_x = 0
                margin_y = 0

                face = img_copy[startY-margin_x:endY+margin_y, startX-margin_x:endX+margin_x]

                # cv2.imshow('face', face)

                predicted = 0

                if face.shape[0]*face.shape[1]*face.shape[2] != 0:
                
                    face  = Image.fromarray(np.uint8(face))
                    face  = transform(face)

                    face  = face.view(1, face.shape[0], face.shape[1], face.shape[2])
                    # print(face.shape)
                    output = model(face)

                    _, predicted = t.max(output.data, 1)

                if predicted == 0:
                    cv2.rectangle(img_copy, (startX, startY), (endX, endY), color_red, thickness=2)
                else:
                    cv2.rectangle(img_copy, (startX, startY), (endX, endY), color_green, thickness=2)

        # フレーム表示
        cv2.imshow(WINDOW_NAME, img_copy)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()