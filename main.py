import cv2
import numpy as np
import torch as t
import torchvision.transforms as transforms
from PIL import Image
from pyzbar.pyzbar import decode
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials 
from training.net import Net
from datetime import datetime

'''
    Doc:

    # -------------------------------------
    # OpenCV ------------------------------
    # -------------------------------------

    # 文字を表示
    cv2.putText(c_frame, TEXT, (X, Y), cv.FONT_HERSHEY_PLAIN, FONT_SIZE, COLOR, THICKNESS, cv.LINE_AA)

    # -------------------------------------
    # スプレッドシート ---------------------
    # -------------------------------------

    # セルの値を受け取る
    worksheet.acell('CELL').value

    # 値をセットする
    worksheet.update_cell(ROW, COL, VALUE)

    # 列の値を全て受け取る
    worksheet.col_values(1~)

    # -------------------------------------
'''

def check_deplicated(id):
    row = 0
    deplicated = False

    for i in range(len(ids)):
        if ids[i] == id:
            if dates[i] == today:
                print(f'{ids[i]} is measured already.')
                deplicated = True
    
    row = len(dates)+1 if deplicated is False else 0

    return row

def write_ss(row, id):
    # 値をセットする
    worksheet.update_cell(row, 1, today)
    worksheet.update_cell(row, 2, id)

    dates.append(today)
    ids.append(id)

def main():

    # device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    # DNNモデル読み込み
    model = Net(IN_C, H, W, MID_C, OUT_C, HIDDEN_UNITS, OUT_UNITS).to(device)
    model.load_state_dict(t.load(MODEL_PATH, map_location=device))
    model.eval()

    # 画像下処理
    transform = transforms.Compose([
                transforms.RandomResizedCrop(128, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
                transforms.Grayscale(),
                ])

    cap = cv2.VideoCapture(DEVICE_ID)

    # 顔検出モデル読込み
    detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    # height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:

        qr = decode(c_frame)

        cv2.rectangle(c_frame, (10, 10), (170, 50), BLACK, -1)
        cv2.putText(c_frame, 'ID: ', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

        if qr:
            id = qr[0].data.decode('utf-8')
            cv2.putText(c_frame, id, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
            row = check_deplicated(id)
            if row != 0:
                write_ss(row, id)

        # 画像の取得と顔の検出
        img = c_frame

        # 300x300に画像をリサイズ、画素値を調整
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # 顔検出の実行
        detector.setInput(blob)
        detections = detector.forward()

        # 検出結果の可視化
        img_copy = img.copy()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
    
            if confidence > CONFIDENCE:
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                margin_x = 0
                margin_y = 0

                face = img_copy[startY-margin_x:endY+margin_y, startX-margin_x:endX+margin_x]

                predicted = 0

                if face.shape[0]*face.shape[1]*face.shape[2] != 0:
                
                    face  = Image.fromarray(np.uint8(face))
                    face  = transform(face)

                    face  = face.view(1, face.shape[0], face.shape[1], face.shape[2])
                    output = model(face)

                    _, predicted = t.max(output.data, 1)

                if predicted == 1:
                    cv2.rectangle(img_copy, (startX, startY), (endX, endY), RED, thickness=2)
                else:
                    cv2.rectangle(img_copy, (startX, startY), (endX, endY), GREEN, thickness=2)

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

if __name__ == "__main__":

    # スプレッドシートの設定 ---------------------------------------------------------------------------

    # 2つのAPIを記述しないとリフレッシュトークンを3600秒毎に発行し続けなければならない
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

    # 認証情報設定：ダウンロードしたjsonファイル名をクレデンシャル変数に設定
    credentials = ServiceAccountCredentials.from_json_keyfile_name('key.json', scope)

    # OAuth2の資格情報を使用してGoogle APIにログイン
    gc = gspread.authorize(credentials)

    #共有設定したスプレッドシートキーを変数[SPREADSHEET_KEY]に格納する。
    SPREADSHEET_KEY = '1rm6spOIOGCce5aczzygqzaePFuHw2EAi8F1XM1jxnHI'

    #共有設定したスプレッドシートのシート1を開く
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    # -------------------------------------------------------------------------------------------------

    # DNN conditions ----------------------------------------------------------------------------------
    BLACK = (  0,   0,   0)
    WHITE = (255, 255, 255)
    RED   = (  0,   0, 255)
    GREEN = (  0, 255,   0)
    BLUE  = (255,   0,   0)
    # -------------------------------------------------------------------------------------------------

    # DNN conditions ----------------------------------------------------------------------------------
    MODEL_PATH    = 'model.pth'
    WINDOW_NAME   = 'Window'
    PROTOTXT_PATH = 'cascade_file/deploy.prototxt.txt'
    WEIGHTS_PATH  = 'cascade_file/res10_300x300_ssd_iter_140000.caffemodel'

    DEVICE_ID     = 0
    ESC_KEY       = 27   
    INTERVAL      = 33 
    CONFIDENCE    = 0.5

    IN_C = 1
    W = 128
    H = 128
    MID_C = 18
    OUT_C = 18
    HIDDEN_UNITS = 3011
    OUT_UNITS = 128
    # -------------------------------------------------------------------------------------------------

    today = datetime.now()
    today = today.strftime('%Y/%m/%d')

    # 日付とIDを取得
    dates = worksheet.col_values(1)
    ids  = worksheet.col_values(2)

    main()