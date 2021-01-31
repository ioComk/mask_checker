import cv2
import numpy as np
import torch as t
import torchvision.transforms as transforms
from PIL import Image
from pyzbar.pyzbar import decode
import gspread
from oauth2client.service_account import ServiceAccountCredentials 
from training.net import Net
from datetime import datetime
import picamera
import picamera.array
import smbus
import time
import RPi.GPIO as GPIO
from sensor import ADConverterClass

def read_temp(mode):
    buf = bus.read_i2c_block_data(addr, mode, 3)
    tmp = (buf[1] << 8) + buf[0]
    temperature = ((tmp*2.0) - 27315.0) / (100.0)
        
    return temperature

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

def write_ss(row, id, temp):
    # 値をセットする
    worksheet.update_cell(row, 1, today)
    worksheet.update_cell(row, 2, id)
    worksheet.update_cell(row, 3, temp)

    dates.append(today)
    ids.append(id)
    temps.append(temp)

def main():

    dist_sensor = ADConverterClass(ref_volts=VOLTS, ch=CH)

    # Servo setup
    servo_pin = 18
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servo_pin, GPIO.OUT)
    servo = GPIO.PWM(servo_pin, 50)
    servo.start(0)

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

    # 顔検出モデル読込み
    detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    is_register = False
    check_qr = False
    counter = 0
    id = 0
    sensor = [0,0]
    hand = False

    begin_time = 0

    # 検温検知するためのリスト
    temp_old = [0,0,0,0,0,0,0]

    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            camera.resolution = (1312, 800)
            camera.rotation = -90

            # メインループ  -------------------------------------
            while True:
                # stream.arrayにRGBの順で映像データを格納
                camera.capture(stream, 'bgr', use_video_port=True)

                dist = dist_sensor.get_dist()

                # 温度センサ読み取り
                temp = read_temp(0x7) + 4

                sensor[1] = sensor[0]

                # 手をかざした瞬間に立ち上がるフラグ
                sensor[0] = 1 if dist < 8 else 0

                if sensor[0] == 1 and sensor[1] == 0:
                    begin_time = time.time()
                    hand = True

                # print(dist)

                if hand:
                    if time.time() - begin_time <= 1:
                        servo.start(0)
                        servo.ChangeDutyCycle(7)
                    elif time.time() - begin_time > 1:
                        servo.ChangeDutyCycle(5)
                        # servo.stop()
                        hand = False
                        begin_time = 0
                
                # QR読み取り
                qr = decode(stream.array)

                cv2.rectangle(stream.array, (10, 10), (450, 90), BLACK, -1)
                cv2.putText(stream.array, 'ID: ', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, WHITE, 3, cv2.LINE_AA)

                if qr:
                    if check_qr == False:
                        id = qr[0].data.decode('utf-8')
                        check_qr = True
                temp_diff = 0

                for i in range(len(temp_old)):
                    temp_diff += abs(temp - temp_old[i])

                # QRコードがかざされたら
                if check_qr:
                    cv2.putText(stream.array, id, (130, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, WHITE, 3, cv2.LINE_AA)
                    txt = str(round(temp,1))+' [Celsius]'
                    cv2.putText(stream.array, txt, (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 2, cv2.LINE_AA)
                    if temp >= 34 and temp_diff <= 20:
                        row = check_deplicated(id)
                        if row != 0:
                            # cv2.putText(stream.array, 'Registering now...', (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2, cv2.LINE_AA)
                            write_ss(row, id, temp)
                            check_qr = False
                        elif row == 0:
                            # cv2.putText(stream.array, 'Already registered.', (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2, cv2.LINE_AA)
                            check_qr = False

                # 300x300に画像をリサイズ、画素値を調整
                (h, w) = stream.array.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(stream.array, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0))
                # 顔検出の実行
                detector.setInput(blob)
                detections = detector.forward()

                margin_x = [0, 0]
                margin_y = [0, 0]

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > CONFIDENCE:

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype('int')

                        face = stream.array[startY-margin_y[0]:endY+margin_y[1], startX-margin_x[0]:endX+margin_x[1]]

                        predicted = 0

                        if face.shape[0]*face.shape[1] != 0:
                        
                            face  = Image.fromarray(np.uint8(face))
                            face  = transform(face)
                            face  = face.view(1, face.shape[0], face.shape[1], face.shape[2])

                            output = model(face)

                            _, predicted = t.max(output.data, 1)

                        if predicted == 1:
                            cv2.rectangle(stream.array, (startX-margin_x[0], startY-margin_y[0]), (endX+margin_x[1], endY+margin_y[1]), RED,         thickness=2)
                        else:
                            cv2.rectangle(stream.array, (startX-margin_x[0], startY-margin_y[0]), (endX+margin_x[1], endY+margin_y[1]), GREEN,       thickness=2)

                temp_old[counter] = temp

                counter += 1
                if counter >= len(temp_old):
                    counter = 0

                # フレーム表示
                cv2.imshow(WINDOW_NAME, stream.array)

                # Escキーで終了
                key = cv2.waitKey(INTERVAL)
                if key == ESC_KEY:
                    break

                # streamをリセット
                stream.seek(0)
                stream.truncate()

    # 終了処理
    cv2.destroyAllWindows()
    servo.stop()
    dist_sensor.Cleanup()
    GPIO.cleanup()

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
    Q_KEY         = 113
    INTERVAL      = 1 
    CONFIDENCE    = 0.5

    IN_C = 1
    W = 128
    H = 128
    MID_C = 18
    OUT_C = 18
    HIDDEN_UNITS = 3011
    OUT_UNITS = 128
    # -------------------------------------------------------------------------------------------------
    CH = 0
    VOLTS = 5

    # 温度センサ
    addr = 0x5a
    bus = smbus.SMBus(1)

    today = datetime.now()
    today = today.strftime('%Y/%m/%d')

    # 日付とIDを取得
    dates = worksheet.col_values(1)
    ids  = worksheet.col_values(2)
    temps = worksheet.col_values(3)

    main()