import spidev

class  ADConverterClass:                                # AD Converter（MCP3208）から値を取得するクラス
    def __init__(self, ref_volts, ch):                  # コンストラクタ
        self.ref_volts = ref_volts                      # 電圧セット
        self.spi = spidev.SpiDev()                      # SpiDevのインスタンスを作成
        self.spi.open(0,0)                              # 0：SPI0、0：CE0
        self.spi.max_speed_hz = 1000000                 # 1MHz SPIのバージョンアップによりこの指定をしないと動かない!
        self.ch = ch                                    # センサのMCP3208への入力チャンネル

    def get_voltage(self, ch):                          # A/Dコンバータ（MCP3208）で電圧を取得する
        raw = self.spi.xfer2([((0b1000+ch)>>2)+0b100,((0b1000+ch)&0b0011)<<6,0])  # Din(RasPi→MCP3208）
        raw2 = ((raw[1]&0b1111) << 8) + raw[2]          # Dout（MCP3208→RasPi）
        volts = (raw2 * self.ref_volts ) / float(4095)  # 取得した値を電圧に変換する（12bitなので4095で割る）
        volts = round(volts,4)                          # 電圧を4桁に四捨五入する
        return volts                                    # 電圧を返す

    def get_dist(self):
        volts = self.get_voltage(self.ch)              #MCP3208のGP2Y0A21を入力したチャンネルの電圧を取得する
        dist_inv = 0.0502*volts - 0.0123                #実測から求めた電圧と1/距離[cm]の近似直線
        dist = 1/dist_inv                               #1/距離[1/cm]を距離[cm]に変換
        return dist                                     #距離を返す

    def Cleanup(self):                                  # 終了処理
        self.spi.close()                                # SpiDevの終了処理