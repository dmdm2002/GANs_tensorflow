import datetime


class Options(object):
    def __init__(self):
        self.epochs = 300
        self.lr = 2e-4
        self.beta_1 = 0.5
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.BatchSZ = 1
        self.BUFFER_SIZE = 400
        self.OUTPUT_CHANNELS = 3
        self.LAMBDA = 100
        self.root = 'E:/Iris_dataset/nd_labeling_iris_data/Cycle_PNG/1-fold'

        self.A_cnt = 2277
        self.B_cnt = 2277

        self.OUTPUT_DIR = 'E:/backup/PIX2PIX/ND/1-fold/{date:%Y-%m-%d_%H%M%S}/'.format(date=datetime.datetime.now())
        self.OUTPUT_DIR_CKP = f'{self.OUTPUT_DIR}/checkpoints'
        self.OUTPUT_DIR_SAMPLE = f'{self.OUTPUT_DIR}/sampling'
        self.OUTPUT_DIR_TEST = f'E:/backup/PIX2PIX/ND/1-fold/2021-12-31_140013/test'
        self.OUTPUT_DIR_LOSS = f'{self.OUTPUT_DIR}/loss'
