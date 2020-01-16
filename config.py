from easydict import EasyDict as edict
import mxnet as mx

config = edict()
config.DATASET_NAME='nwpu'
config.BASE_MODEL_NAME='resnet101_v1'
config.MODEL_NAME='resnet101_v1_eam'
assert config.DATASET_NAME.lower() in ['aid','aid_0.2','nwpu','nwpu_0.1','ucmerced','whu19']
assert config.BASE_MODEL_NAME in ['resnet50_v1','resnet101_v1']
assert config.BASE_MODEL_NAME in ['resnet50_v1','resnet50_v1_eam','resnet101_v1','resnet101_v1_eam']
######################################################################################
config.CTX = [mx.gpu(0)]
config.LR = 0.001
config.BATCH_SIZE=16
config.NUM_EPOCHES=100
config.DTYPE='float32'
config.WORKERS=20
config.WEIGHT_DECAY = 5e-4
#######################################################################################
if config.DATASET_NAME.lower()=='aid':
    config.NUM_CLASSES=30
    config.DATA_PATH = './dataset/AID'
    config.TEST_SIZE = 0.5
elif config.DATASET_NAME.lower()=='aid_0.2':
    config.NUM_CLASSES=30
    config.DATA_PATH = './dataset/AID'
    config.TEST_SIZE = 0.8
elif config.DATASET_NAME.lower()=='nwpu':
    config.NUM_CLASSES = 45
    config.DATA_PATH = '~/project/paper/NWPU45_experiment/NWPU-RESISC45/'
    config.TEST_SIZE = 0.8
elif config.DATASET_NAME.lower()=='ucmerced':
    config.NUM_CLASSES = 21
    config.DATA_PATH = './dataset/UCMerced/'
    config.TEST_SIZE=0.2
elif config.DATASET_NAME.lower() == "whu19":
    config.NUM_CLASSES = 19
    config.DATA_PATH = './dataset/WHU-RS19/'
    config.TEST_SIZE = 0.4
else:
    print('数据集填写错误！')
if __name__ == '__main__':
    print(config)