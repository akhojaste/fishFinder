TRAIN_ROOT = "./DATA/train"
TEST_ROOT = "./DATA/test"
CLASSES = ['carp', 'largemouth_bass', 'pike', 'crappie']
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 160 #MobileNetV2
CROP_LENGTH = 60

LAYERS_TO_TRAIN = 0
L2_REG = 1

BATCH_SIZE = 32
EPOCH = 200
TRAIN_TEST_SPLIT = 0.2

MEAN = [128.99174473, 128.60456352, 118.13461311]
STD = [70.91102704, 68.93907923, 73.85932088]