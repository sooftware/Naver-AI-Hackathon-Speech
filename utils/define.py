import logging, sys
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
from utils.load import load_label
char2index, index2char = load_label('./data/hackathon.labels')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']
DATASET_PATH = './data/'
target_dict = dict()