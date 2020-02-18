import Levenshtein as Lev

from utils.define import logger
from utils.label import label_to_string


def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length

def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length
