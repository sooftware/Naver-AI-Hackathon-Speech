import math
from torch.utils.data import Dataset


def split_dataset(config, wav_paths, script_paths, valid_ratio=0.05, target_dict = dict()):
    train_loader_count = config.workers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(
                                        wav_paths[train_begin_raw_id:train_end_raw_id],
                                        script_paths[train_begin_raw_id:train_end_raw_id],
                                        SOS_token, EOS_token, target_dict))
        train_begin = train_end

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token, target_dict)

    return train_batch_num, train_dataset_list, valid_dataset

class BaseDataset(Dataset):
    # wav_paths : wav_path가 모여있는 리스트
    # script_paths : script_path가 모여있는 리스트 script == label
    # bos_id : Begin Of Script -> script의 시작을 표시하는 Number
    # eos_id : End Of Script -> script의 끝을 표시하는 Number
    def __init__(self, wav_paths, script_paths, bos_id = 1307, eos_id = 1308, target_dict = dict()):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.target_dict = target_dict

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        # 음성데이터에 대한 feature를 feat에 저장 -> tensor 형식
        feat = get_librosa_mfcc(self.wav_paths[idx], n_mfcc = 40)
        # 리스트 형식으로 label을 저장
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id, self.target_dict)
        return feat, script