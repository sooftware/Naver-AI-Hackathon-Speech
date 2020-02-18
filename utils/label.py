from utils.define import EOS_token, index2char


def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

# 정답을 리스트 형식으로 담아주는 함수
def get_script(filepath, bos_id, eos_id, target_dict):
    # key : 41_0508_171_0_08412_03.script 중 41_0508_171_0_08412_03 -> label
    key = filepath.split('/')[-1].split('.')[0]
    # 41_0508_171_0_08412_03 에 해당하는 label 텍스트일 듯

    script = target_dict[key.split('\\')[1]]
    # 텍스트를 ' ' 기준으로 나눈다 -> 10 268 10207 와 같이 레이블 되어 있으니까!!
    tokens = script.split(' ')

    # result를 담을 리스트 초기화
    result = list()

    # result에 bos_id로 시작을 표시하는 듯
    # Begin Of Script 일 듯
    result.append(bos_id)

    # 나눈 token들을 result에 추가하는 듯
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    # 마지막 End Of Script 표시를 해주는 듯
    result.append(eos_id)
    return result