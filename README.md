# Naver-Hackathon-2019-Speech-Team_Kai.Lib  
 네이버 2019 해커톤 - Speech Team Kai.Lib  
 Team_Kai.Lib  
 Rank : 12
 CRR : 75.33%  
![Kai](https://postfiles.pstatic.net/MjAxOTEwMjVfMTUy/MDAxNTcyMDAxNDk4MDQz.wbfC31mT1MSH9W5HsJ_vusdEwVxNg3n9J8kkKxzCNFcg.TwNAlQHP0fXKxZ0VfbMbQD8e6oOAZniuXOR7Ufrx8-8g.PNG.sooftware/Kai.PNG?type=w773)  
Korean-Speech-Recognition Using Pytorch.   
Naver 2019 Hackathon - Speech   Team Kai.Lib   
대회 결선 종료 후 소스코드 업로드 예정  
## Model
![seq2seq_with_mfcc](https://postfiles.pstatic.net/MjAxOTEwMjZfMjU0/MDAxNTcyMDc4NjQ2NjQ0.va5bywkjqAHcmx8eDoh2jofMqA4L85k8c2fC2Y06kb4g.dbx8_Rqte4YdXPjTHGEkMUD48NpwIy50M2YOCnX95cYg.PNG.sooftware/seq2seq%EA%B5%AC%EC%A1%B0.PNG?type=w773)  
- Model Architecture : Seq2seq with Convolution Layer  
- Model based on IBM Pytorch-Seq2seq  
## Data
네이버에서 제공한 100시간 데이터 사용
### Data format
* 음성 데이터 : 16bit, mono 16k sampling PCM, WAV audio
* 정답 스크립트 : 코드와 함께 제공되는 Character kevek ductationary를 통해서 인덱스로 변환된 정답
```
"네 괜찮습니다." => "715 662 127 76 396 337 669 662"
```
### Dataset folder structure
```
* DATASET-ROOT-FOLDER
|--train
   |--train_data
      +--data_list.csv
      +--a.wav, b.wav, c.wav ...
   +--train_label
|--test
|--test_submit
```
* data_list.csv
```
<wav-filename>,<script-filename>
wav_001.wav,wav_001.label
wav_002.wav,wav_002.label
wav_003.wav,wav_003.label
wav_004.wav,wav_004.label
wav_005.wav,wav_005.label
```
* train_label
```
<filename>,<script labels>
wav_001,574 268 662 675 785 661 662 
wav_002,715 662 545 566 441 337 669 662 
wav_003,628 9 625 662 408 690 2 125 71 662 220 630 610 749 62 661 123 662
wav_004,384 638 610 533 784 662 130 602 662 179 192 661 123 662  
...
```
## Score
```
CRR = (1.0 - CER) * 100.0
```
* CRR : Character Recognition Rate
* CER : Character Error Rate based on Edit Distance
![crr](https://postfiles.pstatic.net/MjAxOTEwMjVfODUg/MDAxNTcyMDEwNjg4NDk2.LlS0LaNboVQ7HuVvMqkKHljALDmvtb2WbHx2rkpojRIg.tUZ-dCceP_uloYsX2qTKFdNKEB72jWjedYMet2WiMbMg.PNG.sooftware/image.png?type=w773)
## Feature
![MFCC](https://postfiles.pstatic.net/MjAxOTEwMjdfMTEx/MDAxNTcyMTI2NDEyMTQw.pJz6E6fK0SbPebKIgk_s0Kaw3pg2h4eyPO0nwjYCLPcg.zkb5BMsEEAMpmoD7eKJkW98N4JeGm4d_W98raAa5luUg.PNG.sooftware/image.png?type=w773)   
* MFCC (Mel-Frequency Cepstral Coefficient)
  + frame length : 21ms  
  + overlap : 15.8ms (75%)  
  + stride : 5.2ms
  + MFCC 설명 : https://blog.naver.com/sooftware/221661644808  
## Reference
* Model  
  + based on IBM pytorch-seq2seq : https://github.com/IBM/pytorch-seq2seq
* Dataset  
  + Dataset : https://github.com/clovaai/speech_hackathon_2019
## License
```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
```
데이터등은 비상업적인(NON-COMMERCIAL)인 AI 연구, 학습에서만 사용할 수 있으며, 상업적인 이용은 엄격히 금지됨.
데이터등은 제공된 그대로만 사용하여야 하며, 이를 수정, 변경 등 재가공하는 것음 엄격히 금지됨.
데이터등은 자신만 이용하여야 하며, 이를 제3자에게 임의로 제공하는 것은 엄격히 금지됨.
데이터등을 연구 등에서 비상업적으로 이용하는 경우에도 데이터의 출처를 [NAVER Corp]로 표시하여야 함.
네이버는 제공되는 데이터등의 완전성이나 무결성, 정합성, 정확성, 적절성 등을 보증하지 않으며, 따라서 그에 대한 책임을 부담하지 않음.
※ 제공되는 데이터등은 음성제공자가 제공하면서 동의한 범위 내에서만 사용되어야 하므로, 상업적 이용 등 그를 넘어 이용할 경우 음성제공자 사이와의 관계에서도 법적인 책임을 부담할 수 있으므로, 반드시 위 제한사항을 준수하여 사용하시기 바랍니다.
```

