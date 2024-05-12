# SecureBERT and LLAMA 2 Empowered Control Area Network Intrusion Detection and Classification

**저자:** Xuemei Li, Student Member, IEEE, Huirong Fu, Member, IEEE  
**논문 링크:** [arXiv:2311.12074](https://arxiv.org/pdf/2311.12074)

--- 

이 연구는 Controller Area Network (CAN) 공격 탐지를 위해 사전 훈련된 Transformer 모델의 활용성을 탐구합니다. 연구팀은 **CAN-SecureBERT**와 **CAN-LLAMA2** 두 가지 모델을 개발했습니다. 특히, **CAN-LLAMA2** 모델은 균형 잡힌 정확도, 정밀 탐지율, F1 점수에서 0.999993이라는 뛰어난 성능을 달성했으며, 오경보율은 기존 최고 모델보다 52배 낮은 3.10e-6을 기록하였습니다.

이 연구는 LLM 기반으로 하여 사이버 보안 작업에 적응시키는 방안의 가능성을 보여줍니다. 이를 통해, 향후 사이버 보안 분야에서의 인공지능 활용이 더욱 확대될 것으로 기대됩니다.

## I. INTRODUCTION
CAN은 차량 내 다양한 Electronic Control Units (ECUs)와 제어 구성 요소들을 연결하는 주요 인터페이스로, ECUs 간의 메시지를 모니터링하여 비정상적인 통신을 식별하는 것이 중요합니다. 유럽 연합의 새로운 규정(WP.29 R155)에 따라 2024년 7월부터 모든 신차는 사이버 보안 공격을 감지하고 대응할 수 있는 능력을 갖추어야 하며, 이를 위한 로그 데이터의 체계적인 수집이 요구됩니다. 이에 따라, 다양한 CAN 침입 탐지 방법론이 소개되고 있으며, 이는 지문 기반, 매개변수 모니터링 기반, 데이터 흐름 수준 분석, 그리고 Machine Learning (ML) 기술을 포함합니다. 그러나 기존 방법론은 데이터 전처리, 특징 추출, 복잡한 규칙 및 작업 흐름 생성에 의존하며, 새로운 공격을 식별하는 데 한계가 있습니다. 이를 해결하기 위해, Viden과 같은 새로운 접근 방식과 빈도 기반 기술, 전통적 ML 모델이 개발되었습니다. 또한, 트랜스포머 모델의 뛰어난 능력이 인코더와 디코더 구조를 통해 Natural Language Processing (NLP) 작업에 적용되어 성능 개선을 이루었습니다.

![그림 1](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/4e810be5-1767-4600-aed8-fa1845040031)
 
<p align="center">Fig. 1. In-Vehicle Networks (IVNs) IDS Taxonomy Based on Technology Implementation [2]</p>

CAN-LLAMA2 모델이 가장 높은 성능을 보이고, CAN-SecureBERT가 그 다음으로 높은 성능을 보였습니다. 이 모델들은 전통적인 데이터 전처리 과정 없이도 raw CAN message logs 직접 분석할 수 있는 능력을 가지고 있어, 제한된 데이터로도 우수한 성능을 달성할 수 있으며, 더 큰 데이터셋으로 훈련 시 일반화 능력이 향상됩니다. 또한, cybersecurity domain knowledge 통합한 CAN-SecureBERT는 기대와 달리 CAN 공격 탐지 및 분류에 직접적인 기여를 하지 않았습니다. CAN-LLAMA2는 Low-Rank Adaptation (LoRA) 기술을 활용하여 훈련되었으며, 이는 모델 매개변수의 단 0.57%만 변경되어 다양성을 유지하는 것을 가능하게 합니다. 이러한 접근 방식은  Vehicle Security Operations Center (VSOC) 팀이 Fine-tuning과 adapter head integration을 통해 다양한 하위 작업에 모델을 적용할 수 있도록 합니다. 

## II. RELATED WORKS
이 Section에서는 CAN 공격 탐지에 Transformer 기반 모델을 활용하는 최신 연구 노력들을 식별합니다. 최신 발견들의 종합적인 개요와 관련 연구들의 내재된 한계점들을 평가하는 것을 목표로 합니다. Nwafor et al.은 BERT를 활용한 언어 기반 침입 탐지 모델을 제안했습니다. 그들은 먼저 BERT 모델을 CAN 메시지 내의 의미를 이해하도록 훈련시킨 후, CAN 메시지 분류를 위해 모델을 Fine-tuning했습니다. 훈련 절차는 데이터의 64%를 사용하고, 20%는 검증을 위해, 나머지 16%는 테스트를 위해 할당되었습니다. 그들의 모델은 거의 100%에 가까운 정확도, 정밀도, 재현율, F1 스코어를 달성했습니다. 그러나 그들의 성능에 대한 구체적인 세부 사항은 보고되지 않았습니다. 관련 연구에서, Natasha et al.은 "CAN-BERT" 모델을 소개했습니다. 그들의 주요 목표는 이상 탐지 모델을 설계하고 CAN 메시지 분류를 정상 및 비정상 메시지를 구분하는 이진 분류 문제로 설정하는 것이었습니다. 그들은 BERT 모델을 채택하고 이를 표준 CAN 메시지를 사용하여 훈련시키고, Masked Language Model (MLM) 기술을 통합했습니다. 이후, 모델은 무작위로 마스킹된 테스트 시퀀스의 각 CAN ID에 대한 확률 분포를 예측했습니다. 이 접근 방식은 정상 메시지와 관련된 CAN ID가 없는 메시지를 비정상으로 분류했습니다. 그들의 모델은 다양한 유형의 공격에 대해 0.81에서 0.99까지의 F1 스코어를 달성했습니다. 그러나, 그들의 접근 방식은 정상 메시지 CAN ID의 확률 분포에 크게 의존하여, 정상 메시지와 유사한 CAN ID를 사용하는 주입 메시지 공격을 탐지할 수 없게 만들었습니다. 또한, 그들의 모델은 fuzzy 및 기능 장애 공격에 대해 0.9 미만의 F1 스코어를 보였습니다. 이 접근 방식의 또 다른 한계는 이진 분류의 성격으로, 추가적인 이상 분류 모델의 개발이 필요하다는 것입니다. Ehsan et al.은 그들의 연구에서 SecureBERT를 소개했습니다. 이는 사이버 보안 관련 텍스트, 예를 들어 사이버 위협 인텔리전스(CTI) 내의 텍스트 함의를 포착하기 위해 설계된 사이버 보안 언어 모델입니다. SecureBERT는 사이버 보안 도메인 토큰에 맞춤화된 커스텀 토크나이저를 특징으로 하는 사전 훈련된 RoBERTa 모델을 기반으로 구축되었습니다. 이 모델은 방대한 사이버 보안 텍스트 코퍼스에 대해 훈련되었으며 표준 MLM 방법을 사용하여 평가되었습니다. SecureBERT는 사이버 보안 분야의 도메인별 지식을 내포한 귀중한 사전 훈련 모델로서, 우리는 이를 활용하여 CAN-SecureBERT를 개발할 것입니다. 별도의 노력에서, Meta GENAI의 Hugo와 Louis et al.은 LLAMA 2를 공개했습니다. 이는 700M에서 70B 파라미터에 이르는 다양한 규모의 사전 훈련 및 미(fine-tuning)세 조정 모델의 새로운 패밀리입니다. LLAMA 2 모델은 기존 Transformer 모델을 기반으로 하여, RMSNorm을 사용한 사전 정규화, SwiGLU 활성화 함수, 회전 위치 임베딩, 그리고 그룹화된 쿼리 주의를 특징으로 합니다. 이 모델들은 2000억 토큰의 전처리 데이터를 사용하여 훈련되었으며, 따라서 SecureBERT에 비해 상당한 지식을 담고 있습니다. 우리 연구에서는 CAN-LLAMA2를 개발하기 위해 LLAMA 2를 사용할 것입니다. 결과적으로, 위의 문헌 검토에서 나타난 바와 같이, CAN 침입 탐지 및 분류를 위해 SecureBERT와 LLAMA 2를 이전에 사용한 연구 노력이 없습니다. 따라서, 우리의 연구는 이 분야에서 중요한 공백을 채우고자 합니다. 

## III. MODEL ARCHITECTURE
이 Section에서는 Transformer 아키텍처와 제안된 CAN-C-BERT, CAN-SecureBERT, CAN-LLAMA2 모델의 아키텍처에 대한 종합적인 개요를 제시합니다. Transformer는 2017년 구글에 의해 처음 제안된 deep learning architecture로, BERT, GPT, LLAMA 2를 포함한 여러 NLP LLM의 기본 모델로 사용되었습니다. 이 architecture는 인코딩과 디코딩 단계로 구성되며, 각각 6개의 인코더와 디코더 스택으로 변환 과정을 거칩니다. 이 과정은 "Transformer"의 구성 요소를 분해하고 다시 조립하는 과정과 유사합니다. 각 스택 처리는 다른 초기 값에 기반한 여덟 개의 블록으로 세분화되며, 각 블록은 구성 요소의 무게와 상호 관계를 기록하는 "transformation manual"을 가지고 있습니다. 이는 self-attention mechanism입니다. 각 인코더는 self-attention 및 feed-forward networks라는 두 가지 기본 모듈을 통합합니다. 각 self-attention mechanism은 multi-head attention heads를 가지고 있습니다. 먼저, 입력 embeddings은 tokenizers를 사용하여 생성되며, 각 토큰에 대한 위치 정보는 위치 인코딩을 통해 포함됩니다. 결과물은 이후 8개의 곱셈을 특징으로 하는 multi-head attention heads층으로 전송되며, 여기서 세 개의 사전 훈련된 벡터(Key, Query, Value)를 사용하여 구성 요소의 무게와 상호 관계를 결정합니다. 결과적으로 가중치 합이 주의 출력 벡터를 생성합니다. 그 후, 출력은 fully connected feed-forward layer를 통해 처리됩니다. 또한, multi-head attention heads layer과 fully connected feed-forward layer 모두에 잔여 연결과 층 정규화가 적용됩니다. 디코더는 유사한 구성을 가지고 있지만 인코더와 몇 가지 차이점을 보입니다. 디코더 스택에서 파생된 출력에 대해 masked multi- head attention mechanism을 도입합니다. 또한, 디코더 내의 multi-head self-attention mech-anism층은 이전 또는 현재 위치 후에 위치 마스킹을 포함하도록 수정됩니다.

![그림 2](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/fae65ffa-befe-46ee-859b-68b3d27f98d5)
 
<p align="center">Fig. 2. Transformer Model Architecture [5]</p>

Masked adaptation이 시퀀스 내에서 subsequent predictions에 대한 주의를 방지하기 위해 사용됩니다. 이는 autoregressive tasks (한 번에 하나의 token을 예측)에서 필수적이며, 그렇지 않으면 미래 positions의 정보가 current prediction에 영향을 미치지 않도록 합니다, 이는 data leakage로 이어질 수 있습니다. Encoder 상태의 Ks와 Vs의 곱셈과 decoder 상태의 Qs를 이용한 self-attention mechanism은 softmax 함수를 통한 normalization 단계를 따릅니다. Softmax layer는 최종 출력으로 가장 높은 확률을 가진 단어를 식별합니다.
Transformer network의 훈련 방법론은 gradient descent algorithm을 따르며, backpropagation을 활용하여 모델 매개변수의 weights를 조정하고 predictions과 actual values 사이의 오차를 최소화함으로써 최적의 학습 결과를 달성합니다.

B. CAN-C-BERT <br>
BERT(Bidirectional Encoder Representations from Transformers)는 2018년 Google에 의해 소개된 "Encoder-only" Transformer로, 대규모 데이터셋에서의 강력한 사전 훈련(pre-training)을 통해 다양한 downstream tasks에 fine-tuning할 수 있으며, natural language understanding에서의 다양성과 효과성을 강조합니다. BERT 모델은 라벨이 없는 텍스트로부터 양방향의 심층적인 표현을 생성하며, 사전 훈련 후에는 sequence classification과 같은 다양한 downstream tasks에 fine-tuning될 수 있습니다. 이 연구에서는 BookCorpus 데이터셋과 English Wikipedia에 사전 훈련된 BERT base version 모델을 사용했습니다.

CAN-C-BERT 모델은 침입 분류(intrusion classification) 모델로, "C"는 "Classification"를 의미합니다. CAN-C-BERT 모델 구조는 사전 훈련된 BERT base version 모델과 classification head를 포함합니다. 사전 훈련 과정은 문법과 문맥에 대한 기초적인 이해를 확립하고, MLM(Masked Language Model) 및 NSP(Next Sentence Prediction)의 두 가지 주요 훈련 전략을 사용합니다. MLM에서는 문장의 일부 단어를 가리고, 양방향 맥락을 기반으로 가려진 단어를 예측합니다. NSP에서는 [CLS] 특수 토큰과 문장 분리를 나타내는 [SEP] 토큰을 포함합니다.

Fine-tuning 과정에서는 특정 작업에 맞춰 BERT 모델을 추가로 훈련시키는데, 이때 sequence classification 작업을 위해 [CLS] 토큰의 임베딩 벡터에 classification head를 추가합니다. 이 classification head는 hidden layer, output layer, 그리고 softmax activation layer로 구성된 fully connected neural network입니다. [CLS] 임베딩 벡터는 hidden layer를 통해 output layer로 전달되고, 이곳에서 나온 출력값은 softmax activation layer를 통해 확률 벡터로 변환됩니다. 최종적으로 가장 높은 확률을 가진 클래스가 예측 결과로 선택됩니다.

이 과정을 통해, BERT 모델의 유연성과 다양한 자연어 처리(NLP) 작업에의 적용 가능성이 입증됩니다. 사전 훈련된 BERT 모델은 sequence classification과 같은 특정 작업에 fine-tuning되어 복잡한 언어 이해와 문맥 분석을 통해 높은 성능을 달성할 수 있습니다. 특히, CAN-C-BERT 모델과 같이 보안 관련 작업에 특화된 모델은 사전 훈련과 fine-tuning 과정을 통해 침입 시도를 효과적으로 감지하고 분류할 수 있으며, BERT의 양방향 표현력을 활용하여 정확한 예측과 분류가 가능합니다.

![그림 3](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/db138336-e4f3-4a7e-acd8-8a497c87d7f6)
 
<p align="center">Fig. 3. BERT Pre-training Model [9]</p>

![그림 4](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/8c179f15-25bd-473c-8ddd-484ddcaa4f4a)
 
<p align="center">Fig. 4. CAN-C-BERT Fine-tuning Model</p>

![그림 5](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/b325e8cc-8d64-4375-bef3-0e14bdc89c6d)
 
<p align="center">Fig. 5. CAN-C-BERT Embedding Input Structure: The BERT model combines Tokenization Embedding (1, n, 768) and Position Embedding (1, n, 768) through summation to obtain an input Embedding (1, n, 768) for the model.</p>

CAN 공격 detection 및 classification 과정은 CAN 메시지의 tokenization으로 시작됩니다. 이 tokens은 사전 훈련된 BERT 모델에 입력되며, [CLS] token의 출력 embedding이 classification head의 입력이 됩니다. 훈련하는 동안, 실제 ground truth labels과 비교하여 예측 labels에 대한 cross-entropy loss가 계산됩니다.

![그림 6](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/4824cfc9-a33e-4aa2-b439-6b117c76aaf3)
 
<p align="center">Fig. 6. CAN-SecureBERT Fine-tuning Model</p>

C. CAN-SecureBERT <br>
CAN-SecureBERT를 사용하여 CAN 메시지를 분류하는 과정은 CAN-C-BERT에 의해 사용된 접근 방식을 반영합니다. 이는 사전 훈련된 SecureBERT 모델과 fully connected neural network로 구성된 classification head를 포함합니다. SecureBERT는 사전 훈련된 RoBERTa-base 모델의 아키텍처를 활용하며, 12개의 hidden transformer 및 attention layers와 하나의 input layer를 특징으로 합니다. 이러한 적용은 98,411개의 사이버보안 관련 textual elements(10억 tokens에 해당)의 대규모 데이터셋을 활용하여 RoBERTa-base 모델을 fine-tuning하는 것을 포함합니다. 모델은 원래 RoBERTa tokenizer를 기반으로 한 맞춤형 tokenizer를 통합하여 전체 vocabulary를 50,265로 확장합니다. 이 맞춤형 tokenizer는 모델이 textual corpora에서 사이버보안 관련 tokens를 추출하는 능력을 향상시킵니다. 이름에서 알 수 있듯이, SecureBERT는 Security와 BERT를 결합합니다. 모델의 효과는 훈련 단계에서 vocabulary의 token weights에 noise를 도입함으로써 더욱 증가됩니다.

CAN-SecureBERT의 모델 아키텍처는 Figure 6에서 CAN-C-BERT와 유사하게 표현됩니다. 이는 사전 훈련된 SecureBERT 모델과 classification head를 포함합니다. 사전 훈련된 SecureBERT 모델은 총 123 million model parameters를 가진 12개의 transformer blocks를 가집니다. Classification head는 fully connected neural network로 구현됩니다. 이는 SecureBERT 모델에서 분류 토큰 [CLS]의 embedding vector가 도출된 후에 통합됩니다. 이는 hidden layer, output layer, 그리고 softmax activation layer를 포함하는 fully connected neural networks로 구성됩니다. [CLS] embedding vector는 hidden layer를 통과한 후 output layer에 연결됩니다. Output layer에서 나온 출력은 softmax activation layer로 전달되어 확률 벡터를 얻습니다. 가장 높은 확률을 가진 클래스가 최종 예측 출력입니다. 

D. CAN-LLAMA2 <br>
LLAMA 2는 generative text 모델의 두 번째 버전으로, 사전 훈련 및 미세 조정(fine-tuned)이 완료된 모델 모음입니다. 2023년 7월 18일, Meta와 Microsoft 간의 협력 프로젝트로 공식적으로 소개되었습니다. Meta에 의해 개발되고 출시된 LLAMA 2 모델은 7 billion, 13 billion, 70 billion 파라미터를 가진 세 가지 다른 크기로 제공됩니다. 이 모델들은 CommonCrawl의 웹 페이지, GitHub의 오픈 소스 저장소 코드, 20개 언어로 된 Wikipedia 내용, 공공 도메인 책, ArXiv의 과학 논문에서 가져온 Latex 소스 코드, Stack Exchange의 질문과 답변 등 다양한 출처에서 2 trillion tokens으로 구성된 방대한 데이터셋에서 훈련되었습니다. 데이터셋 큐레이션 과정에서 개인 데이터를 포함한 웹사이트는 신중하게 제거되었으며, 신뢰할 수 있는 출처에서의 샘플은 업샘플링되었습니다.

CAN-LLAMA2 모델의 아키텍처는 '그림 7'에 나타나 있습니다. CAN-LLAMA2 모델은 사전 훈련된 LLAMA 2 모델과 classification head로 구성됩니다. 이 연구에서 사용된 사전 훈련된 LLAMA 2 모델은 half-precision 모델이며, 32개의 transformer decoder 블록을 포함해 총 7 billion 모델 파라미터를 가지고 있습니다. Classification head는 완전 연결 신경망(fully connected neural network)으로 구현됩니다.

CAN 공격 감지 및 분류 작업을 위해 사전 훈련된 LLAMA 2 모델을 훈련시킬 때, LLAMA 2 모델에서 파생된 마지막 토큰([EOS])(문장의 끝)의 embedding vector에 classification head가 통합됩니다. 이 classification head는 hidden layer, output layer, softmax activation layer를 포함하는 완전 연결 신경망입니다. [EOS] embedding vector는 hidden layer를 거쳐 output layer에 연결되며, output layer에서 나온 출력은 softmax 활성화를 거쳐 확률 벡터를 생성합니다. 최종 예측 출력은 가장 높은 확률을 가진 공격 유형에 해당합니다. 

![그림 7](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/b9aab1bb-7ce1-4da7-8cf5-76ffc386c2dc)
 
<p align="center">Fig. 7. CAN-LLAMA2 Fine-tuning Model</p>

CAN 공격 탐지 및 분류를 위한 첫 단계는 LLAMA 2 tokenizer를 사용하여 CAN 메시지를 토큰화(tokenize)하는 것입니다. 사전 학습된(pre-trained) LLAMA 2 모델에 의해 마지막 토큰의 임베딩(embedding)이 추출됩니다. 마지막 토큰의 결과 출력 임베딩은 분류 헤드(classification head)로 전달됩니다. 훈련 과정 동안, 예측된 라벨과 실제 ground truth 라벨을 비교하여 교차 엔트로피 손실(cross-entropy loss)이 계산됩니다.

## IV. FINE-TUNING PROCESS
이 Section에서는 수학적 표현을 사용하여 FINE-TUNING 과정을 설명합니다. 주요 단계는 사전 훈련된 모델에서 클래스 토큰을 획득하는 것, classification head를 적용하여 클래스 확률 벡터를 획득하는 것, loss function을 계산하는 것, 그리고 AdamW 알고리즘을 이용한 weight 최적화 절차를 포함합니다. 또한, LLAMA2 모델의 FINE-TUNING 용이하게 하는 parameter 기반 FINE-TUNING의 효율성을 향상시키는 방법론을 소개합니다.

A. CAN-C-BERT와 CAN-SecureBERT FINE-TUNING <br>
처음에 개별 원시 CAN 메시지는 토큰화됩니다. 이 과정은 입력 시퀀스를 transformer 모델과 연관된 tokenizer를 사용하여 subword 토큰으로 나누는 것을 포함합니다. X를 토큰화된 입력 시퀀스로 상징하겠습니다. 또한, 모델을 transformer 모델로 표시하며, 여기서 θ는 모델의 parameters에 해당합니다. 출력은 Z로 표현되며, 다음과 같이 방정식 (1)로 나타낼 수 있습니다.

<p align="center">Z = model(X, θ)</p>


transformer model 위에는 sequence 분류를 수행하는 분류 헤드(classification head)가 추가되어 있습니다. 이 분류 헤드는 숨겨진 계층(hidden layer), 출력 계층(output layer)으로 구성되어 있으며, 이어서 소프트맥스(softmax) 활성화 함수가 적용됩니다. 여기서 (W)는 선형 계층(linear layer)의 가중치 행렬(weight matrix)을, (b)는 바이어스 벡터(bias vector)를 의미하며, (C)는 총 클래스 수(total number of classes)를 나타냅니다. 각 클래스 (j)에 대해 계산된 예측 클래스 확률 (P_j)는 다음과 같이 표현됩니다.

$$
P_j = \frac{e^{W_j x + b_j}}{\sum_{k=1}^{C} e^{W_k x + b_k}}
$$

이 연구에서 선택된 손실 함수는 크로스 엔트로피 손실(cross-entropy loss)이며, 예측된 확률과 실제 ground truth 라벨 사이의 차이를 측정하는 데 사용됩니다. 변수 yij는 인디케이터 변수(indicator variable)를 나타내며, 만약 CAN 메시지 i의 실제 라벨이 클래스 j에 해당하면 1의 값을 가지고, 그렇지 않으면 0의 값을 가집니다. 개별 CAN 메시지에 대한 크로스 엔트로피 손실은 Li로 표시되며, Equation (3)에서 설명된 바와 같이 정의될 수 있습니다.

$$
L_i = - \sum_{j=1}^{C} y_{ij} \log(P_j)
$$

미니 배치의 크기가 N인 경우 집계 손실은 L로 표시되며, 개별 손실 Li의 평균으로 계산됩니다. 이는 방정식 (4)와 같이 표현될 수 있습니다.

$$
L = \frac{1}{N} \sum_{i=1}^{N} L_i
$$

과적합(overfitting)의 위험을 완화하기 위해, 이 연구에서 사용된 최적화 알고리즘은 [26]에서 제시된 AdamW입니다. AdamW는 가중치 감소(weight decay) 정규화를 포함한 Adam 최적화기(optimizer)의 변형입니다. 이를 통해 모델의 매개변수 $\theta$를 조정하여 손실 함수 $L$을 최소화하는 것을 돕습니다. 가중치 업데이트 과정은 방정식(5)를 통해 설명될 수 있으며, 여기서 $\theta_t$는 $t$번째 반복에서의 모델 매개변수를 나타내고, $\eta$는 학습률을 의미하며, $\hat{m}_t$과 $\hat{v}_t$는 각각 그래디언트와 그 제곱의 편향 보정된(bias-corrected) 이동 평균을 나타냅니다. $\epsilon$은 수치적 안정성을 위해 도입된 작은 상수를 나타내고, $\lambda$는 가중치 감소 계수를 의미합니다.

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \cdot \lambda \cdot \theta_t
$$

B. Fine-tuning CAN-LLAMA2
LLAMA 2는 70억 개의 파라미터와 반정밀도를 사용하여 운영할 때 14GB의 GPU RAM이 필요합니다. 제한된 계산 자원을 사용하여 훈련 및 추론을 용이하게 하기 위해, 이 연구는 파라미터 효율적인 파인튜닝 라이브러리 [27]에서 소개된 LoRA 기술을 사용하여 CAN-LLAMA 2를 파인튜닝합니다. LoRA는 모델의 가중치 행렬을 lower-rank 행렬로 근사화하는 방법을 포함합니다. 이 근사화는 사전 훈련된 모델에서 재훈련해야 할 파라미터의 수를 크게 줄입니다. 사전 훈련된 모델의 특정 층의 가중치 행렬을 W로 나타내며, W ∈ Rmn으로 표현됩니다. 여기서 m은 층의 출력 뉴런 수, n은 입력 뉴런 수를 의미합니다. 가중치 행렬 W는 두 개의 lower-rank 행렬 U와 V를 사용하여 근사화할 수 있으며, 이는 방정식 (6)에서 보여집니다. 이때, U ∈ Rmr 및 V ∈ Rr*n이며, r은 원하는 rank입니다. hyperparameter r의 선택은 정확도와 압축 수준 사이의 균형을 고려하여 결정됩니다.
<p align="center">W = UV</p>

주요 목표는 저차원(low-rank) 행렬 U와 V의 매개변수(parameters)를 수정하여, 방정식 3(Equation 3)에서 보여지는 바와 같이 교차 엔트로피 손실(cross-entropy loss)을 최소화하는 것입니다. 여기서 L은 특정 작업(task-specific)에 대한 손실 함수(loss function)를 나타냅니다. 이 컨텍스트에서, θU와 θV는 각각 U와 V에 해당하는 매개변수를 의미하며, f는 저차원 근사(low-rank approximation)를 포함하는 모델(model)의 전방 전달(forward pass)을 나타냅니다.

$$
L_{fine-tune} = L(\theta_U, \theta_V) = L(f(UV, X), Y)
$$

모델을 업데이트하는 전 과정에서, 매개변수 (\theta_U)와 (\theta_V)는 AdamW 최적화 알고리즘을 사용하여 조정됩니다. 이 업데이트는 미세 조정(fine-tuning) 손실 (L_{fine-tune})을 최소화하기 위해 수행됩니다.


V. PERFORMANCE METRICS <br>
이 section에서는 성능 벤치마크를 위해 선택된 지표(metrics)들을 소개합니다. 우리는 불균형 데이터셋을 다룰 수 있고, 최종 사용자에게 직접적인 영향을 줄 수 있는 지표들을 선택했습니다.

제안된 CAN-C-BERT, CAN-SecureBERT, 및 CAN-LLAMA2 모델의 성능 평가는 BA(Balanced Accuracy, 균형 정확도), PREC(Precision, 정밀도), DR(Detection Rate, 탐지율) 또는 Recall(재현율), FAR(False Alarm Rate, 오경보율), F1 점수, 그리고 모델 파라미터 크기와 같은 핵심 지표들에 의존합니다. 이 지표들의 수학적 표현은 [28]에서 유도될 수 있으며, 여기서 TP는 True Positive(진짜 양성), TN은 True Negative(진짜 음성), FP는 False Positive(가짜 양성), FN은 False Negative(가짜 음성)을 의미합니다.
해킹 데이터셋의 고유한 특성으로 인해, 공격 데이터의 인스턴스가 정상 데이터에 비해 현저히 적어 불균형을 나타냅니다. 결과적으로, 전통적인 정확도(accuracy) 지표는 이러한 시나리오에서 오해의 소지가 있을 수 있습니다. 이를 해결하기 위해, BA가 사용되며, 이는 개별 클래스 예측의 평균 정확도를 나타냅니다. BA는 불균형한 클래스 분포를 가진 데이터셋에 특히 적합한 모델 성능의 더 신뢰할 수 있는 측정치를 제공합니다.

$$
BA = \frac{1}{C} \sum_{i=1}^{C} \frac{TP_i}{TP_i + FN_i}
$$

PREC는 모델이 만들어낸 긍정적 예측의 정확성을 측정합니다.
$$
PREC = \frac{TP}{TP + FP}
$$

DR은 일반적으로 Recall로 알려져 있으며, 실제 긍정적인 사례 전체에 대비하여 TP(참 긍정) 예측의 비율을 평가합니다.
$$
DR = \frac{TP}{TP + FN}
$$

FAR는 실제 부정적인 사례들 중에서 FP(잘못된 긍정) 예측의 비율을 측정합니다. 이 지표는 특히 IDS(침입 탐지 시스템, Intrusion Detection System)의 맥락에서 중요합니다. 예를 들어, FAR가 1e-3인 모델은 대부분의 ML(머신러닝, Machine Learning) 작업에서 탁월한 성능으로 간주될 수 있지만, IDS의 경우에는 실질적인 함의가 명확해집니다. 매일 수백만 대의 차량에서 생성되는 수백만 개의 CAN(컨트롤러 영역 네트워크, Controller Area Network) 메시지를 고려해 보세요. 하루에 1000만 개의 메시지가 생성된다고 가정했을 때, FAR가 1e-3이라면, 거짓 경보 메시지의 수는 10,000개에 이릅니다. 이러한 규모의 거짓 경보를 매일 관리하고 분류하는 것은 VSOC(차량 보안 운영 센터, Vehicle Security Operations Center) 팀에게 실질적으로 비현실적입니다.
$$
FAR = \frac{FP}{FP + TN}
$$

F1 점수는 PREC(정밀도)와 DR(재현율) 또는 Recall의 조화 평균을 나타냅니다. 이 지표는 모델의 성능을 균형 있고 포괄적으로 측정하는 데 사용됩니다.
$$
F1 = \frac{2 \cdot PREC \cdot DR}{PREC + DR}
$$

VI. DATASETS <br>
이 연구에서는 현대의 YF 쏘나타에서 수집된 차량 해킹 데이터셋을 사용했습니다. 이 데이터셋은 DoS(DoS) 공격, 퍼지(Fuzzy) 공격, 그리고 스푸핑(Spoofing) 공격 등 세 가지 유형의 공격을 포함하고 있습니다. 데이터셋은 실제 차량의 OBD-II(On-Board Diagnostics II) 포트를 통해 CAN 트래픽을 캡처하면서 시뮬레이션된 메시지 공격이 실행되는 동안에 수집되었습니다. 데이터셋 내의 속성은 타임스탬프(Timestamp), CAN ID, 데이터 길이 코드(DLC, Data Length Code), DATA[0]에서 DATA[7]까지, 그리고 플래그(Flag)를 포함합니다. [13]에 자세히 설명된 각 공격 유형에 대한 추가적인 통찰은 다음과 같습니다:
1) DoS 공격: ‘0x000’과 같은 고우선순위의 CAN 메시지를 주입하는 것을 포함합니다.

<p align="center">TABLE I</p>
<p align="center">CAR HACKING DATASET</p>

| Attack Type          | Messages   | Normal    | Injected |
|----------------------|------------|-----------|----------|
| DoS Attack           | 3,665,771  | 3,078,250 | 587,521  |
| Fuzzy Attack         | 3,838,860  | 3,347,013 | 491,847  |
| Spoofing Drive Gear  | 4,443,142  | 3,845,890 | 597,252  |
| Spoofing RPM Gauge   | 4,621,702  | 3,966,805 | 654,897  |
| Attack-Free (Normal) | 988,987    | 988,872   | NA       |

특정 간격으로 '0x000'과 같은 CAN ID 메시지를 주입하는 것을 포함합니다. 구체적으로, 이러한 '0x000' CAN ID 메시지는 매 0.3밀리초(millisecond)마다 주입되었습니다.
2) Fuzzy(퍼지) 공격: 위조된 무작위 생성된 CAN ID 및 DATA 값이 포함된 메시지의 주입을 포함합니다. 이 주입은 무작위화된 CAN ID 및 데이터 값이 포함된 메시지와 함께 0.5밀리초 간격으로 수행되었습니다.

3) RPM/Gear(RPM/기어) 공격: RPM 및 기어 정보와 관련된 특정 CAN ID와 연관된 메시지의 주입을 중심으로 합니다. RPM 및 기어와 관련된 메시지는 1밀리초 간격으로 도입되었습니다. 이러한 공격은 현대 YF 쏘나타 내에서 실제 상황 및 행동을 재현하고자 CAN 네트워크에 메시지를 도입함으로써 실행되었습니다. 차량 해킹 데이터셋에 대한 개요는 표 I에서 제시됩니다.

VII. EXPERIMENTS AND RESULTS <br>
이 section에서는 제안된 모델을 학습(train)하고 평가(evaluate)하기 위해 데이터셋을 어떻게 처리하는지에 대해 설명합니다. 실험적 설정(experimental setup), 사용된 하이퍼파라미터(hyperparameters), 그리고 모델의 복잡성(complexity)이 소개되며, 실험 결과에 대한 광범위한 논의와 주목할 만한 관찰 결과가 강조됩니다.

A. 하이퍼파라미터 세밀 조정(Fine-tuning of Hyperparameters)
데이터셋 분할: 자동차 해킹 데이터셋은 70% 학습 데이터셋(training dataset)과 30% 테스트 데이터셋(test dataset)으로 나뉩니다. 특히, 학습 데이터의 양이 모델 성능에 미치는 영향을 탐구하기 위해, 전체 데이터셋에서 무작위로 선택된 1% 및 10%의 학습 데이터셋 하위 집합(subsets)이 사용됩니다. 이 과정은 데이터의 균형 있는 대표성을 유지합니다.

하드웨어 사양: AMD Ryzen 9 5900X 12코어 프로세서, 3.70 GHz 클럭 속도, 128GB RAM, Nvidia RTX 3090 GPU 2개가 Nvidia SLI 브리지로 연결된 구성입니다.

학습 하이퍼파라미터: CAN-C-BERT와 CAN-SecureBERT는 학습 배치 크기 4, 검증 배치 크기 32, 학습률(learning rate) 5e-5, 가중치 감소(weight decay) 0.01로 설정됩니다. CAN-LLAMA2는 학습 배치 크기 4, 검증 배치 크기 16, 그라데이션(gradient) 누적 단계 4, 학습률 3e-5, 가중치 감소 0.01로 설정됩니다. CAN-LLAMA2는 제한된 계산 자원을 고려하여 4비트 정밀도로 모델 파라미터를 로드합니다.

LoRA 설정: LoRA(Low-Rank Adaptation)는 학습 파이프라인에 통합되며, LoRA 주의 차원은 16, 알파 파라미터는 64로 설정되고, LoRA 레이어에는 0.1의 드롭아웃 확률과 0의 편향 값이 적용됩니다.
학습 기간: 모든 세 모델은 총 10 에폭 동안 학습됩니다.

B. Model Complexity <br>
Table II 미세 조정(fine-tuning)된 모델 크기와 파라미터에 대한 비교 분석을 제공합니다. CAN-C-BERT와 CAN-SecureBERT는 훈련 중 모든 파라미터를 미세 조정(fine-tuning)할 수 있는 기능을 제공합니다. 특히, 1% 훈련 데이터셋을 사용할 때, 이 모델들의 훈련 시간은 각각 약 4분과 5분입니다. 반면에, CAN-LLAMA2 모델을 훈련하는데는 118분이라는 훨씬 많은 시간이 필요합니다. 그러나 CAN-LLAMA2의 추론(inference) 속도는 다른 두 모델보다 약 8배 느립니다. 이 차이는 주로 계산 자원의 제한 때문입니다. CAN-LLAMA2는 70억(7 billion) 파라미터를 포함하고 있습니다. LoRA를 구현한 후, 모델의 선형 레이어(linear layers)에서 약 4천만(40 million) 파라미터를 미세 조정(fine-tuning)할 수 있습니다. 그러나 CAN-LLAMA2의 행렬 곱셈(matrix multiplication), 활성화(activation), 그리고 다른 수학적 연산에 대한 계산 요구사항은 다른 두 모델보다 훨씬 높습니다. 또한, GPU 메모리 크기의 제한으로 인해, CAN-LLAMA2는 검증(validation)을 위해 최대 16개의 CAN 메시지 배치 크기(batch size)만 수용할 수 있습니다.

미세 조정(fine-tuning) 중에 CAN-LLAMA2 모델의 파라미터 중 오직 0.57%만 변경된다는 것은 또 다른 주목할 만한 관찰입니다. 이는 LLAMA2 모델의 원래 파라미터 대부분이 변경되지 않는다는 것을 의미합니다. 따라서, CAN-LLAMA2는 다른 언어 관련 작업에 재사용될 수 있습니다. VSOC 팀은 사전 훈련된(pretrained) 모델을 미세 조정(fine-tuning)하고 어댑터 헤드(adapter heads)를 추가하여 다양한 다운스트림(downstream) 작업을 수행함으로써 동일한 모델을 활용할 수 있습니다. 

C. Results <br>

이 section에서는 우리의 연구 질문에 대응하고 제안된 모델(model)의 성능(performance)에 대한 통찰(insights)을 얻기 위해 결과에 대한 철저한 분석(analysis)을 수행합니다. 

<p align="center">TABLE II</p>
<p align="center">MODEL SIZE</p>

| Model Type       | Model Parameters | Trainable Parameters | Training Time | Inference Speed   |
|------------------|------------------|----------------------|---------------|-------------------|
| **CAN-C-BERT**     | 110 Million      | 110 Million          | 4 mins        | 894 messages/s    |
| **CAN-SecureBERT** | 123 Million    | 123 Million          | 5 mins        | 965 messages/s    |
| **CAN-LLAMA2**   | 7 Billion        | 40 Million            | 118 mins      | 14 messages/s     |

![그림 8](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/41c2ba3d-136f-4502-85d8-68a1f21dbcd4)

검증(validation) 결과 training loss(훈련 손실)과 validation loss(검증 손실) 결과는 그림 8과 9에 나타나 있습니다. 모든 모델은 10 epoch 내에 0에 가까운 손실로 수렴하는데, 이는 CAN-C-BERT, CAN-SecureBERT, CAN-LLAMA2 모두 10 epoch 내에 수렴한다는 것을 의미합니다. 그림 8은 10%의 데이터로 훈련된 모델이 1%의 데이터로만 훈련된 모델보다 더 빠르게 수렴함을 명확히 보여줍니다. 모든 훈련 손실이 0에 가까워, 모두가 수렴했음을 나타냅니다. 하지만 그림 9에서 볼 수 있는 검증 손실의 경우, 10% 데이터로 훈련된 모델들이 1% 데이터로 훈련된 모델들보다 훨씬 더 0에 가까운 손실을 가집니다. 이는 10% 데이터로 훈련된 모델들이 더 정확하게 표현됨을 의미합니다. 검증 손실이 0에 가깝다는 것은 이 모델들이 검증 데이터셋 내의 보이지 않는 데이터에 대해 효과적으로 수행할 수 있음을 의미합니다.

Balanced Accuracy(BA, 균형 정확도), Precision(PREC, 정밀도), Detection Rate(DR, 탐지율), F1 score(F1 점수)는 그림 10에서 13까지 나타나 있습니다. 모든 6개 모델은 이 메트릭들에 대해 1에 가까운 결과를 달성했는데, 이는 모든 모델이 매우 정확함을 나타냅니다. 1% 훈련 데이터를 사용한 모델의 성능을 보여주는 그림 10-13에서, CAN-LLAMA2가 초기 단계에서 우수한 성능을 보이며 다른 두 모델보다 더 빠르게 1.0의 값에 수렴함을 관찰할 수 있습니다. 10% 데이터로 훈련된 모델들이 모든 메트릭에 대해 1% 데이터로 훈련된 모델들보다 더 나은 성능을 보입니다. 모든 6개 모델 중, 10% 데이터로 훈련된 CAN-LLAMA2가 가장 좋은 성능을 보입니다. 10% 데이터로 훈련된 CAN-SecureBERT가 모든 메트릭에 대해 두 번째로 좋은 성능을 보입니다. 

![그림 9](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/26bd2ffe-95a8-489a-b6f8-d87a2eb6f563)
 
요약하자면, 다른 모델들과 비교했을 때, 10% 데이터로 훈련된 모델들이 1% 데이터로 훈련된 모델들보다 모든 지표에서 우수한 성능을 보입니다. 이 중에서, 10% 데이터로 훈련된 CAN-LLAMA2가 가장 뛰어난 성능을 보이며, 이어서 CAN-SecureBERT가 두 번째로 좋은 성능을 나타냅니다. CAN-LLAMA2는 BA(Balance Accuracy), PREC(Precision), DR(Detection Rate), F1 점수에서 0.999993을 달성했고, FAR(False Alarm Rate)은 3.1e-6으로, 이는 1000만 개의 CAN(컨트롤러 영역 네트워크) 메시지 데이터셋에서 단 31개의 메시지만이 잘못된 경보로 예상된다는 의미입니다. CAN-SecureBERT도 우수한 성능을 보이며, 1000만 개의 CAN 메시지에 대해 35개의 잘못된 경보가 예상됩니다. 이러한 결과는 MTH-IDS와 CAN-C-BERT를 크게 능가하며, VSOC 팀이 분류 작업 부담을 줄일 수 있음을 시사합니다. 이는 차량 네트워크 침입 탐지를 위한 사전 훈련된 지식을 가진 복잡한 모델의 효과성을 강조하며, 다양한 모델과 알고리즘의 조합으로 구성된 IDS(Intrusion Detection System)를 능가하는 결과를 보여줍니다.

![그림 10](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/6f37d598-6026-42e4-8955-1f22c8ca6cee)

![그림 11](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/12c8f1cc-a12a-494d-a284-78b739a6e410)

![그림 12](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/0c937422-fa4c-4273-af69-f37137ea6485)

![그림 13](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/f8c77ba3-af0c-4e2a-bd5d-ad775d4a4084)

<p align="center">TABLE III</p>
<p align="center">MODEL PERFORMANCE</p>

| Model Type                        | BA         | PREC       | DR         | FAR        | F1         |
| --------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| DCNN[13]                          | 0.999300   | -          | 0.998400   | 1.60e-3    | 0.999100   |
| E-MLP[14]                         | 0.990000   | 0.990000   | 0.990000   | 3.00e-3    | 0.990000   |
| GIDS[15]                          | 0.975250   | 0.976250   | 0.986500   | 1.00e-3    | 0.987900   |
| Transformer Sequential CAN ID[16] | -          | 0.998980   | 999350     | 5.80e-4    | 0.999170   |
| KNN[17]                           | 0.974000   | -          | 0.963000   | 5.30e-2    | 0.934000   |
| SVM[17]                           | 0.965000   | -          | 0.957000   | 4.80e-2    | 0.933000   |
| XYF-K[18]                         | 0.991000   | -          | 0.983900   | -          | 0.987900   |
| SAIDuCANT[19]                     | 0.872100   | -          | 0.866600   | 1.70e-2    | 0.920000   |
| SSAE[20]                          | -          | -          | 0.985000   | 1.76e-2    | 0.920000   |
| LSTM-Autoencoder[21]              | 0.990000   | -          | 0.990000   | -          | 0.990000   |
| MTH-IDS[12]                       | 0.999990   | -          | 0.999990   | 6.00e-4    | 0.999990   |
| CAN-C-BERT-1%-data                | -          | 0.992874   | 0.991377   | 9.90e-3    | 0.991773   |
| CAN-SecureBERT-1%-data            | 0.999718   | 0.999718   | 0.999718   | 1.30e-4    | 0.999718   |
| CAN-LLAMA2-1%-data                | 0.999587   | 0.999588   | 0.999587   | 3.10e-4    | 0.999587   |
| CAN-C-BERT-10%-data               | 0.999965   | 0.999965   | 0.999965   | 3.60e-5    | 0.999965   |
| CAN-SecureBERT-10%-data           | 0.999991   | 0.999991   | 0.999991   | 3.50e-6    | 0.999991   |
| CAN-LLAMA2-10%-data               | 0.999993   | 0.999993   | 0.999993   | 3.10e-6    | 0.999993   |


<p align="center">TABLE IV</p>
<p align="center">ATTACK CLASSIFICATION PERFORMANCE - CAN-C-BERT</p>

| Attack Type  | Validation Instances | PREC    | DR      | FAR     | F1      |
|--------------|----------------------|---------|---------|---------|---------|
| DoS          | 117674               | 0.999230| 1.0     | 3.1e-5  | 0.999614|
| Fuzzy        | 98867                | 0.999868| 0.999888| 4.5e-6  | 0.999878|
| Spoofing     | 119707               | 1.0     | 1.0     | 0       | 1.0     |
| RPM Spoofing | 131324               | 1.0     | 1.0     | 0       | 1.0     |


<p align="center">TABLE V</p>
<p align="center">ATTACK CLASSIFICATION PERFORMANCE - CAN-C-SECUREBERT</p>

| Attack Type  | Validation Instances | PREC     | DR       | FAR    | F1      |
|--------------|----------------------|----------|----------|--------|---------|
| DoS          | 117674               | 1.0      | 1.0      | 0      | 1.0     |
| Fuzzy        | 98867                | 0.999898 | 0.999828 | 3.5e-6 | 0.999878|
| Spoofing     | 119707               | 1.0      | 1.0      | 0      | 1.0     |
| RPM Spoofing | 131324               | 1.0      | 1.0      | 0      | 1.0     |

<p align="center">TABLE VI</p>
<p align="center">ATTACK CLASSIFICATION PERFORMANCE - CAN-LLAMA2</p>

| Attack Type  | Validation Instances | PREC     | DR        | FAR      | F1        |
|--------------|----------------------|----------|-----------|----------|-----------|
| DoS          | 117674               | 1.0      | 1.0       | 0        | 1.0       |
| Fuzzy        | 98867                | 0.999909 | 0.999858  | 3.1e-6   | 0.999883  |
| Spoofing     | 119707               | 1.0      | 1.0       | 0        | 1.0       |
| RPM Spoofing | 131324               | 1.0      | 1.0       | 0        | 1.0       |


D. Discussions <br>
위의 결과를 요약하면, 큰 데이터셋으로 훈련된 모델이 작은 데이터셋으로 훈련된 모델보다 일반화(generalization) 성능이 더 우수함을 보여줍니다. 이는 훈련 손실(training loss)과 검증 손실(validation loss)의 변화 추세에서 명확히 관찰됩니다. 특히, 10% 데이터로 훈련된 모델이 1% 데이터로 훈련된 모델보다 BA(Balance Accuracy), PREC(Precision), DR(Detection Rate), F1 점수에서 일관되게 더 나은 성능을 보입니다.
CAN 메시지 로그 분류에서 모든 제안된 모델은 뛰어난 성능을 달성하며, 이는 원시 텍스트(raw text) 기반의 CAN 메시지를 직접 처리함으로써 달성됩니다. 이러한 모델들의 BA, PREC, DR, F1 점수는 모두 0.99를 초과합니다. 이는 트랜스포머(transformer) 기반 모델이 특징 엔지니어링(feature engineering)과 데이터 전처리(data preprocessing) 없이도 CAN 메시지 로그를 효과적으로 분류할 수 있음을 의미합니다.
CAN-LLAMA2는 특히 1% 데이터만을 가진 상태에서도 다른 모델보다 빠르게 수렴(convergence)하며, 이는 CAN-LLAMA2가 복잡한 CAN 메시지 패턴을 더 잘 포착할 수 있는 능력을 가지고 있음을 나타냅니다. 더 많은 레이어와 매개변수를 가진 CAN-LLAMA2는 레이어 간 정보 공유를 통해 더 빠른 수렴에 기여합니다.
더 많은 사전 훈련된 지식을 가진 모델은 덜 훈련된 모델보다 성능이 좋으며, LLAMA2는 BERT와 SecureBERT에 비해 더 큰 데이터셋에서 사전 훈련됩니다. LLAMA2는 더 많은 매개변수를 가지고 있으며, 이는 더 많은 사전 훈련된 지식을 포착하는 데 기여합니다. 그러나 LoRa 적용 후에는 매개변수의 일부만 미세 조정(tuning)될 수 있음에도 불구하고, CAN-LLAMA2는 다른 모델을 능가합니다.
CAN-SecureBERT는 사이버 보안(cybersecurity)에 초점을 맞춘 데이터로 훈련된 SecureBERT를 사용하며, 이는 CAN-C-BERT보다 우수하지만 CAN-LLAMA2보다는 다소 낮은 성능을 보입니다. 매개변수 크기가 더 큰 SecureBERT의 성능 차이가 사전 훈련된 도메인 지식이나 매개변수 수 증가에 기인하는지는 명확하지 않습니다

VIII. CONCLUSION <br>
이 연구에서는 사전 훈련된 트랜스포머(transformer) 기반 모델을 미세 조정하여 CAN 침입 탐지 및 공격 분류를 위한 새로운 접근 방식을 제안합니다. 구체적으로, CAN-C-BERT, CAN-SecureBERT, CAN-LLAMA2와 같은 세 가지 독특한 모델이 개발되었습니다. 이 모델들은 CAN 메시지 로그를 직접 사용하며, 데이터 전처리가 필요 없습니다. 사전에 균형 잡힌 CAN 데이터셋으로 훈련된 후, 이들 모델의 성능은 최신 모델들과 비교되었습니다. CAN-LLAMA2는 모든 실증적인 최신 IDS(Intrusion Detection System) 시스템보다 더 높은 성능을 보여주며, CAN-SecureBERT는 두 번째로 좋은 모델로 평가되었습니다. 특히, CAN-LLAMA2는 BA, PREC, DR, F1 점수에서 0.999993에 이르고, 3.1e-6의 인상적으로 낮은 FAR(False Alarm Rate)을 달성하여, MTH-IDS의 FAR보다 약 52배 더 우수한 결과를 보였습니다. 전반적으로, 이 연구는 CAN IDS 분야의 발전을 도모하며, 모델 디자인, 성능 및 사이버 보안 응용에 대한 중요한 통찰력을 제공합니다.

IX. FUTURE WORKS
우리 연구의 주요 한계 중 하나는 컴퓨터 자원의 제약(constraint)입니다. 앞으로의 연구에서 우리는 제안된 CAN-LLAMA2 모델의 크기를 더욱 줄이고 추론(inference) 속도를 향상시키는 방법을 탐구할 것입니다.
