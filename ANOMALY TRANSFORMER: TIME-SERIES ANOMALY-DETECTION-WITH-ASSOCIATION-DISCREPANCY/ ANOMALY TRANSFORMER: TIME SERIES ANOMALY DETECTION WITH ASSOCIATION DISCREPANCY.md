# ANOMALY TRANSFORMER: TIME SERIES ANOMALY DETECTION WITH ASSOCIATION DISCREPANCY

**Authors:** Jiehui Xu\*, Haixu Wu\*, Jianmin Wang, Mingsheng Long (B)  
**Paper Link:** [Read the paper](https://arxiv.org/pdf/2110.02642)

<p align="center">ABSTRACT</p>
시계열에서 이상치(Anomaly)를 비지도 학습(Unsupervised Learning)으로 감지하는 것은 매우 어렵습니다. 모델은 구별 가능한 기준(Criterion)을 도출해내야 합니다. 이전 방법들은 주로 점 단위 표현(Pointwise Representation)이나 쌍 단위 연관성(Pairwise Association) 학습을 통해 문제를 해결하려 했으나, 복잡한 동적(Dynamic) 특성을 이해하기에는 부족했습니다. 최근 트랜스포머(Transformers)가 점 단위 표현과 쌍 단위 연관성의 통합 모델링에서 큰 성과를 보였고, 각 시간 지점의 자기 주의(Self-Attention) 가중치 분포가 전체 시리즈와의 풍부한 연관성을 내포할 수 있다는 것이 밝혀졌습니다.

우리가 주목한 것은, 이상치(Anomaly)의 드물기 때문에 비정상 점에서 전체 시리즈로의 비일상적인 연관성을 만들어내기가 매우 어렵다는 점입니다. 따라서, 이상치(Anomaly)의 연관성은 그들의 인접한 시간 지점(Adjacent Time Points)에 주로 집중되어야 합니다. 이 인접 집중 편향(Adjacent-Concentration Bias)은 정상 점과 비정상 점 사이에서 본질적으로 구별 가능한 연관성 기반의 기준을 암시합니다. 이는 우리가 연관성 차이(Association Discrepancy)를 통해 강조합니다.

기술적으로, 새로운 이상-주의 메커니즘(Anomaly-Attention Mechanism)을 사용하여 연관성 차이를 계산하는 이상치 트랜스포머(Anomaly Transformer)를 제안합니다. 최소-최대(Minimax) 전략은 연관성 차이의 정상-비정상 구별 가능성을 확대하기 위해 고안되었습니다. 이상치 트랜스포머는 서비스 모니터링(Service Monitoring), 우주 및 지구 탐사(Space & Earth Exploration), 그리고 물 처리(Water Treatment)와 같은 세 가지 응용 분야에서 여섯 가지 비지도 학습 시계열 이상치 감지 벤치마크에서 최고의 결과를 달성했습니다.

## 1 INTRODUCTION
실제 시스템(Real world)은 항상 연속적인 방식으로 작동하며, 이는 산업 장비, 우주 탐사 등과 같은 다중 센서에 의해 모니터링되는 여러 연속적인 측정값을 생성할 수 있습니다. 대규모 시스템 모니터링 데이터에서 장애를 발견하는 것은 시계열에서 비정상적(abnormal)인 시점을 감지하는 것으로 간소화될 수 있으며, 이는 보안을 확보하고 재정적 손실을 피하는 데 매우 의미가 있습니다. 그러나 비정상(abnormal)은 보통 드물고 방대한 정상 점들에 의해 숨겨져 있어 데이터 라벨링을 어렵고 비싸게 만듭니다. 따라서, 우리는 비지도(Unsupervised) 설정하에 시계열 이상 감지에 초점을 맞춥니다.

비지도 시계열 이상 감지(Unsupervised Time-Series Anomaly Detection)는 실제로 매우 도전적인 과제입니다. 모델은 비지도(Unsupervised) 작업을 통해 복잡한 시간적 역학에서 유익한 표현을 학습해야 하지만, 또한 많은 정상 시점(Normal Points)들로부터 드문 이상(Anomalies)을 감지할 수 있는 구별 가능한 기준을 도출해야 합니다. 다양한 클래식 이상 감지 방법들이 많은 비지도(Unsupervised) 패러다임을 제공했습니다. 예를 들어, 지역 이상 인자(Local Outlier Factor, LOF), 클러스터링(Clustering) 기반 방법들, 제시된 한 클래스 SVM(One-Class SVM, OC-SVM) 및 SVDD 등이 있습니다. 이 클래식 방법들은 시간적 정보를 고려하지 않으며 실제로 보지 못한 시나리오에 일반화하기 어렵습니다. 신경망(Neural Networks)의 표현 학습 능력으로 인해, 최근의 심층 모델들(Deep Models)은 뛰어난 성능을 달성했습니다. 주요 방법 범주 중 하나는 잘 설계된 순환 네트워크(Recurrent Networks)를 통해 점별 표현(Point-wise Representations)을 학습하고, 재구성(Reconstruction) 또는 자기회귀(Autoregressive) 작업에 의해 자기감독(Self-Supervised)됩니다. 여기에서, 자연스럽고 실용적인 이상 기준은 점별 재구성 또는 예측 오류(Prediction Error)입니다. 그러나 이상의 드물기 때문에, 점별 표현은 복잡한 시간 패턴에 대해 덜 유익할 수 있으며 정상 시점들에 의해 지배되어 이상을 덜 구별 가능하게 만들 수 있습니다. 또한, 재구성 또는 예측 오류는 점별로 계산되므로 시간적 맥락의 포괄적인 설명을 제공할 수 없습니다.

이 연구는 명시적인 연관 모델링을 기반으로 한 이상 감지 방법론을 다룹니다. 벡터 자기회귀(Vector Autoregression)와 상태 공간 모델(State Space Models) 같은 전통적 방법들은 시간 시리즈의 미세한 연관성을 모델링하는 데 한계가 있습니다. 최근에는 그래프 신경망(Graph Neural Network, GNN)이 동적 그래프를 학습하는 데 적용되었지만, 이 방법도 단일 시간 점에 제한되어 복잡한 시간 패턴을 충분히 다루지 못합니다. 부분수열 기반 방법은 더 넓은 시간적 맥락을 탐색하지만, 시간적 연관성을 포착하는 데 실패합니다.

이 논문에서는 비지도 체제에서 시계열 이상 감지에 트랜스포머(Transformers)를 적용합니다. 트랜스포머는 자연어 처리(Natural Language Processing, NLP), 기계 시각(Machine Vision) 및 시계열 분석 등 다양한 분야에서 성공적으로 사용되었습니다. 트랜스포머를 사용함으로써, 각 시간 점의 시간적 연관성을 자기-주의 맵(Self-Attention Map)에서 얻을 수 있으며, 이는 시간적 맥락에 대한 유익한 설명과 동적 패턴을 제공합니다. 이 연구는 이상이 전체 시리즈와 연관성을 구축하기 어렵다는 점을 관찰하고, 이상과 정상 패턴 간의 연관성 차이(Association Discrepancy)를 새로운 이상 기준으로 제안합니다. 이상은 정상 시간 점보다 더 작은 연관성 차이를 나타낼 것으로 예상됩니다.

이 연구는 비지도 시계열 이상 감지를 위해 트랜스포머(Transformer)를 사용하고, '이상 트랜스포머(Anomaly Transformer)'라는 새로운 접근 방식을 제안합니다. 이 방법은 자기 주의(Self-Attention) 메커니즘을 '이상 주의(Anomaly-Attention)'로 변형하여, 시간 시리즈 데이터의 연관성 차이(Association Discrepancy)를 계산합니다. 이상 트랜스포머는 사전-연관성(Prior-Association)과 시리즈-연관성(Series-Association)을 모델링하는 두 가지 분기 구조를 포함합니다. 사전-연관성은 인접한 시간 점들 사이의 관계를 가우시안 커널을 사용하여 나타내고, 시리즈-연관성은 원시 시리즈에서 학습된 자기 주의 가중치를 사용합니다. 또한, 두 분기 간의 미니맥스(Minimax) 전략을 적용하여 정상과 이상 사이의 구별 가능성을 증폭시키고, 새로운 연관성 기반의 검출 기준을 도출합니다. 이 방법은 여섯 개의 벤치마크에서 세 가지 실제 응용 분야에 대해 최신의 이상 감지 성능을 달성합니다.

핵심 성과는 다음과 같습니다 
1. 사전-연관성(Pre-Association)과 시리즈-연관성(Series-Association)을 동시에 다룰 수 있는 새로운 이상 감지 방식인 이상 트랜스포머(Anomaly Transformer)를 소개합니다. 이 방식은 이상 주의(Anomaly-Attention) 메커니즘을 활용합니다. <br>
2. 정상 상태와 이상 상태를 더 명확하게 구분하고, 연관성 기반의 새로운 탐지 기준(Detection Criterion)을 마련하기 위해, 미니맥스(Minimax) 전략을 도입합니다. <br>
3. 이상 트랜스포머는 다양한 벤치마크(Benchmark)에서 최신의 이상 감지 성능을 보여주며, 이 과정에서 심도 있는 실험(Experiments)과 사례 연구(Case Studies)를 제공합니다. <br>

## 2 RELATED WORK
2-1 UNSUPERVISED TIME SERIES ANOMALY DETECTION
이 논문(Paper) 비지도 시계열 이상 감지(unsupervised time-series anomaly detection)라는 중요한 실제 문제를 다룹니다. 이상 결정 기준에 따라, 패러다임은 대체로 밀도 추정(density estimation), 클러스터링 기반(clustering-based), 재구성 기반(reconstruction-based) 및 자기 회귀 기반(autoregression-based) 방법을 포함합니다. 밀도 추정 방법에서는, 클래식 방법인 로컬 아웃라이어 팩터(LOF, Local Outlier Factor), 연결성 아웃라이어 팩터(COF, Connectivity-based Outlier Factor)가 로컬 밀도와 로컬 연결성을 계산하여 이상을 결정합니다. DAGMM과 MPPCACD는 가우시안 혼합 모델(Gaussian Mixture Model)을 통합하여 표현의 밀도를 추정합니다.

클러스터링 기반 방법에서는 이상 점수(Anomaly Score)가 클러스터 중심까지의 거리로 정의됩니다. SVDD(서포트 벡터 데이터 디스크립션, Support Vector Data Description)와 Deep SVDD는 정상 데이터를 하나의 밀집된 클러스터로 모읍니다. THOC(시간적 계층적 원-클래스, Temporal Hierarchical One-Class)는 계층적 클러스터링을 통해 시간적 특징을 융합하고 이상을 감지합니다. ITAD(통합 텐서 이상 감지, Integrated Tensor Anomaly Detection)는 분해된 텐서에서 클러스터링을 수행합니다.

재구성 기반(Reconstruction-based) 모델은 재구성 오류를 통해 이상을 감지합니다. LSTM-VAE(장단기 기억 - 변이형 오토인코더, Long Short-Term Memory - Variational AutoEncoder) 모델과 OmniAnomaly는 정규화 흐름을 통해 확장된 LSTM-VAE 모델을 사용하며, InterFusion은 계층적 VAE(변이형 오토인코더, Variational AutoEncoder)를 사용하여 여러 시리즈 간의 의존성을 모델링합니다. GAN(생성적 적대 신경망, Generative Adversarial Network)도 재구성 기반(Reconstruction-based) 이상 감지에 사용됩니다.

자기 회귀 기반 모델은 예측 오류를 통해 이상을 감지합니다. VAR(Vector Autoregression)는 ARIMA(AutoRegressive Integrated Moving Average)를 확장한 것이며, LSTM(Long Short-Term Memory)로 대체될 수 있습니다.

이 논문(Paper)은 새로운 연관성 기반 기준을 특징으로 합니다. 기존의 랜덤 워크(random walk) 및 부분 시퀀스 기반 방법과 달리, 이 기준은 시간적 모델의 공동 설계를 통해 시간 지점의 연관성을 학습하기 위한 것입니다.

2.2 TRANSFORMERS FOR TIME SERIES ANALYSIS
최근 변환기(Transformer)는 자연어 처리(Natural Language Processing), 음성 처리(Audio Processing), 컴퓨터 비전(Computer Vision) 등 순차적 데이터 처리(Sequential Data Processing)에서 중요한 역할을 해왔습니다. 시계열 분석(Time Series Analysis)에서도 자기 주의(self-attention) 메커니즘을 통해 장기 시간적 의존성(Long-term Temporal Dependencies)을 탐색하는 데 사용됩니다. 특히, GTA는 IoT 센서 간 관계를 이해하고 이상 감지(Anomaly Detection)를 위해 변환기(Transformer)와 재구성 기준(Reconstruction Criterion)을 사용합니다. 이상 변환기(Anomaly Transformer)는 기존 사용법과 달리 연관성 불일치(Correlation Mismatch)를 기반으로 자기 주의 메커니즘을 이상 주의(Anomaly-Attention)로 혁신함으로써 새로운 접근 방식을 제시합니다.

## 3 METHOD
연속적인 d개의 측정값을 모니터링하고 시간에 따라 일정한 간격으로 관측된 데이터를 기록하는 상황에서, 비지도 시계열 이상 감지(Unsupervised Time Series Anomaly Detection)는 레이블 없이 관측값 xt가 이상한지 아닌지를 결정하는 문제입니다. 이 문제의 핵심은 정보가 풍부한 표현(Informative Representations)을 학습하고 구별 가능한 기준(Distinguishable Criterion)을 찾는 것입니다. Anomaly Transformer는 이를 위해 더 정보가 풍부한 연관성을 발견하고, 연관성 불일치(Association Discrepancy)를 학습함으로써 정상과 비정상을 구별할 수 있는 새로운 접근 방식을 제시합니다. Anomaly-Attention은 사전 연관성(Prior-Association)과 시계열 연관성(Series-Associations)을 구현하고, 미니맥스 최적화(Minimax Optimization) 전략을 통해 구별 가능한 연관성 불일치를 얻습니다. 이 아키텍처와 함께 설계된 연관성 기반 기준(Association-Based Criterion)은 학습된 연관성 불일치를 기반으로 합니다.

3.1 ANOMALY TRANSFORMER <br>
이상 감지(Anomaly Detection)를 위한 트랜스포머(Transformers)의 한계를 극복하기 위해, 'Anomaly Transformer'를 도입하였습니다. 이 모델은 Anomaly-Attention(이상-주의) 메커니즘과 피드-포워드(Feed-Forward) 레이어를 번갈아 가며 쌓아 올린 구조를 특징으로 합니다. 이 적층(Stacking) 구조는 깊은 다중 레벨(Multi-Level) 특징에서의 연관성(Associations)을 학습하는 데 유리합니다. 모델은 총 L개의 레이어(Layers)를 포함하며, 입력 시계열 데이터(Input Time Series Data)가 N의 길이를 가진다고 가정할 때, l번째 레이어의 방정식은 다음과 같습니다.


$$
Z_l = \text{Layer-Norm} \left( \text{Anomaly-Attention} \left( X_{l-1} \right) + X_{l-1} \right)
$$

$$
X = \text{Layer-Norm} \left( \text{Feed-Forward}(Z) + Z \right)
$$

![Figure 1](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/94f4af43-7541-48cf-9125-87b53ec7e603)
Figure 1은 이상치 주의 변환기(Anomaly-Attention Transformer)의 구조를 설명합니다. 이 모델은 사전 연관성(prior association)과 시리즈 연관성(series association)을 동시에 고려합니다. 주요 특징으로는 재구성 손실(reconstruction loss)과 함께, 특별히 설계된 정지-그라디언트(stop-gradient) 메커니즘을 사용하는 최소-최대(minimax) 전략을 통해 최적화가 이루어진다는 점입니다. 이러한 최적화 과정은 사전 및 시리즈 연관성을 제한하여 연관성 차이(association discrepancy)를 더욱 명확하게 구별할 수 있게 합니다.

**이상 감지(anomaly detection)** 위해 기존의 단일-가지(self-branch) 자기주의 메커니즘(self-attention mechanism)은 이전 연관성(prior-association)과 시리즈 연관성(series-association)을 동시에 모델링할 수 없다는 한계가 있습니다. 우리는 이를 극복하기 위해 두 가지 가지 구조(two-branch structure)를 가진 이상 주의(Anomaly-Attention)를 제안합니다. 이전 연관성을 위해, 우리는 배울 수 있는 가우시안 커널(Gaussian kernel)을 채택하여 상대적 시간 거리(relative temporal distance)에 대한 사전을 계산합니다. 가우시안 커널의 단일 모드(unimodal) 특성을 활용함으로써, 이 설계는 인접한 범위(adjacent horizon)에 더 많은 주의를 기울일 수 있습니다. 또한, 다양한 시계열 패턴(time series patterns)에 적응할 수 있도록 가우시안 커널에 대한 배울 수 있는 스케일 매개변수(scale parameter, σ)를 사용합니다. 이 두 형태는 각 시간 지점의 시간 의존성(temporal dependencies)을 유지하며, 점 단위 표현(point-wise representation)보다 더 많은 정보를 제공합니다. 따라서 정상(normal)과 비정상(abnormal) 사이를 구별할 수 있습니다. l번째 레이어(layer)에서의 이상 주의(Anomaly-Attention)은 원시 시리즈(raw series)로부터 연관성을 학습하는 시리즈 연관성 가지(series-association branch)를 포함하며, 인접 집중 사전(adjacent-concentration prior)과 학습된 실제 연관성(learned real associations)을 각각 반영할 수 있습니다.

![수식 2](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/b7b7c368-002d-4350-99c5-a4add16982f9)

![수식 + 내용 1](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/a871015e-80e2-4eea-b5d1-f1767696b4ee)

Figure 2 : Minimax association learning은 두 단계로 구성됩니다. minimize 단계에서는 가우스 커널(Gaussian kernel)에 의해 유도된 분포 가족 내에서 연관성 차이(Association Discrepancy)를 최소화합니다. 이 단계에서는 사전 연관성(prior-association)이 사용됩니다. 다음으로, maximize 단계에서는 복원 손실(reconstruction loss) 하에 연관성 차이를 최대화하는 작업이 수행됩니다. 이때 시리즈 연관성(series-association)이 사용됩니다.

![수식 + 내용 2](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/ab078abc-adbb-4c63-b11a-683917ff63d5)

![수식 + 내용 3](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/adf1a919-8a9e-45f8-a659-efdb00e56e08)
**Association Discrepancy** 
![수식 + 내용 4](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/0f5d7cb8-27d2-401f-a563-25977f71adfe)

**Minimax Strategy**
![수식 + 내용 5](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/032b19e8-035c-4bbd-b7f4-e03aa85a9725)


## 4 EXPERIMENTS
Anomaly Transformer는 세 가지 실용적 응용 분야에서 여섯 가지 벤치마크를 통해 광범위하게 평가되었습니다.

**Datasets** <br>
(1) SMD(Server Machine Dataset, Su et al., 2019)는 대규모 인터넷 회사에서 수집된 5주간의 데이터셋으로, 38개 차원(dimensions)을 가집니다.

(2) PSM(Pooled Server Metrics, Abdulaal et al., 2021)은 eBay의 여러 애플리케이션 서버 노드에서 내부적으로 수집된 26개 차원의 데이터입니다. 

(3) MSL(Mars Science Laboratory 로버)과 SMAP(Soil Moisture Active Passive 위성)은 NASA(Hundman et al., 2018)에서 공개한 데이터셋으로, 각각 55개와 25개 차원을 갖으며, 우주선 모니터링 시스템의 사건 놀라움 이상(ISA) 보고서에서 유래한 원격 측정 이상 데이터를 포함합니다. 

(4) SWaT(Secure Water Treatment, Mathur & Tippenhauer, 2016)는 연속 운영되는 중요 인프라 시스템의 51개 센서에서 얻은 데이터입니다. 

(5) NeurIPS-TS(NeurIPS 2021 Time Series Benchmark)는 Lai et al., 2021에 의해 제안된 데이터셋으로, 점-전역(point-global), 패턴-문맥(pattern-contextual), 패턴-셰이플릿
(pattern-shapelet), 패턴-계절(pattern-seasonal), 패턴-추세(pattern-trend)의 다섯 가지 시계열 이상 시나리오를 포함합니다. 통계적 세부사항은 부록의 표 13에 요약되어 있습니다.

**Implementation details** <br>
Shen et al. (2020)의 프로토콜을 따라, 중복되지 않는 슬라이딩 윈도우(sliding window)를 통해 부분 시리즈(sub-series)를 얻습니다. 모든 데이터셋에 대해 윈도우 크기는 100으로 고정됩니다. 이상 점수(anomaly scores, Equation 6)가 임계값 δ보다 큰 시간 점을 이상(anomalies)으로 라벨링합니다. 검증 데이터셋에서 r 비율이 이상으로 라벨링되도록 δ를 설정, SWaT은 0.1%, SMD는 0.5%, 그 외 데이터셋은 1%로 설정합니다. 연속적인 이상 구간에서 하나의 이상이 감지되면, 그 구간의 모든 이상이 정확히 감지된 것으로 간주하는 조정 전략(adjustment strategy)을 사용합니다. Anomaly Transformer는 3개 레이어(layer)를 가지며, 숨겨진 상태의 채널 번호(dmodel)는 512, 헤드 수(heads, h)는 8입니다. 손실 함수의 두 부분 사이의 균형을 위해 하이퍼파라미터 λ(Equation 4)는 3으로 설정됩니다. 초기 학습률 10^-4의 ADAM 옵티마이저를 사용하며, 훈련은 배치 크기 32로 10 에폭 내에 조기 종료됩니다. 실험은 Pytorch (Paszke et al., 2019)와 NVIDIA TITAN RTX 24GB GPU에서 수행됩니다.

![Figure 3](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/0f88edda-1e62-4db1-9c1b-cca694d1ae35)
Figure 3 : ROC 곡선(수평축: 거짓 양성 비율(False Positive Rate); 수직축: 진짜 양성 비율(True Positive Rate))은 5개 데이터셋에 대해 제시됩니다. ROC 곡선 아래 영역(AUC 값, Area Under the Curve)이 클수록 성능이 더 좋습니다. 사전에 정의된 임계값 비율(Threshold Ratio) r은 {0.5%, 1.0%, 1.5%, 2.0%, 10%, 20%, 30%} 중에서 선택됩니다.

Table 1 : Anomaly Transformer(우리 모델)의 5개 실제 데이터셋에서의 정량적 결과입니다. P, R, 그리고 F1은 각각 정밀도(Precision), 재현율(Recall), 그리고 F1-점수(F1-score)(%)를 나타냅니다. F1-점수는 정밀도(Precision)와 재현율(Recall)의 조화 평균(Harmonic Mean)입니다. 이 세 가지 지표(Metrics)에 대해, 높은 값(Higher Value)은 더 나은 성능(Better Performance)을 의미합니다.
![Table 1](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/a0f947bd-978c-452f-a1ab-f6345b9d03ec)

4.1 MAIN RESULTS 
**Real-world datasets** <br>
우리는 10개의 경쟁 베이스라인(Competitive Baseline)과 함께 5개의 실제 데이터셋(Real-World Dataset)에서 우리 모델(Anomaly Transformer)을 광범위하게 평가했습니다. 표 1(Table 1)에 나타난 바와 같이, Anomaly Transformer는 모든 벤치마크(Benchmark)에서 일관된 최신 성능(State-of-the-Art)을 달성합니다. 시간 정보(Temporal Information)를 고려하는 심층 모델(Deep Model)이 Deep-SVDD(Ruff et al., 2018), DAGMM(Zong et al., 2018)과 같은 일반적인 이상 탐지 모델(General Anomaly Detection Model)보다 우수한 성능을 보임을 관찰했습니다, 이는 시간 모델링(Temporal Modeling)의 효과를 검증합니다. 우리가 제안한 Anomaly Transformer는 RNN이 학습한 점별(Point-Wise) 표현을 넘어서 더 많은 정보를 담은 연관성(Associations)을 모델링합니다. 표 1의 결과는 시계열 이상 탐지(Time Series Anomaly Detection)에서 연관성 학습(Association Learning)의 장점을 설득력 있게 보여줍니다. 또한, 완전한 비교를 위해 그림 3(Figure 3)에서 ROC 곡선(ROC Curve)을 그렸습니다. Anomaly Transformer는 모든 5개 데이터셋에서 가장 높은 AUC 값(Highest AUC Values)을 가집니다. 이는 다양한 사전 선택된 임계값(Pre-Selected Thresholds) 하에서 거짓 양성 비율(False Positive Rate)과 진짜 양성 비율(True Positive Rate)에서 우리 모델이 잘 수행한다는 것을 의미하며, 이는 실제 세계 응용 프로그램(Real-World Applications)에 중요합니다.

**NeurIPS-TS benchmark** <br>
이 벤치마크는 Lai et al. (2021)에 의해 제안된 잘 설계된 규칙을 바탕으로 생성되었으며, 점별(point-wise) 이상과 패턴별(pattern-wise) 이상을 포함해 모든 유형의 이상을 완벽하게 포함합니다. 그림 4에서 보이는 바와 같이, Anomaly Transformer는 여전히 최신 성능(state-of-the-art)을 달성합니다. 이는 다양한 이상에 대한 우리 모델의 효과를 검증합니다.

![Figure 4](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/9f76c701-a26b-4362-b568-72e4ed3c728a)

**Ablation study** <br>
Table 2에서 우리 모델의 각 부분이 어떤 영향을 주는지 추가 조사했습니다. 연관성 기반 기준(association-based criterion)은 일반적으로 사용되는 재구성 기준(reconstruction criterion)을 지속적으로 능가합니다. 이 기준은 평균 절대 F1-점수(average absolute F1-score)를 18.76% (76.20→94.96) 상승시킵니다. 연관성 차이(association discrepancy)를 기준으로 사용해도 F1-점수(F1-score) 91.55%의 좋은 성능을 보이며, 이전 최신 모델 THOC(F1-score: 88.01%)를 넘어섭니다. 학습 가능한 사전 연관성(learnable prior-association, σ)과 미니맥스 전략(minimax strategy)은 각각 8.43% (79.05→87.48)와 7.48% (87.48→94.96)의 평균 절대 상승을 가져옵니다. 우리가 제안한 이상(Anomaly) Transformer는 순수 Transformer보다 18.34% (76.62→94.96) 높은 성능을 보여줍니다. 이 결과들은 우리 설계의 각 모듈이 효과적이고 필요함을 확인시켜 줍니다. 연관성 차이에 대한 더 많은 분석은 부록 D(Appendix D)에서 확인할 수 있습니다.

Table 2 : Ablation 연구 결과는 이상 감지 기준(anomaly criterion), 사전 연관성(prior-association), 그리고 최적화 전략(optimization strategy)에 초점을 맞추고 있습니다. Recon(재구성), AssDis(연관성 차이), Assoc(연관성 기반 기준)은 각각 순수 재구성 성능, 순수 연관성 차이, 그리고 제안된 연관성 기반 기준을 나타냅니다. Fix는 사전 연관성의 학습 가능한 스케일 매개변수 σ를 1.0로 고정하는 것을 의미합니다. Max(최대화)와 Minimax(미니맥스)는 연관성 차이를 다루는 두 가지 전략인 최대화 방식과 미니맥스 방식을 각각 지칭합니다.

![Table 2](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/a0f947bd-978c-452f-a1ab-f6345b9d03ec)

4.2 MODEL ANALYSIS
모델의 직관적인 이해를 돕기 위해, 이상 감지 기준(anomaly criterion), 학습 가능한 사전 연관성(learnable prior-association), 그리고 최적화 전략(optimization strategy)이라는 세 가지 핵심 설계에 대한 시각화 자료와 통계 결과를 제공합니다.

![Figure 5](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/d2c95f60-d997-4fe8-856e-33d8edcae5f5)


**Anomaly criterion visualization** <br>
직관적인 사례를 얻기 위하여, 연관 기준(association-based criterion)의 작동 방식과 Lai 등(2021)의 분류에 따른 다양한 이상 유형에서의 성능을 그림 5에서 시각화를 통해 탐구합니다. 연관 기준은 일반적으로 더 뚜렷하게 구별됩니다. 특히, 연관 기준은 정상 부분에서 일관되게 작은 값을 얻을 수 있으며, 이는 점-문맥(point-contextual) 및 패턴-계절적(pattern-seasonal) 사례에서 두드러집니다. 반면, 재구성 기준(reconstruction criterion)의 지터(jitter) 곡선은 탐지 과정을 혼란스럽게 하여 실패합니다. 이는 연관 기준이 이상을 강조하고 정상 및 비정상 점에 구별되는 값을 제공하여 탐지의 정확성을 높이고 거짓 양성 비율(false-positive rate)을 줄일 수 있음을 입증합니다.

![Figure 6](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/a3c0fe7a-3b82-45ff-a391-af5c5874e853)
다양한 유형의 이상(anomalies)에 대해 학습된 스케일(scale) 파라미터(σ) (빨간색으로 강조)

4.2 MODEL ANALYSIS
모델의 직관적인 이해를 돕기 위해, 이상 감지 기준(anomaly criterion), 학습 가능한 사전 연관성(learnable prior-association), 그리고 최적화 전략(optimization strategy)이라는 세 가지 핵심 설계에 대한 시각화 자료와 통계 결과를 제공합니다.

![Figure 5](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/d2c95f60-d997-4fe8-856e-33d8edcae5f5)

Figure 5 다양한 이상(anomaly) 카테고리 시각화 : NeurIPS-TS 데이터셋의 원시(raw) 시리즈와 해당 복원(reconstruction) 및 연관(association) 기준을 표시합니다. 점별(point-wise) 이상은 빨간 원(circle)으로, 패턴별(pattern-wise) 이상은 빨간 선분(segments)으로 나타납니다. 잘못 감지된(wrongly detected) 사례는 빨간 상자(boxes)로 표시됩니다.


**Anomaly criterion visualization** <br>
직관적인 사례를 얻기 위하여, 연관 기준(association-based criterion)의 작동 방식과 Lai 등(2021)의 분류에 따른 다양한 이상 유형에서의 성능을 그림 5에서 시각화를 통해 탐구합니다. 연관 기준은 일반적으로 더 뚜렷하게 구별됩니다. 특히, 연관 기준은 정상 부분에서 일관되게 작은 값을 얻을 수 있으며, 이는 점-문맥(point-contextual) 및 패턴-계절적(pattern-seasonal) 사례에서 두드러집니다. 반면, 재구성 기준(reconstruction criterion)의 지터(jitter) 곡선은 탐지 과정을 혼란스럽게 하여 실패합니다. 이는 연관 기준이 이상을 강조하고 정상 및 비정상 점에 구별되는 값을 제공하여 탐지의 정확성을 높이고 거짓 양성 비율(false-positive rate)을 줄일 수 있음을 입증합니다.

![Figure 6](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/a3c0fe7a-3b82-45ff-a391-af5c5874e853)
다양한 유형의 이상(anomalies)에 대해 학습된 스케일(scale) 파라미터(σ) (빨간색으로 강조)

**Prior-association visualization** <br>
미니맥스(minimax) 최적화 동안, 사전 연관(prior-association)은 시리즈 연관(series-association)에 가까워지도록 학습됩니다. 따라서, 학습된 σ는 시계열의 인접 집중도(adjacent-concentrating degree)를 반영합니다. 그림 6에서, σ가 시계열의 다양한 데이터 패턴에 적응하여 변화한다는 것을 발견했습니다. 특히, 이상 현상(anomalies)의 사전 연관은 일반 시간 지점보다 작은 σ를 가지며, 이는 이상 현상의 인접 집중 유도 편향(adjacent-concentration inductive bias)과 일치합니다.

**Optimization strategy analysis** <br>
재구성 손실(reconstruction loss)만 사용할 경우, 이상(abnormal) 및 정상(normal) 시간 지점들은 인접(adjacent) 시간 지점들에 대한 연관 가중치(association weights)에서 유사한 성능을 보여, 대조 값(contrast value)이 1에 가깝게 나타납니다(표 3). 연관 차이(association discrepancy)를 최대화하는 것은 시리즈 연관(series-association)이 비인접(non-adjacent) 영역에 더 많은 주의를 기울이도록 강제합니다. 그러나, 더 나은 재구성을 얻기 위해서, 이상 현상은 정상 시간 지점들에 비해 훨씬 큰 인접 연관 가중치를 유지해야 하며, 이는 더 큰 대조 값에 해당합니다. 하지만 직접적인 최대화(direct maximization)는 가우시안 커널(Gaussian kernel)의 최적화 문제를 일으키며, 예상대로 정상 및 이상 시간 지점들 사이의 차이를 강하게 확대할 수 없습니다(SMD: 1.15→1.27). 미니맥스(minimax) 전략은 사전 연관(prior-association)을 최적화하여 시리즈 연관에 더 강한 제약을 제공합니다. 따라서, 미니맥스 전략은 직접 최대화보다 더 구별 가능한 대조 값들을 얻어(SMD: 1.27→2.39) 더 나은 성능을 발휘합니다.

https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/c8dce3de-5b8f-4237-9403-1dfd3e4f0e99
