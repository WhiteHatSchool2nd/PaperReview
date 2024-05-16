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


