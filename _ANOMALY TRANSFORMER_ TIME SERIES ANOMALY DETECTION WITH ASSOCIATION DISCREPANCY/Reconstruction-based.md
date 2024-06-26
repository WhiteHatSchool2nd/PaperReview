
재구성 기반 판별 방식(Reconstruction-based method)은 이상을 판별할 데이터를 저차원 형태의 잠재 구조(latent structure)를 획득하고, 그 이후 인위적으로 재구성한 데이터를 생성하기 위한 모델을 사용한다.

최근에는 딥러닝 기반기술인 Auto-Encoder(AE), Variational Auto-Encoder(VAE), LSTM 기반 인코더(encoder)-디코더(decoder) 구조를 활용한 알고리즘을 활용한 재구성 기반 이상(anomaly) 판별 방식이 많이 소개되었다.

# 예시 

![[Pasted image 20240626163049.png]]
출처 : https://hoya012.github.io/blog/anomaly-detection-overview-1/

- 해당 그림은 AE기반의 이상탐지 알고리즘의 개념도를 나타낸다. 원본 입력(x)과 오토인코더로 재구성한 데이터(x’) 간의 차이(diff)를 계산하여 이상(anomaly) 여부를 판단한다. 오토인코더의 [[인코더]](encoder)를 통해 저차원의 잠재 변수(latent variable)를 얻을 수 있다. 이 과정에서 원본 입력(x)이 이상(anomaly) 데이터라면 잠재 변수에 값이 손실된 값을 얻게 되어 디코딩 과정에서 제대로 복원이 어려워진다.