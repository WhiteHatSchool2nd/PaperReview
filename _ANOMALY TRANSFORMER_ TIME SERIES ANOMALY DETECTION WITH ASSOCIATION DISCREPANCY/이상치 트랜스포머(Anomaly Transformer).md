
Anomaly Transformer는 [[RNN]]이 아닌 [[트랜스포머(Transformers)]]를 사용해서 시계열 데이터에서 특성(feature)을 추출한다.

Transformer로 추출한 특성에서 입력 시계열을 복원한다. 이때 함께 추출한 시계열 축 데이터 연관성 분포를 우리가 알고 있는 확률 분포에 근사시킨다.

해당 논문의 저자는 [[가우시안 분포]]를 사용하는 것이 가장 좋은 결과를 보여줬다고 밝혔다. 

# 예시

'I am a cute cat'이라는 문장을 다음과 같이 학습할 수 있다. 
![[Pasted image 20240626194857.png]]

- 모델은 I와 cat이 연관성이 높다고 말할 것이다. 

여기서 Anomaly Transformer의 아이디어가 추가된다. 정상적인 흐름을 가진 시계열 데이터라면 분포가 고르게 나타날 것이다. 

반대로 이상 데이터라면 오직 인접한 데이터들과 높은 연관성을 지닐 것이다. 


![[Pasted image 20240626195126.png]]
- 정상인 시계열 데이터의 모습.

![[Pasted image 20240626195149.png]]
- 이상 데이터가 포함된 시계열 데이터의 모습.

이처럼 Anomaly Transformer [[연관성 차이(Association Discrepancy)]]를 통해 이상탐지를 하게 된다. 

# 구조

이상 트랜스포머는 [[사전-연관성(Prior-Association)]]과 [[시리즈-연관성(Series-Association)]]을 모델링하는 두 가지 분기 구조를 포함한다. 

이 모델은 Anomaly-Attention(이상-주의) 메커니즘과 [[피드-포워드(Feed-Forward) 레이어]]를 번갈아 가며 쌓아 올린 구조를 특징으로 한다. 

모델은 총 L개의 레이어(Layers)를 포함하며, 입력 시계열 데이터(Input Time Series Data)가 N의 길이를 가진다고 가정할 때, l번째 레이어의 방정식은 다음과 같다.
$$
Z_l = \text{Layer-Norm} \left( \text{Anomaly-Attention} \left( X_{l-1} \right) + X_{l-1} \right)
$$

$$
X = \text{Layer-Norm} \left( \text{Feed-Forward}(Z) + Z \right)
$$


![Figure 1](https://github.com/WhiteHatSchool2nd/PaperReview/assets/165824811/94f4af43-7541-48cf-9125-87b53ec7e603) 

