
시계열을 분석해서 다음에 올 값을 예측하고 실제 값과의 차이를 시계열 이상탐지에 사용하는 방법이다. 

예측 기반 판별 방식은 앞으로 발생할 이벤트를 예측하기 때문에 시계열 분석에 효과적인 알고리즘을 사용하여 구현할 수 있다.

대표적인 알고리즘으로 ARIMA, Holt-Winters, FDA, HTM (Hierarchical Temporal Memory), vanilla RNN, LSTM등으로 구현할 수 있다.

# 예시) HTM을 사용한 이상탐지
![[Pasted image 20240626150232.png]]
출처 : https://www.sciencedirect.com/science/article/pii/S0925231217309864

HTM 네트워크는 입력 데이터 를 사용하여 현재 입력을 나타내는 벡터 와 예측을 출력하는 벡터를 출력한다.


# 참고 문헌
https://spri.kr/posts/view/23193?code=data_all&study_type=industry_trend