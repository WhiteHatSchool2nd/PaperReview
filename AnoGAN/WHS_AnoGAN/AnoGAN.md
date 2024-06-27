[[unsupervised learning 방식]]
[[GAN을 사용한 Anomaly Detection]]

## Train
![[Pasted image 20240627204842.png]]

## Inference
- 불량을 판단하기 위해 입력으로 받은 이미지와 가장 유사한 이미지를 Generator가 만들어내야 한다
	- 두 이미지 차이를 근거로 anormal score를 판단

![[Pasted image 20240627205530.png]]
- loss가 클수록 불량으로 판단 