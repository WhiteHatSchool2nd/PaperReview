import matplotlib.pyplot as plt

# 수식
formula = r'$X_l \in \mathbb{R}^{N \times d_{\text{model}}}, \quad l \in \{1, \cdots, L\}$'

# 이미지 생성
fig = plt.figure()
fig.text(0.1, 0.5, formula, fontsize=12)

plt.axis('off')
plt.grid(False)

# 이미지로 저장
plt.savefig("layer_output_dimension.png", bbox_inches='tight', pad_inches=0.1)
plt.close()
