# 선형 -> 비선형 효과적

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
"""
pf = PolynomialFeatures(degree=2, include_bias=False) # 다항식, default = True // degree=3 이면 3차원
x_pf = pf.fit_transform(x)
print(x_pf)
"""
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

# default = True
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

###  통상적으로
# 선형모델(lr등)에 쓸 경우에는 include_bias=True를 써서 1만 있는 컬럼을 만드는게 좋음
# 왜냐하면 y = wx + b 의 bias = 1의 역할을 하기 때문
# 비선형모델 (rf, xgb 등)에 쓸 경우에는 inculde_bias = False가 좋음

pf = PolynomialFeatures(degree=3, include_bias=False) # 다항식, default = True // degree=3 이면 3차원
x_pf = pf.fit_transform(x)
print(x_pf)
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]

#  X1, x2, x1^2,  x1*x2,  x2^2,  x1^3,  x1^2*x2,  x1*x2^2,  x2^*