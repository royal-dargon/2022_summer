import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x
from sympy import diff, dsolve, simplify, Function

data = pd.read_excel("CUMCM-2018-Problem-A-Chinese-Appendix.xlsx", sheet_name='附件2', skiprows=[0])
true_data = np.array(data)

y = Function('y')
eq = diff()


plt.scatter(true_data[:, 0], true_data[:, 1], 1)
plt.show()


