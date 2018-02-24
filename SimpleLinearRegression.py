### using the code practise provided in:
### https://www.linkedin.com/pulse/machine-learning-regression-day-1-shivam-panchal/

from sklearn.datasets import load_boston
data = load_boston()

# print(data)

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('bmh')

plt.figure(figsize =(15, 6))
plt.hist(data.target)
plt.xlabel('price($1000s)')
plt.ylabel('count')
plt.tight_layout()

for index, feature_name in enumerate(data.feature_names):
    plt.figure(figsize=(4,3))
    plt.scatter(data.data[:, index], data.target)
    plt.ylabel('Price', size=15)
    plt.tight_layout()
    
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf .fit(X_train, Y_train)
predicted = clf.predict(X_test)
expected = Y_test

plt.figure(figsize=(15,6))
plt.scatter(expected, predicted)
plt.plot([0,50], [0,50], '--k')
plt.axis('tight')
plt.xlabel('True Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.tight_layout()
