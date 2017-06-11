import python.sampling.KFold as Sample
from sklearn.linear_model import Lasso

kf = Sample.kf

X_train, X_test, y_train, y_test = Sample.X_train, Sample.X_test, Sample.y_train, Sample.y_test

# Run Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Performance
print('Lasso score (R^2):', lasso.score(X_test, y_test))
