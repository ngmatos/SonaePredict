import python.Sampling as Sample
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

kf = Sample.kf

train_set = Sample.train_set
test_set = Sample.test_set

for train in train_set:
    print('train -> ' + train)

'''
lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

n_folds = kf.n_splits
print('n folds -> ' + str(n_folds))

for alpha in alphas:
    print(alpha)
    lasso.alpha = alpha
    this_scores = cross_val_score(lasso, train_set, test_set, cv=n_folds, n_jobs=1)
    print('done for alpha: ' + str(alpha))
    scores.append(np.mean(this_scores))
    print(str(np.mean(this_scores)))
    scores_std.append(np.std(this_scores))
'''
