# Demonstrate that price_promotion, mailing_ind, topo_in, ilha_ind, monofolha_ind, price_grp, card_grp, lxpy
# are useless in this case
import pandas as pd

print 'Demonstrating that some columns are useless for the problem'

df = pd.read_pickle('h5/AggregatedDataset.pkl')

stat = df.describe()

print stat

del stat
del df
