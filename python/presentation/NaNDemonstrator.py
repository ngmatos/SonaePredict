# Demonstrate that price_promotion, mailing_ind, topo_in, ilha_ind, monofolha_ind, price_grp, card_grp, lxpy
# are useless in this case
import pandas as pd
import python.Config as Config
from IPython.display import HTML, display


print('Demonstrating that some columns are useless for the problem')

df = pd.read_pickle(Config.H5_PATH + '/AggregatedDataset.pkl')

stat = df.describe()

print(stat)

del stat
del df
