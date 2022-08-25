import matplotlib as mpl
from time import time
import ezpackage
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

ezpackage.__author__

# load sample user behavior data as a pandas dataframe:
df = pd.read_csv("tracks.csv")
df = pd.read_csv("tracks.csv")
df = df[df["EVENT"] != "error"]
df = df[(df['USER_AGENT_BROWSER_OS'] != 'iOS') & (
    df['USER_AGENT_BROWSER_OS'] != 'Android OS')]

df.sort_values(by='TIMESTAMP')
g1 = set(df[(df['USER_AGENT_BROWSER_OS'] == 'iOS') | (
    df['USER_AGENT_BROWSER_OS'] == 'Android OS')]['USER_ID'])
g2 = set(df['USER_ID']) - g1


user_col = 'USER_ID'
event_col = 'EVENT'
time_col = 'TIMESTAMP'

groups = ezpackage.matrix.matrix(
    df, max_steps=5, event_col=event_col, index_col=user_col, time_col=time_col, thresh=0.01)
plt.savefig("foo.svg")
plt.show()
