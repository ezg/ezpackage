import ezpackage
import pandas as pd

ezpackage.__author__

df = pd.read_csv("tracks.csv")
df = df[df["EVENT"] != "error"]

g1 = set(df[(df['USER_AGENT_BROWSER_OS'] == 'iOS') | (
    df['USER_AGENT_BROWSER_OS'] == 'Android OS')]['USER_ID'])
g2 = set(df['USER_ID']) - g1

user_col = 'USER_ID'
event_col = 'EVENT'

groups = ezpackage.funnel.funnel(data=df, targets=[
    'account_created', 'operator_created'], event_col=event_col, index_col=user_col, groups=(g1, g2),
    group_names=('mobile', 'non-mobile'))

ezpackage.funnel.plot(groups)
