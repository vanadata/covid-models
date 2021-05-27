import pandas as pd


def run():
    dfs_by_region = pd.read_excel('LPHAMobOutput0510.xlsx', engine='openpyxl', sheet_name=None)
    df = pd.concat(dfs_by_region).reset_index(level=1, drop=True).set_index('Date', append=True)
    df.to_csv('regional_model_output.csv')


if __name__ == '__main__':
    run()
