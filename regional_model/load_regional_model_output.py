import pandas as pd
from db_utils.conn import db_engine


def run():
    dfs = {}
    for fname in ['LPHAMobOutput0527.xlsx', 'MetroMobOutput0524.xlsx']:
        dfs_by_region = pd.read_excel(fname, engine='openpyxl', sheet_name=None)
        df = pd.concat(dfs_by_region).reset_index(level=1, drop=True).set_index('Date', append=True)
        dfs[fname.split('MobOutput')[0]] = df

    combined = pd.concat(dfs)
    combined.index.names = ['region_type', 'region', 'measure_date']
    combined = combined.rename(columns={
        'Day': 't'
        , 'Hospitalizations': 'hosp'
        , 'HospPer100000': 'hosp_per_100k'
        , 'Incidence': 'incidence'
        , 'pImmune': 'immun_share'
        , 'PrevPer100000': 'prev_per_100k'
        , 'CumulativeInfToDate': 'cumu_inf'
        , 'ReEstimate': 're'})

    engine = db_engine()
    combined.to_sql('regional_model_results', engine, schema='stage', method='multi', chunksize=1000, if_exists='replace')


if __name__ == '__main__':
    run()
