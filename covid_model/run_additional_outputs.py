import pandas as pd
import numpy as np
import datetime as dt
from db import db_engine
from data_imports import get_hosps_df, get_vaccinations_by_county

if __name__ == '__main__':
    engine = db_engine()

    hosps = pd.DataFrame(get_hosps_df(engine)).reset_index().rename(columns={'currently_hospitalized': 'Iht', 'measure_date': 'date'})
    hosps['time'] = ((pd.to_datetime(hosps['date']) - dt.datetime(2020, 1, 24)) / np.timedelta64(1, 'D')).astype(int)
    hosps[['time', 'date', 'Iht']].to_csv('output/CO_EMR_Hosp.csv', index=False)

    vacc_by_county = get_vaccinations_by_county(engine)
    vacc_by_county.to_csv('output/daily_vaccination_by_age_by_county.csv', index=False)