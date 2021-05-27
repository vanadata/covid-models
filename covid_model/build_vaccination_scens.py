import urllib.request as request
from covid_model.model import CovidModel
import pandas as pd
import numpy as np
import re
import copy
from db_utils.conn import db_engine
import datetime as dt
import json


def main():
    model = CovidModel(params='params.json', tslices=[0, 600])
    model.engine = db_engine()

    model.prep(vacc_proj_scen='high-uptake')
    model.write_vacc_to_csv('daily_vaccination_rates.csv')

    model.prep(vacc_proj_scen='low-uptake')
    model.write_vacc_to_csv('daily_vaccination_rates_with_lower_vacc_cap.csv')

if __name__ == '__main__':
    main()
