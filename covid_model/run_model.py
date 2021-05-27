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
    fit_id = 1044

    engine = db_engine()
    model = CovidModel.from_fit(engine, fit_id, params='params.json')
    model.engine = engine
    last_ef = model.ef_by_slice[-1]
    model.add_tslice(600, last_ef)
    tags = {'purpose': 'Test', 'batch': 'test_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')}
    # model.gparams['max_vacc'] = {"0-19": 0.239, "20-39": 0.56, "40-64": 0.62, "65+": 0.94}
    print(tags)
    model.prep()
    model.write_gparams_lookup_to_csv('gparams_lookup.csv')
    # model.solve_seir()
    # json.dump(model.gparams_lookup, open('gparams_lookup.json', 'w'))
    # model.write_to_db(engine, tags=tags)


if __name__ == '__main__':
    main()
