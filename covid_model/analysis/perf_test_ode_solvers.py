import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import json
from covid_model.db import db_engine
from covid_model.model import CovidModelFit, CovidModel
from charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates
from time import perf_counter
from timeit import timeit

from covid_model.model_specs import CovidModelSpecifications
from vacc_scen_comparison import build_mab_prevalence

if __name__ == '__main__':
    engine = db_engine()

    # tc = [0.197316961609296,0.454544487240692,0.685069601735486,0.856637090778988,0.861426345809302,0.854460394603348,0.915440489892051,0.892098858977771,0.739492039679914,0.637422871591765,0.798469951938621,0.851835863406033,0.813730276031466,0.813700514087861,0.783855416121104,0.660199184274321,0.651709658632707,0.625130887314897,0.657608378080961,0.73156467338279,0.849593417871158,0.848625913420463,0.833159585040641,0.780310629600215,0.883445159121751,0.7398057107140255,0.8609249950340425,0.7158116763408557,0.7622678721711535,0.6177221761571133,0.6873098783020569,0.8040523766488482,0.7918391066823716,0.8684805695277086,0.833325279039802,0.7805306079204452,0.8034081654073754,0.6607799134829009,0.723623378637067,0.7378782272700403,0.8027115573066969,0.8060396354961565,0.7310349476535561,0.740356922881879,0.7395080997027447]

    cms = CovidModelSpecifications.from_db(engine, 65)

    model = CovidModel(engine, end_date=dt.date(2020, 7, 6))
    model.prep(specs=cms)

    # model.set_tc(tc)
    # model.prep(tc=tc)

    fig, ax = plt.subplots()
    print('Prepping model...')
    print(timeit('model.prep()', number=1, globals=globals()), 'seconds to prep model.')

    # for method in ['RK45', 'RK23', 'BDF', 'LSODA', 'Radau', 'DOP853']:
    for method in ['RK45']:
        print(timeit('model.solve_seir(method=method)', number=1, globals=globals()), 'seconds to run model.')

    # model.write_to_db(engine)

    modeled(model, 'Ih')
    actual_hosps(engine)
    plt.show()
