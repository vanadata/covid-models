from hospitalizations import get_hosps, get_deaths, get_hosps_df
from model import CovidModel, CovidModelFit
from db import db_engine
import scipy.stats as sps
import matplotlib.pyplot as plt
import copy
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder


def actual_hosps(engine, **plot_params):
    hosps = get_hosps_df(engine)
    hosps.plot(**{'color': 'red', 'label': 'Actual Hosps.', **plot_params})


def actual_deaths(engine, **plot_params):
    deaths_df = get_deaths(engine)
    # deaths = list(deaths_df['cumu_deaths'])
    deaths_df['cumu_deaths'].plot(**{'color': 'red', 'label': 'Actual Deaths', **plot_params})


def actual_new_deaths(engine, rolling=1, **plot_params):
    deaths_df = get_deaths(engine)
    # deaths = list(deaths_df['new_deaths'].rolling(rolling).mean())
    deaths_df['new_deaths'].rolling(rolling).mean().plot(**{'color': 'red', 'label': 'Actual New Deaths', **plot_params})


def total_hosps(model, **plot_params):
    plt.plot(model.daterange, model.total_hosps(), **{'c': 'blue', 'label': 'Modeled Hosps.', **plot_params})


def total_deaths(model, **plot_params):
    modeled_deaths = model.solution_ydf_summed['D']
    modeled_deaths.index = model.daterange
    modeled_deaths.plot(**{'c': 'blue', 'label': 'Modeled Deaths.', **plot_params})


def new_deaths(model, **plot_params):
    modeled_deaths = model.solution_ydf_summed['D'] - model.solution_ydf_summed['D'].shift(1)
    modeled_deaths.index = model.daterange
    modeled_deaths.plot(**{'c': 'blue', 'label': 'Modeled New Deaths', **plot_params})


# UQ sqaghetti plot
def uq_spaghetti(fit, sample_n=100, tmax=600, **plot_params):
    # get sample TC values
    fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_efs, cov=fit.fitted_efs_cov)
    samples = fitted_efs_dist.rvs(sample_n)

    # for each sample, solve the model and add a line to the plot
    model = copy.copy(fit.model)
    model.add_tslice(tmax, 0)
    model.prep()
    for sample_fitted_efs in samples:
        model.set_ef_by_t(list(fit.fixed_efs) + list(sample_fitted_efs) + [sample_fitted_efs[-1]])
        model.solve_seir()
        plt.plot(model.daterange, model.total_hosps(), **{'color': 'darkblue', 'alpha': 0.03, **plot_params})


if __name__ == '__main__':
    engine = db_engine()

    # actual_hosps(engine)
    # model = CovidModel.from_fit(engine, 1150)
    # model.prep()
    # model.solve_seir()
    # total_hosps(model)
    # plt.legend(loc='best')
    # plt.show()
    # exit()

    # actual_new_deaths(engine, rolling=7, label='Actual New Deaths (7-day avg.)')
    #
    # model = CovidModel.from_fit(engine, 1150)
    # model.prep()
    # model.solve_seir()
    # new_deaths(model, c='royalblue', label='Current Parameters')
    #
    # model = CovidModel.from_fit(engine, 1150)
    # model.gparams['variants']['b117']['multipliers']['dh'] = 1.0
    # model.gparams['variants']['b117']['multipliers']['dnh'] = 1.0
    # model.prep()
    # model.solve_seir()
    # new_deaths(model, c='orange', label='With No B117 Impact on Death')
    #
    # model = CovidModel.from_fit(engine, 1150)
    # for vacc in ['mrna', 'jnj']:
    #     for shot in ['first_shot', 'second_shot']:
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dh'] = 0.4
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dnh'] = 0.2
    # model.prep()
    # model.solve_seir()
    # new_deaths(model, c='lightseagreen', label='With 90% Vacc. Protection vs. Death (instead of 75%)')
    #
    # model = CovidModel.from_fit(engine, 1150)
    # model.gparams['variants']['b117']['multipliers']['dh'] = 1.0
    # model.gparams['variants']['b117']['multipliers']['dnh'] = 1.0
    # for vacc in ['mrna', 'jnj']:
    #     for shot in ['first_shot', 'second_shot']:
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dh'] = 0.4
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dnh'] = 0.2
    # model.prep()
    # model.solve_seir()
    # new_deaths(model, c='tomato', label='With Both Vacc & Variant Adjust.')

    # model = CovidModel.from_fit(engine, 1150)
    # for vacc in ['mrna', 'jnj']:
    #     for shot in ['first_shot', 'second_shot']:
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dh'] = 0.4
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dnh'] = 0.2
    # model.gparams['dh']['0-19'] *= 0.1
    # model.gparams['dnh']['0-19'] *= 0.1
    # model.gparams['dh']['20-39'] *= 0.1
    # model.gparams['dnh']['20-39'] *= 0.1
    # model.gparams['dh']['40-64'] *= 0.1
    # model.gparams['dnh']['40-64'] *= 0.1
    # model.gparams['dh']['65+'] *= 1.12
    # model.gparams['dnh']['65+'] *= 1.12
    # model.prep()
    # model.solve_seir()
    # total_deaths(model, c='tab:purple', label='With Under-40 Death-Rate Reduced by 20%')

    actual_hosps(engine)
    uq_spaghetti(CovidModelFit.from_db(engine, 1225), sample_n=200)

    plt.legend(loc='best')
    plt.xlabel('Days')
    # plt.xlim('2021-02-01', '2021-05-17')
    # plt.ylim(0, 25)
    plt.grid()
    plt.show()


