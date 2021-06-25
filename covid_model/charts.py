from data_imports import get_deaths, get_hosps_df, get_hosps_by_age, get_deaths_by_age
from model import CovidModel, CovidModelFit
from db import db_engine
import scipy.stats as sps
import matplotlib.pyplot as plt
import copy
import pandas as pd


def actual_hosps(engine, **plot_params):
    hosps = get_hosps_df(engine)
    hosps.plot(**{'color': 'red', 'label': 'Actual Hosps.', **plot_params})


def actual_hosps_by_group(engine, fname, axs, **plot_params):
    df = get_hosps_by_age(engine, fname)
    for g, ax in zip(CovidModel.groups, axs.flat):
        df.xs(g, level='group').plot(ax=ax, **{'label': f'Actual Hosps.', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def actual_deaths_by_group(fname, axs, **plot_params):
    df = get_deaths_by_age(fname)
    for g, ax in zip(CovidModel.groups, axs.flat):
        df.xs(g, level='group').plot(ax=ax, **{'label': f'Actual Deaths', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def actual_deaths(engine, **plot_params):
    deaths_df = get_deaths(engine)
    # deaths = list(deaths_df['cumu_deaths'])
    deaths_df['cumu_deaths'].plot(**{'color': 'red', 'label': 'Actual Deaths', **plot_params})


def actual_new_deaths(engine, rolling=1, **plot_params):
    deaths_df = get_deaths(engine)
    # deaths = list(deaths_df['new_deaths'].rolling(rolling).mean())
    deaths_df['new_deaths'].rolling(rolling).mean().plot(**{'color': 'red', 'label': 'Actual New Deaths', **plot_params})


def total_hosps(model, group=None, **plot_params):
    if group is None:
        hosps = model.total_hosps()
    else:
        hosps = model.solution_ydf.xs(group, level='group')['Ih']
    plt.plot(model.daterange, hosps, **{'c': 'blue', 'label': 'Modeled Hosps.', **plot_params})


def modeled_by_group(model, axs, compartment='Ih', **plot_params):
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, model.solution_ydf.xs(g, level='group')[compartment], **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def new_deaths_by_group(model, axs, **plot_params):
    deaths = model.solution_ydf['D'] - model.solution_ydf['D'].groupby('group').shift(1)
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, deaths.xs(g, level='group'), **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def total_deaths(model, **plot_params):
    modeled_deaths = model.solution_ydf_summed['D']
    modeled_deaths.index = model.daterange
    modeled_deaths.plot(**{'c': 'blue', 'label': 'Modeled Deaths.', **plot_params})


def new_deaths(model, **plot_params):
    modeled_deaths = model.solution_ydf_summed['D'] - model.solution_ydf_summed['D'].shift(1)
    modeled_deaths.index = model.daterange
    modeled_deaths.plot(**{'c': 'blue', 'label': 'Modeled New Deaths', **plot_params})


def actual_vs_modeled_hosps_by_group(actual_hosp_fname, model, **plot_params):
    fig, axs = plt.subplots(2, 2)
    actual_hosps_by_group(model.engine, actual_hosp_fname, axs=axs, c='red', **plot_params)
    modeled_by_group(model, axs=axs, compartment='Ih', c='blue', **plot_params)
    fig.tight_layout()


def actual_vs_modeled_deaths_by_group(actual_deaths_fname, model, **plot_params):
    fig, axs = plt.subplots(2, 2)
    actual_deaths_by_group(actual_deaths_fname, axs=axs, c='red', **plot_params)
    new_deaths_by_group(model, axs=axs, c='blue', **plot_params)
    fig.tight_layout()


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

    model = CovidModel(params='input/params.json', tslices=[0, 700], engine=engine)
    # model.gparams.update({
    #   "N": 5840795,
    #   "groupN": {
    #     "0-19": 1513005,
    #     "20-39": 1685869,
    #     "40-64": 1902963,
    #     "65+": 738958
    #   }})
    # model.set_ef_from_db(1516)
    # model.set_ef_from_db(1644)
    # model.set_ef_from_db(1792)
    # model.efs[9] -= 0.07
    # model.efs[10] += 0.07
    # model.efs[11] += 0.01
    model.set_ef_by_t(model.efs)
    model.prep()
    model.solve_seir()
    actual_hosps(engine)
    total_hosps(model)
    # actual_vs_modeled_hosps_by_group('input/hosps_by_group_20210611.csv', model)
    # actual_vs_modeled_deaths_by_group('input/deaths_by_group_20210614.csv', model)
    plt.show()

    # fig, axs = plt.subplots(2, 2)
    # actual_deaths_by_group('input/deaths_by_group_20210614.csv', axs=axs)
    # plt.show()

    # actual_hosps(engine)
    # model = CovidModel.from_fit(engine, 1150)
    # model.prep()
    # model.solve_seir()
    # total_hosps(model)
    # plt.legend(loc='best')
    # plt.show()
    # exit()

    # actual_new_deaths(engine, rolling=7, label='Actual New Deaths (7-day avg.)')



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

    # actual_hosps(engine)
    # model = CovidModel('input/params.json', [0, 600], engine=engine)
    # model.set_ef_from_db(1225)
    # model.prep()
    # model.solve_seir()
    # total_hosps(model)
    # uq_spaghetti(CovidModelFit.from_db(engine, 1225), sample_n=200)

    # plt.legend(loc='best')
    # plt.xlabel('Days')
    # # plt.xlim('2021-02-01', '2021-05-17')
    # # plt.ylim(0, 25)
    # plt.grid()
    # plt.show()


