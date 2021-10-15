from data_imports import get_deaths, get_hosps_df, get_hosps_by_age, get_deaths_by_age
from model import CovidModel, CovidModelFit
from db import db_engine
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib import cm, colors
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd
from time import perf_counter


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


def actual_deaths_by_group(engine, axs, **plot_params):
    df = get_deaths_by_age(engine)
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


def modeled(model, compartments, transform=lambda x: x, **plot_params):
    if type(compartments) == str:
        compartments = [compartments]
    plt.plot(model.daterange, transform(model.solution_sum('seir')[compartments].sum(axis=1)), **{'c': 'blue', **plot_params})


def modeled_by_group(model, axs, compartment='Ih', **plot_params):
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, model.solution_ydf.xs(g, level='group')[compartment], **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def transmission_control(model, **plot_params):
    plt.plot(model.tslices[:-1], model.efs, **plot_params)


def re_estimates(model, **plot_params):
    plt.plot(model.daterange, model.re_estimates, **plot_params)


def new_deaths_by_group(model, axs, **plot_params):
    deaths = model.solution_ydf['D'] - model.solution_ydf['D'].groupby('group').shift(1)
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, deaths.xs(g, level='group').shift(6), **{'c': 'blue', 'label': 'Modeled', **plot_params})
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


def actual_vs_modeled_deaths_by_group(engine, model, **plot_params):
    fig, axs = plt.subplots(2, 2)
    actual_deaths_by_group(engine, axs=axs, c='red', **plot_params)
    new_deaths_by_group(model, axs=axs, c='blue', **plot_params)
    fig.tight_layout()


# UQ TC plot
def uq_tc(fit: CovidModelFit, sample_n=100, **plot_params):
    # get sample TC values
    fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_efs, cov=fit.fitted_efs_cov)
    samples = fitted_efs_dist.rvs(sample_n)
    for sample_fitted_efs in samples:
        plt.plot(fit.tslices[:-1], list(fit.fixed_efs) + list(sample_fitted_efs), **{'marker': 'o', 'linestyle': 'None', 'color': 'darkorange', 'alpha': 0.025, **plot_params})

    plt.xlabel('t')
    plt.ylabel('TCpb')


# UQ sqaghetti plot
def uq_spaghetti(fit, sample_n=100, tmax=600,
                 tc_shift=0, tc_shift_days=70, tc_shift_date=dt.date.today() + dt.timedelta(2) + dt.timedelta((6-dt.date.today().weekday()) % 7),
                 compartments='Ih', **plot_params):
    # get sample TC values
    fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_efs, cov=fit.fitted_efs_cov)
    samples = fitted_efs_dist.rvs(sample_n)

    # for each sample, solve the model and add a line to the plot
    model = CovidModel(fit.tslices, engine=engine)
    model.add_tslice(tmax, 0)
    model.prep()
    for i, sample_fitted_efs in enumerate(samples):
        print(i)
        # model = base_model.prepped_duplicate()
        current_ef = sample_fitted_efs[-1]
        if tc_shift_days is None:
            model.set_ef_by_t(list(fit.fixed_efs) + list(sample_fitted_efs) + [current_ef + tc_shift])
        else:
            model.set_ef_by_t(list(fit.fixed_efs) + list(sample_fitted_efs) + [current_ef])
            for i, tc_shift_for_this_day in enumerate(np.linspace(0, tc_shift, tc_shift_days)):
                model.ef_by_t[(tc_shift_date - model.datemin.date()).days + i] += tc_shift_for_this_day
            for t in range((tc_shift_date - model.datemin.date()).days + tc_shift_days, tmax):
                model.ef_by_t[t] += tc_shift
        # base_model.set_ef_by_t(list(fit.fixed_efs) + list(sample_fitted_efs) + [sample_fitted_efs[-1] + tc_shift])
        model.solve_seir()
        modeled(model, compartments=compartments, **{'color': 'darkblue', 'alpha': 0.025, **plot_params})
        # plt.plot(model.daterange, model.total_hosps(), **{'color': 'darkblue', 'alpha': 0.025, **plot_params})

# UQ histogram plot
def uq_histogram(fit: CovidModelFit, sample_n=100, compartments='Ih', from_date=dt.datetime.now(), to_date=dt.datetime.now()+dt.timedelta(days=90)):
    # get sample TC values
    fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_efs, cov=fit.fitted_efs_cov)
    samples = fitted_efs_dist.rvs(sample_n)

    # for each sample, solve the model and add a line to the plot
    model = CovidModel(fit.tslices, engine=engine)
    model.add_tslice((to_date - model.datemin).days, 0)
    model.prep()
    for i, sample_fitted_efs in enumerate(samples):
        print(i)
        model.set_ef_by_t(list(fit.fixed_efs) + list(sample_fitted_efs) + [sample_fitted_efs[-1] + tc_shift])
        model.solve_seir()


def tc_for_given_r_and_vacc(solved_model: CovidModel, t, r, vacc_share):
    y = solved_model.solution_ydf_summed.loc[t]
    p = solved_model.gparams_lookup[t]
    current_vacc_immun = y['V'] / p[None]['N']
    acq_immun = (y['R'] + y['RA']) / p[None]['N'] * (1 + current_vacc_immun)  # adding back in the acq immun people who were moved to vacc immun
    eff_beta_at_tc100 = p[None]['beta'] * p[None]['rel_inf_prob'] * (y['I'] * p[None]['lamb'] + y['A'] * 1) / (y['I'] + y['A'])
    r0_at_tc100 = eff_beta_at_tc100 / p[None]['gamma']

    jnj_share = 0.078
    vacc_immun = (jnj_share*0.72 + (1-jnj_share)*0.9) * 0.9 * vacc_share
    r_at_tc100_space = r0_at_tc100 * (1 - vacc_immun - acq_immun + vacc_immun*acq_immun)
    return 1 - r / r_at_tc100_space


def r_for_given_tc_and_vacc(solved_model: CovidModel, t, tc, vacc_share):
    y = solved_model.solution_ydf_summed.loc[t]
    p = solved_model.gparams_lookup[t]
    current_vacc_immun = y['V'] / p[None]['N']
    acq_immun = (y['R'] + y['RA']) / p[None]['N'] * (1 + current_vacc_immun)  # adding back in the acq immun people who were moved to vacc immun
    # new_acq_immun_by_t = solved_model.solution_ydf_summed['E'] / 4.0 * 0.85 * np.exp(-(t - solved_model.solution_ydf_summed.index.values)/2514)
    # acq_immun = new_acq_immun_by_t.cumsum().loc[t] / p[None]['N']
    eff_beta_at_tc100 = p[None]['beta'] * p[None]['rel_inf_prob'] * (y['I'] * p[None]['lamb'] + y['A'] * 1) / (y['I'] + y['A'])
    r0_at_tc100 = eff_beta_at_tc100 / p[None]['gamma']

    jnj_share = 0.078
    vacc_immun = (jnj_share*0.72 + (1-jnj_share)*0.9) * 0.9 * vacc_share
    r_at_tc100_space = r0_at_tc100 * (1 - vacc_immun - acq_immun + vacc_immun*acq_immun)
    return r_at_tc100_space * (1 - tc)


def r_equals_1(solved_model: CovidModel, t=None):
    if t is None:
        t = (dt.datetime.now() - solved_model.datemin).days

    increment = 0.001
    vacc_space = np.arange(0.3, 0.95, increment)
    tc_space = np.arange(1.0, 0.4, -1 * increment)
    r_matrix = np.array([r_for_given_tc_and_vacc(solved_model, t, tc, vacc_space) for tc in tc_space])
    r_df = pd.DataFrame(data=r_matrix, index=['{:.0%}'.format(tc) for tc in tc_space], columns=['{:.0%}'.format(v) for v in vacc_space])
    # r_df = pd.DataFrame(data=r_matrix, index=tc_space, columns=vacc_immun_space)

    fig, ax = plt.subplots()
    sns.heatmap(r_df, ax=ax, cmap='Spectral_r', center=1.0, xticklabels=50, yticklabels=50, vmax=3.0, cbar_kws={"ticks": [0.0, 1.0, 2.0, 3.0]})
    fig.get_axes()[1].set_yticklabels(['Re = 0.0', 'Re = 1.0', 'Re = 2.0', 'Re = 3.0'])
    # sns.heatmap(r_df, ax=ax, cmap='bwr', center=1.0, xticklabels=50, yticklabels=50, vmax=3.0)

    # plot R = 1
    tc_for_r_equals_1_space = tc_for_given_r_and_vacc(solved_model, t, 1.0, vacc_space)
    ax.plot(range(len(vacc_space)), (tc_space.max() - tc_for_r_equals_1_space) // increment, color='navy')
    ax.set_xlabel('% of Population Fully Vaccinated')
    ax.set_ylabel('TCpb')
    ax.set_title('Re by Vaccination Rate and TCpb')
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # plot current point
    current_vacc = solved_model.vacc_rate_df.loc[t, 'cumu'].sum() / solved_model.gparams_lookup[t][None]['N']
    current_tc = solved_model.ef_by_t[t]
    current_r = r_for_given_tc_and_vacc(solved_model, t, current_tc, current_vacc)
    print(f"Current Vacc.: {'{:.0%}'.format(current_vacc)}")
    print(f"Current TCpb: {'{:.0%}'.format(current_tc)}")
    print(f"Current Re: {round(current_r, 2)}")
    ax.plot((current_vacc - vacc_space.min()) // increment, (tc_space.max() - current_tc) // increment, marker='o', color='navy')

    # formatting
    ax.grid(color='white')


def vaccination(fname='output/daily_vaccination_by_age (old).csv', **plot_params):
    df = pd.read_csv(fname, parse_dates=['measure_date']).set_index(['vacc_scen', 'group', 'measure_date'])
    first_shot_rate = df.loc['current trajectory', 'first_shot_rate'].groupby('measure_date').sum().rolling(7).mean()
    first_shot_rate.plot(**plot_params)


if __name__ == '__main__':
    engine = db_engine()

    model = CovidModel([0, 700], engine=engine)
    # model.set_ef_from_db(5324)
    model.set_ef_from_db(0)
    # model.efs[-1] = 1

    model.prep(params='input/params.json')
    t0 = perf_counter()
    model.solve_seir()
    t1 = perf_counter()
    print(f'Solved ODE in {t1 - t0} seconds.')
    model.write_to_db(engine)
    modeled(model, 'Ih')
    actual_hosps(engine)
    # print(model.solution_sum(['seir', 'vacc'])[[('Ih', 'vacc_fail')]].sum(axis=1))
    # plt.plot(model.daterange, model.solution_sum(['seir', 'vacc'])[[('Ih', 'vacc_fail')]].sum(axis=1), **{'c': 'blue'})
    plt.show()
    exit()

    # actual_vs_modeled_hosps_by_group('input/hosps_by_group_20210611.csv', model)
    # plt.show()
    # exit()
    #
    # actual_vs_modeled_deaths_by_group(engine, model)
    # plt.show()

    fig, ax = plt.subplots()
    plt.grid(color='lightgray')
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = ['tomato', 'royalblue', 'slategray']
    # fit = CovidModelFit.from_db(engine, 5324)
    fit = CovidModelFit.from_db(engine, 5457)
    for i, tc_shift in enumerate([-0.10, 0.10, 0]):
        # uq_spaghetti(fit, sample_n=200, tmax=700, tc_shift=tc_shift, tc_shift_days=56, color=colors[i], alpha=0.05, compartments='Ih')
        uq_spaghetti(fit, sample_n=200, tmax=700, tc_shift=tc_shift, tc_shift_days=56, color=colors[i], alpha=0.05, compartments='Ih')
        # uq_spaghetti(fit, sample_n=200, tmax=700, tc_shift=tc_shift, color=colors[i], alpha=0.05, compartments='D')
        # uq_histogram(fit)
        # exit()
    locator = mdates.MonthLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylabel('Daily Patients Hospitalized with COVID-19')
    # ax.set_ylabel('Cumulative Deaths from COVID-19')
    plt.show()

    # uq_tc(CovidModelFit.from_db(engine, 4898), sample_n=10)

    # vaccination('output/daily_vaccination_by_age_old.csv', label='Old')
    # vaccination(label='New')
    # plt.legend(loc='best')
    # plt.show()

    # model.prep(vacc_proj_scen='current trajectory')
    # model.solve_seir()
    # r_equals_1(model)
    # plt.tight_layout()
    # plt.show()

    # model = CovidModel(params='input/params.json', tslices=[0, 700], engine=engine)
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
    # model.set_ef_by_t(model.efs)
    # model.prep()
    # model.solve_seir()
    # actual_hosps(engine)
    # total_hosps(model)
    # actual_vs_modeled_hosps_by_group('input/hosps_by_group_20210611.csv', model)
    # actual_vs_modeled_deaths_by_group('input/deaths_by_group_20210614.csv', model)
    # plt.show()

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
    # model = CovidModel('input/params.json', [0, 700], engine=engine)
    # model.set_ef_from_db(1992)
    # model.prep(vacc_proj_scen='current trajectory')
    # model.solve_seir()
    # total_hosps(model)
    # uq_spaghetti(CovidModelFit.from_db(engine, 2330), sample_n=200, tmax=600)
    # uq_spaghetti(CovidModelFit.from_db(engine, 1992), sample_n=200, tmax=600)

    # actual_hosps(engine, color='black')
    # model = CovidModel('input/params.json', [0, 700], engine=engine)
    # model.set_ef_from_db(3472)
    # model.prep(vacc_proj_scen='current trajectory')
    # model.solve_seir()
    # modeled(model, ['Ih'], color='red')
    #
    # plt.show()

    # fig, axs = plt.subplots(2, 2)

    # fits = {'6-12 mo. immunity': 2470, '12-24 mo. immunity': 2467, 'indefinite immunity': 2502}
    # colors = ['r', 'b', 'g', 'black', 'orange', 'pink']
    # for i, (label, fit_id) in enumerate(fits.items()):
    #     fit1 = CovidModelFit.from_db(conn=engine, fit_id=fit_id)
    #     model1 = CovidModel(fit1.model_params, [0, 600], engine=engine)
    #     model1.set_ef_from_db(fit_id)
    #     model1.prep()
    #     model1.solve_seir()
    #     # modeled(model1, compartments=['E'], transform=lambda x: x.cumsum()/4.0, c=colors[i], label=label)
    #     # modeled(model1, compartments=['V', 'Vxd'], transform=lambda x: x/5813208.0, c=colors[i], label=label)
    #     modeled(model1, compartments=['I', 'A'], c=colors[i], label=label)
    #     # transmission_control(model1, c=colors[i], label=label)
    #     # re_estimates(model1, c=colors[i], label=label)
    #     # modeled_by_group(model1, axs=axs, compartments=['I', 'A'], c=colors[i], label=label)
    #
    # plt.legend(loc='best')
    # plt.ylabel('People Infected')
    # plt.show()

    # plt.legend(loc='best')
    # plt.xlabel('Days')
    # # plt.xlim('2021-02-01', '2021-05-17')
    # # plt.ylim(0, 25)
    # plt.grid()
    # plt.show()


