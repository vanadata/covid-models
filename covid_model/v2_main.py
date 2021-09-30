from ode_builder import *
from model import CovidModel
from charts import *
import json

def seir_model(model):
    attr = OrderedDict({'seir': ['S', 'E', 'I', 'Ih', 'A', 'R', 'RA', 'D'], 'age': ['0-19', '20-39', '40-64', '65+'], 'vacc': ['unvacc', 'mrna', 'jnj']})
    # attr = OrderedDict({'seir': ['S', 'E', 'I', 'Ih', 'A', 'R', 'RA', 'D', 'V'], 'age': ['0-19', '20-39', '40-64', '65+']})

    for t in model.trange:
        vacc_per_unvacc[t] = {}
        for age in model.groups:
            for vacc in ('jnj', 'mrna'):
                if t >= 15:
                    unvacc_n = params['groupN'][age] - model.vacc_rate_df.groupby(['t', 'group']).sum().loc[(t - 15, age), 'cumu']
                    vacc_per_unvacc[t][(age, vacc)] = model.vacc_rate_df.loc[(t - 14, age, vacc), 'rate'] / unvacc_n
                else:
                    vacc_per_unvacc[t][(age, vacc)] = 0

    gparams_lookup = {}
    for t in model.trange:
        gparams_lookup[t] = {}
        for seir in attr['seir']:
            for age in attr['age']:
                for vacc in attr['vacc']:
                    p = {k.replace('gamma', 'gamm').replace('beta', 'bet').replace('groupN', 'age_group_pop').replace('N', 'total_pop'): v for k, v in model.gparams_lookup[t][age].items()}
                    p['ef'] = model.ef_by_t[t]
                    p['mrna_per_unvacc'] = vacc_per_unvacc[t][(age, 'mrna')]
                    p['jnj_per_unvacc'] = vacc_per_unvacc[t][(age, 'jnj')]
                    gparams_lookup[t][(seir, age, vacc)] = p

            # {(seir, age): {'ef': model.ef_by_t[t], **{k.replace('gamma', 'gamm').replace('beta', 'bet').replace('groupN', 'age_group_pop').replace('N', 'total_pop'): v for k, v in params.items()}}
            #                 for age, params in model.gparams_lookup[t].items() for seir in attr['seir']}
    # gparams_lookup = {t: {(seir, age): {k.replace('gamma', 'gamm').replace('beta', 'bet').replace('N', 'total_pop'): v for k, v in params.items()} for age, params in model.gparams_lookup[t].items() for seir in attr['seir']} for t in model.trange}

    ode_builder = ODEBuilder(range(600), attr, params=gparams_lookup)
    for age in ode_builder.attributes['age']:
        for seir in attr['seir']:
            ode_builder.add_flow((seir, age, 'unvacc'), (seir, age, 'mrna'), 'mrna_per_unvacc')
            ode_builder.add_flow((seir, age, 'unvacc'), (seir, age, 'jnj'), 'jnj_per_unvacc')
        for vacc in ['unvacc', 'mrna', 'jnj']:
            ode_builder.add_flow(('S', age, vacc), ('E', age, vacc), 'bet * (1 - ef) * rel_inf_prob * lamb / total_pop', scale_by_cmpts=[('I', a, vacc) for a in ode_builder.attributes['age']])
            ode_builder.add_flow(('S', age, vacc), ('E', age, vacc), 'bet  * (1 - ef) * rel_inf_prob / total_pop', scale_by_cmpts=[('A', a, vacc) for a in ode_builder.attributes['age']])
            ode_builder.add_flow(('E', age, vacc), ('I', age, vacc), '1 / alpha * pS')
            ode_builder.add_flow(('E', age, vacc), ('A', age, vacc), '1 / alpha * (1 - pS)')
            ode_builder.add_flow(('I', age, vacc), ('Ih', age, vacc), 'gamm * hosp')
            ode_builder.add_flow(('I', age, vacc), ('D', age, vacc), 'gamm * dnh')
            ode_builder.add_flow(('I', age, vacc), ('R', age, vacc), 'gamm * (1 - hosp - dnh) * immune_rate_I')
            ode_builder.add_flow(('I', age, vacc), ('S', age, vacc), 'gamm * (1 - hosp - dnh) * (1 - immune_rate_I)')
            ode_builder.add_flow(('A', age, vacc), ('RA', age, vacc), 'gamm * immune_rate_A')
            ode_builder.add_flow(('A', age, vacc), ('S', age, vacc), 'gamm * (1 - immune_rate_A)')
            ode_builder.add_flow(('Ih', age, vacc), ('D', age, vacc), '1 / hlos * dh')
            ode_builder.add_flow(('Ih', age, vacc), ('R', age, vacc), '1 / hlos * (1 - dh) * immune_rate_I')
            ode_builder.add_flow(('Ih', age, vacc), ('S', age, vacc), '1 / hlos * (1 - dh) * (1 - immune_rate_I)')
            ode_builder.add_flow(('R', age, vacc), ('S', age, vacc), '1 / dimmuneI')
            ode_builder.add_flow(('RA', age, vacc), ('S', age, vacc), '1 / dimmuneA')
            # ode_builder.add_flow(('S', age), ('V', age), 'vacc_immun_gain / age_group_pop')
            # ode_builder.add_flow(('R', age), ('V', age), 'vacc_immun_gain / age_group_pop')
            # ode_builder.add_flow(('RA', age), ('V', age), 'vacc_immun_gain / age_group_pop')

    return ode_builder

if __name__ == '__main__':
    engine = db_engine()
    model = CovidModel([0, 600], engine=engine)
    model.set_ef_from_db(5324)
    params = json.load(open('input/params.json'))
    # params['beta'] = 0.6
    model.prep(params=params)
    vacc_per_unvacc = {}

    seir = seir_model(model)
    y0_dict = {('S', age): n for age, n in model.gparams['groupN'].items()}
    y0_dict[('I', '40-64')] = 2
    y0_dict[('S', '40-64')] -= 2
    seir.solve_ode(y0_dict)

    # print(seir.solution_ydf)
    # print(seir.solution_sum('seir'))

    # model.solve_seir(seir.ode)
    # model.solution_y = [np.reshape(y, (len(model.vars), len(model.groups))).transpose().flatten() for y in model.solution_y]
    # model.solution_ydf_full = pd.concat([model.y_to_df(model.solution_y[t]) for t in model.trange], keys=model.trange, names=['t', 'group'])

    # print(model.solution_ydf_summed)

    actual_hosps(engine)
    plt.plot(model.daterange[seir.trange], seir.solution_sum('seir')['Ih'], color='blue')
    # total_hosps(model)
    plt.show()
