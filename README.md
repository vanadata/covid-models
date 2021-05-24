# Colorado COVID Models

This repo contains a set of tools for running and processing the State of Colorado COVID-19 models, designed by the Colorado COVID-19 Modeling Group. Other versions of these models can be found in [another repo](https://github.com/agb85/covid-19). Model design documentation and weekly modeling team reports can be found [here](https://agb85.github.io/covid-19/).

## Statewide COVID-19 Model

- Model parameters can be set in [params.json](covid_model/params.json)
- Run fit to determine transmission control parameters using [run_fit.py](covid_model/run_fit.py) (`run_fit.py --help` for details regarding command line arguments)
- Populate data for relevant scenarios using [run_model_scenarios.py](covid_model/run_model_scenarios.py); scenarios include:
  - scenarios for vaccine uptake
  - scenarios for upcoming shifts in transmission control (magnitude and date of shift)
  
## Regional COVID-19 Model

Code for the regional COVID-19 Model can be found in [another repo](https://github.com/agb85/covid-19). This repo contains some tools to do minimal processing of the outputs of the Regional COVID-19 Model.
