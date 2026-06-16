# Nonlinear Transformers for Asset Pricing

This branch contains original Nonlinear Transformer model, that was introduced by Kelly.

## Architecture
The `src` contains two different implementation. `nonlinear_transformer_rolling.py` contains codebase that train the model on rolling window methodology, which is same as implemented by Kelly. The file `nonlinear_transformer.py` contains model that train based on the following dataset split.

Both the codebases contain data processing code included. So, they can be run standalone on the raw datasets.

### Train, Test, Validation Split

All the data split will be done prior to the data processing, and it is done based on the time period. 
> Train Period: 1995 - 2015
> Validation Period: 2016 - 2020
> Test Period: 2021 - 2025

The features are removed, if their respective column has more than 30% missing data. This is processed on train dataset and the kept features are used to filter the data in the validation and test datasets. After filtering the features, the train dataset will be processed. 

> [!Note]
> There are five branches including the main branch in this repository. 
> - The `main` branch contains all the codebase of the final architecture after experimentation. 
> - `dualapproach` branch has different a dual appraoch architecture with MLP layers before the embedding layer, and other additional statistical embedding.
> -  `nonlinear/original` has the original kelly proposed architecture with different embedding varisnts.
> - The branch `nonlinear/time2vec` has a architecture involving Time2Vec encoding and periodic lag data. 
> - The `thesis\resources` contain diagrams for the dissertation.

