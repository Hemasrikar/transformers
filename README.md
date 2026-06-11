# Transformers for Asset Pricing


> [!Note]
> There are four branches including the main branch in this repository. The `main` branch contains all main impplementation codebase. The branche `nonlinear/time2vec` has a architecture involving Time2Vec encoding and periodic lag data. `dualapproach` has different a dual appraoch architecture with MLP layers before the embedding layer, along with Time2Vec encoding. 
> 
> The `thesis\resources` contain diagrams for the dissertation.

## Data Processing

This branch contains the code base for all the data processing needed from this project.

All the data is obtained from WRDS (Wharton Research Data Services).

The [Company Data csv](csv_data\company_data_info.csv) file all the linking data from Compustat and CRSP. The datasets are seperated in three ways: Emerging Markets, USA market, and all the markets in the world. Primarily, we will work with the EM dataset.

---

### Train, Test, Validation Split

All the data split will be done prior to the data processing, and it is done based on the time period. 
> Train Period: 1995 - 2015
> Validation Period: 2016 - 2020
> Test Period: 2021 - 2025

The features are removed, if their respective column has more than 30% missing data. This is processed on train dataset and the kept features are used to filter the data in the validation and test datasets. After filtering the features, the train dataset will be processed. 

> [!NOTE]
> The `csv_to_parquet` notebook is just to convert the csv files downloaded from the wrds to parquet, since some of the query form does not have parquet file format as an option. Parquet format is foavoured becuase it is smaller in size when compared to csv and faster to work with.
