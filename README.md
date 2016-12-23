# PokemonGo_Predict_Them_All
- ProjectID: 201612-75

# Team members
- Tianhao Lu, tl2710
- Qing Ma, qm2124
- Shengzhong Yin, sy2615

# Introduction
- We used PokemonGo dataset featuring the occurrences of different pokemon and the rich set of features of the physical location, time, etc. We did data cleaning to make the dataset easy to manipulate with, and used SystemG to make some exploratory data analysis to get insight about the data. Then we performed various machine learning algorithms regarding the binary labels of the dataset and compared the results, including PySpark’s MlLib library. Finally we filter the dataset to that of NYC and compared some of the rare pokemons’ frequency in the city with the relevant complaints received from 311. This comparison indicates further work that may shed light on feature engineering and predictive analytics.

# Dataset
- PokemonGo_Predict_Them_All.csv, 402 MB
- 311_Service_Request.csv, 1.27 GB
- evol.csv, assisting dataset
- type_id.csv, assisting dataset

# Run code
- We run the .py file by jupyter notebook. Code can also be executed by python/anaconda compiler.
- We run .R or .rmd code by r-studio.
- To run the classifier models, first execute clean_poke.py to clean the data, and then run the classfier. 
- Run maps.py to draw the pokemon distribution maps
- Run graph.R to build e systemG cluster
