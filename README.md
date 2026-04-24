This is the general pipline of the code, it is explained in the suggested order to run the code 

1) Data Extraction.py
This code downloads data frm WRDS (you need to connect to the server with account and password). For computational load reasons, it is done one country at a time.
To do so, under #CONFIG there is a part that says "MARKET = __" fill in the blank with the country to be dowloaded, you will have to do this for every country. Country name the only thing to be changed, the time periods defined elsewhere

2) US_US.py and US_US_NN.py have the models train on US data and store results elsewhere.
To run this, some paths and folder contructions have to be altered. I will explain how my enviroment is set up, tailor it to your situation.
my VScode has its directory set in a 'group' folder. The path from this directory to the US data file is: /group/raw_data/cleenerst
upon executing, three folders will be created: model_parameters, forecasts, and summary

3) US_international.py and US_international_NN.py have models apply US-trained models to each country
   to run this code, again it assumes you have stored all data in csv form in the following path of folders: /group/raw_data/cleenerst ... 
   within cleenerst, there should be csv data for each country with the following naming convention {country_name}_clean.csv

4) into_into.py and into_into_NN.py have the models do market specific training
   if you have run previous code, then the raw data for each country should already by in the right place (cleenerst) folder and the code should run smoothly

5) pooled.py and pooled_NN.py
this creates a pooled sample by appending the csv data in the cleenerst folder. assuming everything there and the previous code has run succesfully, this code should work.

6) ssd_figure.py
previous codes output and store model information, this code just takes the SSD calculation info and creates the relative importance figure (non-essential)

Each code file has a more detailed explanation added as comments in the first few lines if you want more detail. 
for example, the model estimation codes are made such that not all models have to be run at once in order to work. this is explained in the comments i have left
