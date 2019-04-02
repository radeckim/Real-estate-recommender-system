import pandas as pd
import numpy as np
from sklearn import preprocessing 
from scipy import stats

# The class handling data loading and preprocessing.
# It involves all of the aspects regarding data preparation for the real estate
# data recommender systems, i.e.:
# - loading data
# - generating additional variables
# - data normalisation
# - removing 'impossible' real estates
# - encoding categorical variables
#
# @author - Michal Radecki

class DataProcessing:
    
    path = "Data.csv"                                                          #path to dataset
    numerical = ["PRICE", "GBA", "LANDAREA", "AYB"]                            #num variables
    categorical = ["AC", "BATHRM", "BEDRM", "ROOMS"]                           #cat variables  
    generated = ["Trans_Score", "Walk_Score", "School_Score"]                  #generated numerical variables 
    colnames = numerical + categorical                                         #variables names 
    missing = {'PRICE':[0,''],'AYB':[0],'AC':[0]}                              #possibly empty values 
    
    scaler = preprocessing.MinMaxScaler()
    
    # Create an instance of the class - load data
    def __init__(self):        
      
        self.dataFrame = pd.read_csv(DataProcessing.path,                      #load data 
                                     na_values = DataProcessing.missing,       #drop 0 values
                                     usecols = DataProcessing.colnames)        #add col names
     
    # Generate additional numerical variables.    
    def generate_var(self):
        
        #extend variables range by generated values
        DataProcessing.numerical.extend(DataProcessing.generated)
        
        #generate and add new variables        
        for var in DataProcessing.generated:
            
            self.dataFrame[var] = np.random.randint(1, 
                                                    100, 
                                                    size=len(self.dataFrame))
                
    # Drop rows with missing values. Also select a needed number of rows.           
    def drop_missing(self):
        
        self.dropped = self.dataFrame.dropna()
        self.dropped = self.dropped.iloc[0:500]                             #fixed number of rows 
    
    # Preprocess numerical variables
    def handle_numerical(self):
        
        self.numericalData = self.dropped[DataProcessing.numerical] 
        
        outliers = self.numericalData[(
                np.abs(stats.zscore(self.numericalData)) < 2).all(axis=1)]

        #drop possibly non-existring properties - PRICE BELOW 10K
        self.processedNum = self.numericalData.drop(
                outliers.loc[outliers['PRICE'] < 10000].index)

        #standardize data - scaling is done for future euclidean distance measure 
        dataNorm = DataProcessing.scaler.fit_transform(self.processedNum)
        self.dataNorm = pd.DataFrame(dataNorm)
     
    # Preprocess categorical variables
    def handle_categorical(self):
        
        categoricalData = self.dropped.loc[self.processedNum.index,
                                           DataProcessing.categorical]
        #air condition flag - 0 or 1
        labels = categoricalData["AC"].astype('category').cat.categories.tolist()
        replace_map_comp = {"AC" : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}
        categoricalData.replace(replace_map_comp, inplace=True)
        
        #room style category
        bins = np.arange(0, 16, 2)
        categoricalData['ROOMS_CAT'] = np.digitize(categoricalData.ROOMS, bins)
        
        #concat two preprocessed frames
        finalData = pd.concat([self.processedNum, categoricalData], axis = 1)
        
        #drop houses which number of rooms is less then number of bedrooms or bathrooms
        roomsAnomaly = list(finalData[finalData["ROOMS"] < finalData["BEDRM"]].index) 
        roomsAnomaly = roomsAnomaly + list(finalData[finalData["ROOMS"] <= finalData["BATHRM"]].index)
        roomsAnomaly = roomsAnomaly + list(finalData[finalData["ROOMS"] >= 29].index)
        finalData.drop(roomsAnomaly)
        
        self.finalData = finalData.reset_index(drop=True)
        self.categoricalData = categoricalData.reset_index(drop=True)
        self.numericalData = self.numericalData.reset_index(drop=True)        
