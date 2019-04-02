import pandas as pd
import scipy
import numpy as np

# The class responsible for recommendations.
# It implements an empirical fuzzy rule based system for real estate data.
# The extension to the solution is also implemented, i.e. evolving fuzzy rule based system

class Recommender:
    
    #the sequence used for transforming data to continuous
    sequence = np.arange(0, 1, 0.001)
    threshold = 0.85 # EMF threshold
    
 
    # Create the recommender - load needed data    
    def __init__(self, dataNorm, categoricalData, finalData):
        
        self.finalData = finalData
        self.dataNorm = dataNorm
        self.categoricalData = categoricalData
        self.averRelevants = []
    
    # Select data    
    def prototype_select(self):
        
       self.prototypes_idxs = list(map(int, input('Enter numbers: ').split()))

       self.prototypes = self.dataNorm.iloc[[idx for idx in self.prototypes_idxs]]

       #calculate average prototype
       self.averageProt = pd.DataFrame(
               data = self.prototypes.sum()/len(self.prototypes)) 
       
    # Form Voronoi tesselations
    def form_voronoi(self):
        
        dist = scipy.spatial.distance.cdist(
                self.dataNorm.iloc[:,0:], 
                self.prototypes.iloc[:,0:], 
                metric='euclidean')
        
        euclidDist = pd.DataFrame(dist)
        euclidDist.columns = list(self.prototypes.index)

        #find the nearest prototype for each data sample - forming voronoi tesselations
        min_values = euclidDist.idxmin(axis=1)
        self.sampleCenters = pd.DataFrame(min_values)
        
    # Find the standard deviation of each tesselation
    def std_of_voronois(self):
        
        grouped = self.sampleCenters.groupby(0)

        self.std = pd.DataFrame(columns=['sumOfStdSquared'])
        stdDevGroup = pd.DataFrame()

        for name, group in grouped:
        
            #find indexes of data samples in a data cloud
            groupDF = pd.DataFrame(group).index
        
        #find standard deviation for each attribute 
        stdDevGroup[name] = pd.DataFrame(self.dataNorm.iloc[groupDF]).std().pow(2)
        self.std.loc[name] = stdDevGroup[name].sum() 

        self.stdDevGroup = np.transpose(stdDevGroup)
        
    # Calculate empirical membership function for each occurance in the dataset
    def calculate_EMF(self, featuresIdxs, weight): 
        
        self.form_voronoi()
        self.std_of_voronois()
    
        EMF = pd.DataFrame(columns=[idx for idx in self.prototypes_idxs])
        
        chosenNorm = self.dataNorm.loc[:, featuresIdxs]    
        chosenProto = self.prototypes.loc[:, featuresIdxs]
        
        for index_value, row_data in chosenProto.iterrows():  
            
            #substract each sample by a prototype vector
            dist = chosenNorm.apply(lambda row: row - row_data, axis = 1).pow(2)
            
            dist *= weight
                                        
            #get normed vector space
            normDistVec = dist.apply(lambda row: row.sum(), axis = 1)
            
            #divide normalized vectors by a difference of the scalar product of a cloud and prototype vector
            #which gives a squared sum of standard deviations within a cloud
    
            EMFstd = self.std.loc[index_value]['sumOfStdSquared']           ##standard deviation of the current data cloud
            quotient = normDistVec.apply(lambda row: row / EMFstd)
    
            #finally calculate eMF for each sample for a particular prototype
            EMF.loc[:,index_value] = quotient.apply(lambda row: 1 / (1 + row))
            
        return EMF
    
    # Display recommendations basing on the threshold and calculated EMF
    def display_recommendations(self, EMF):
    
        #find indexes defined by the threshold
        recommendationsIdxs = EMF[(EMF.loc[:,[idx for idx in self.prototypes_idxs]] > Recommender.threshold).any(axis=1)].index
        recommended = self.categoricalData.iloc[recommendationsIdxs]
    
        catDict = self.categoricalData.iloc[self.prototypes_idxs].to_dict(orient = "list")
        finalRecomendIdxs = recommended[recommended[list(catDict)].isin(catDict).all(axis=1)].index
    
        finalRecomendIdxs = list(set(recommendationsIdxs) - set(self.prototypes_idxs))
        
        Visualization.display(pd.DataFrame(self.finalData.iloc[recommendationsIdxs]))
        
        return len(finalRecomendIdxs) + 1
    
    # The function used for basic recommendation process - it does not involve evolving eFRB
    def recommendation_process(self):
        
        self.prototype_select()
        EMF = self.calculate_EMF([0, 1, 2, 3, 4, 5], 1)
        idxs = self.display_recommendations(EMF, 0.85)
        print(idxs)
        
    # Find the vector representing the average relevant recommendation         
    def find_average_relevant(self):
    
        print('Enter ID\'s of relevant recommendations: ')    
        relevant_idxs = list(map(int, input().split()))
    
        relevant = self.dataNorm.iloc[[idx for idx in relevant_idxs]]
    
        #calculate average relevant recommendation
        averageRelevant = relevant.sum()/len(relevant)
        
        return averageRelevant
    
    # Calculate empirical membership function for each feature - in continous form
    def calcEMFPF(self, prototypes):
    
        (lenProt, wProt) = prototypes.shape
        membershipPF = np.zeros((wProt, len(Recommender.sequence), lenProt + 1))
    
        for i in range(wProt):
        
            for j in range(lenProt + 1):
                
                membershipPF[i, :, j] = (self.stdDevGroup.iloc[0, i] /
                        (np.power((Recommender.sequence - prototypes.iloc[0, i]), 2)  
                        + self.stdDevGroup.iloc[0, i]))
                
                
        return membershipPF, lenProt
        
    # Make the system evolve basing on user preferences
    def evolve_EMF(self, numOfIter):
        
                               
        wAv = self.averageProt.shape[0]
        averageR = self.find_average_relevant()
        self.averRelevants.append(averageR)
        (EMF, _) = self.calcEMFPF(self.averageProt.T)
        intersectionPoints = []
                
        for i in range(wAv):     
                        
            proper = (Recommender.sequence * (max(self.dataNorm.iloc[:, i]) - min(self.dataNorm.iloc[:, i])) + min(self.dataNorm.iloc[:, i]))
            idx = np.argwhere(np.diff(np.sign(EMF[i, :, 0] - Recommender.threshold))).flatten()
            intersectionPoints.append(proper[idx][0])
      
        dfIntersect = pd.DataFrame(intersectionPoints)
        dfRelevants = pd.DataFrame(self.averRelevants)
             
        distance = abs(dfIntersect - self.averageProt)
        if numOfIter > 1:
            average = dfRelevants.iloc[0:(numOfIter - 1), :].apply(lambda row: abs(row - list(self.averageProt.T)), axis = 1)
        else: average = dfRelevants.iloc[0, :]
                
        weight = list(10.*(distance.T / (((numOfIter - 1) * average) + abs(self.averRelevants[numOfIter - 1] - list(self.averageProt.T)))).iloc[0, :])
            
        for index, i in enumerate(weight):  
            
            if(i < 1): weight[index] = 1
        
        evolvedEMF = self.calculate_EMF([0, 1, 2, 3, 4, 5, 6], weight) 
        
        self.display_recommendations(evolvedEMF)        
        
    
    # Conduct the evaluation process - evolving empirical fuzzy rule based system       
    def evaluate(self, iterations):
        
        self.prototype_select()        
        exampleEMF = self.calculate_EMF([0, 1, 2, 3, 4, 5, 6], 1)    
        self.display_recommendations(exampleEMF)
        
        for i in range(1, iterations + 1):
        
            self.evolve_EMF(i)
 
        
    
    