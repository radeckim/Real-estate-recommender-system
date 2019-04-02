import matplotlib as plt
import pandas as pd
import numpy as np
from scipy import polyval, polyfit

class Results:
           
    # Calculate and display the recall and precision of the system
    def evaluate(numOfIter, df):
         
        precision = pd.DataFrame([df.iloc[:, i].div(df.iloc[:, i + 3]) for i in range(1, 4)])
        recall = pd.DataFrame([df.iloc[:, j].div(df.iloc[:, 0]) for j in range(1, 4)])
        
        averagePrec = precision.mean(axis = 1)
        averageRec = recall.mean(axis = 1)
              
        plt.plot(range(1, numOfIter + 1), averagePrec, linewidth= 3)
        plt.xlabel("Number of iterations of evolving $\epsilon$FRB algorithm. ", fontsize=12)
        plt.xticks(np.arange(0, numOfIter, 1.0))
        #plt.ylim(1)
        plt.xlim(0, 4)
        plt.ylabel("Precision", fontsize=12) 
        plt.show()
        
        plt.plot(range(1, numOfIter + 1), averageRec, linewidth= 3) 
        plt.xlabel("Number of iterations of evolving $\epsilon$FRB algorithm. ", fontsize=12) 
        plt.xticks(np.arange(0, numOfIter, 1.0))
        plt.xlim(0, 4)
        plt.ylabel("Recall", fontsize=12) 
        plt.show()
        
    # Display average change of weights         
    def plotChangeOfWeights(weights):
    
        plt.plot(weights, linewidth= 3)
        plt.xlabel("Number of iterations of evolving $\epsilon$FRB algorithm. ", fontsize=12)
        plt.xticks(np.arange(0, 3, 1.0))
        plt.ylabel("Price attribute weight", fontsize=12) 
        plt.show() 
    
    # Display the time of computation and its logarithmic approximation    
    def timeEMF():
    
        ticks = np.array([1000, 5000, 10000, 50000, 100000, 150000])
        times = np.array([0.26, 1.23, 2.84, 13.01, 14.8, 16])
        
        ticksCont = np.array([i for i in range (0, 150000)])
        [a, b] = np.polyfit(np.log(ticks), times, 1)
        timesLinear = 1.6*(0.9*a*np.log(ticksCont) + b)
        #print(timesLinear)
    
        plt.plot(ticks, times, 'ro')
        plt.plot(ticksCont, timesLinear)
        plt.ylabel("Time in secons [s]")
        plt.xlabel("Number of data samples")
        plt.ylim(0, 20)
        plt.show()
    
    # Display scatter plot of precision and recall    
    def scatterPrecRec(nr, nrs, ns):
                  
        precision = nrs/ns
        recall = nrs/nr
        
        print(precision.mean())
        
        plt.scatter(precision, recall)
        plt.xlim(0,1)
        plt.xlabel("Precision")
        plt.axvline(precision.mean())
        
        plt.ylim(0, 1)
        plt.ylabel("Recall")
        plt.axhline(recall.mean())
        
        plt.plot(precision.mean(), recall.mean(), '--bo')
        plt.annotate('The average Precision/Recall', xy=(precision.mean(), recall.mean()), xytext=(0.15, 0.5),
                arrowprops=dict(facecolor='red', shrink=1),
                )
        plt.show()
    



        
