import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# The class used for data visualisation, which implements the following functionality:
# - histograms of numerical data
# - correlograms of numerical/categorical data
# - display a given data frame

class Visualization:
    
    # Display histograms
    def histograms(data):
   
        for ii in range(0, len(data.columns)):
    
            minimum = min(data.iloc[:, ii])
            
            if (data.columns[ii] == "PRICE" or 
               data.columns[ii] == "LANDAREA" or data.columns[ii] == "GBA"):
    
                maximum = max(data.iloc[:, ii]) / 1.5
    
            else: 
    
                maximum = max(data.iloc[:, ii])
    
            data.iloc[:, ii].plot.hist(grid=True, bins=100, rwidth=0.9, color= "#607c8e", range = (minimum, maximum))
            plt.title(data.columns[ii])
            plt.show()
            
    # Display correlograms                    
    def correlograms(data):
        
        plt.style.use('seaborn-colorblind')
        sns.pairplot(data, kind="scatter", 
                           diag_kind = 'kde', 
                           plot_kws = {'alpha': 0.33, 's': 40, 'edgecolor': 'k'}, 
                           height = 3)
        plt.show()
           
    
    
    # Display a panda dataframe in a readable format.         
    def display(df):
    
        with pd.option_context('display.max_rows', 
                               None, 
                               'display.max_columns', 
                               None):
            print(df)
            
    # Display continuous form of empirical membership function for each feature            
    def EMF_per_feature(membershipPF, lenProt, processedNum):
        
        threshold = 0.85
                
        for ii in range(0, 6):
            
            for z in range(lenProt):
        
                xAxis = (Recommender.sequence * (max(processedNum.iloc[:, ii]) - min(processedNum.iloc[:, ii])) + min(processedNum.iloc[:, ii]))
                
                name = "Prototype"
                if(z == lenProt): name = "Weighted"
                
                plt.plot(xAxis, membershipPF[ii, :, z], linewidth= 3, label= name)        
                plt.xlabel(processedNum.columns[ii], fontsize=12) 
                plt.axhline(threshold, color='r')
                                
                idx = np.argwhere(np.diff(np.sign(membershipPF[ii, :, z] - threshold))).flatten()
                plt.vlines(xAxis[idx[0]], ymax = threshold, ymin = 0, color = 'r', linestyle = "--")
                if(len(idx) > 1): plt.vlines(xAxis[idx[1]], ymax = threshold, ymin = 0, color = 'r', linestyle = "--")
        
                
            plt.ylabel(r'$\epsilon$' + "MF", fontsize=12)
            plt.ylim(ymax = 1.3, ymin = 0)
            plt.xlim(xmin = 0)
                       
            plt.axhline(y = 1, linestyle = "--", linewidth = 0.5)
            ax = plt.gca()
            ax.set_facecolor('xkcd:white')
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black') 
            #xlims = ax.get_xlim()
            #ax.set_xticks(xlims)
            
            plt.grid(color='gray', linestyle='-', linewidth=0.3)
                    
            plt.legend(loc='upper left', frameon=True)
            plt.show()    
            
    # Display a correlation of empirical membership function value and two given attributes            
    def display3DPlot(attribute1, attribute2, data, EMF, len_prot):
    
        for prototype in range(0, len(len_prot)):
            
            y = data.iloc[:, attribute1]
            x = data.iloc[:, attribute2]
                            
            fig = plt.figure(figsize=(7,5))
            ax = fig.gca(projection='3d')
            ax.set_zlabel(r'$\epsilon$' + "MF", fontsize=12, rotation = 90)
            ax.zaxis.set_rotate_label(False)
            ax.set_xlabel(x.name)
            ax.set_ylabel(y.name, rotation = 105)
            ax.scatter(x, y, EMF, c=EMF, cmap='viridis', linewidth=0.5, s = 5);
            ax.view_init(elev=45, azim=75)   
        