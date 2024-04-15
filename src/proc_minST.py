import pandas as pd
import argparse
from minST import *



if __name__=="__main__":
  
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--infolder",help="Path of the input file.",type=str)
    parser.add_argument("-f","--filterthresh",help="The filtering threshold for cell similarity network. Default: None",type=float)

    args = parser.parse_args()
    infolder = args.infolder
    filt = args.filterthresh
    
    if infolder[-1]!='/':
        infolder = infolder + r'/'
    
    infile = infolder + r'cellbygene_lognorm.csv'
    outfile = infolder + r"csim.csv"
    print(outfile)

    data=pd.read_csv(infile,index_col=0)
    data_dist=minST(data=data,dist_type='seuclidean',reduction='pca', filt = filt)
    data_dist=pd.DataFrame(data_dist)
    data_dist.to_csv(outfile)
    
    
    
    
