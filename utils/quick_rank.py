import pandas as pd
import numpy as np

from collections import defaultdict
textrank = {"inspec":[21.58,27.53,27.62], "sm2017":[16.43,25.83,30.50],"sm2010":[7.42,11.27,13.47], "duc":[11.02,17.45,18.84],"krapivin":[6.04,9.43,9.95],"nus":[1.80,3.02,3.53]}
singlerank = {"inspec":[14.88,21.50,24.13], "sm2017":[18.23,27.73,31.73],"sm2010":[8.69,12.94,14.40], "duc":[19.14,23.86,23.43],"krapivin":[8.12,10.53,10.42],"nus":[2.98,4.51,4.92]}
topicrank = {"inspec":[12.20,17.24,19.33], "sm2017":[17.10,22.62,24.87],"sm2010":[9.93,12.52,12.26], "duc":[19.97,21.73,20.97],"krapivin":[8.94,9.01,8.30],"nus":[4.54,7.93,9.37]}
multipartite = {"inspec":[13.41,18.18,20.52],"sm2017":[17.39,23.73,26.87],"sm2010":[10.13,12.91,13.24],	"duc":[21.70,24.10,23.62], "krapivin":[9.29,9.35,9.16], "nus":[6.17,8.57,10.82]}
YAKE = {"inspec":[8.02,11.47,13.65], "sm2017":[11.84,18.14,20.55],"sm2010":[6.82,11.01,12.55], "duc":[11.99,14.18,14.28],"krapivin":[8.09,9.35,9.12],"nus":[7.85,11.05,13.09]}
EmbedRank = {"inspec":[14.51,21.02,23.79], "sm2017":[20.21,29.59,33.94],"sm2010":[9.63,13.90,14.79], "duc":[21.75,25.09,24.68],"krapivin":[8.44,10.47,10.17],"nus":[2.13,2.94,3.56]}
SIFRank = {"inspec":[29.38,39.12,39.82], "sm2017":[22.38,32.60,37.25],"sm2010":[11.16,16.03,18.42], "duc":[24.30,27.60,27.96],"krapivin":[1.62,2.52,3.00],"nus":[3.01,5.34,5.86]}
PD = {"inspec":[28.92,38.55,39.77], "sm2017":[20.03,31.01,36.72],"sm2010":[10.46,16.35,19.35], "duc":[8.12,11.62,13.58],"krapivin":[4.05,6.60,7.84],"nus":[3.75,6.34,8.11]}
# once = {"inspec":[27.93,37.38,39.11], "sm2017":[20.56,30.95,36.07],"sm2010":[10.16,15.40,17.69], "duc":[9.11,13.49,16.47],"krapivin":[4.61,7.21,8.15],"nus":[3.92,6.52,8.85]}
# subset = {"inspec":[29.25,36.55,38.08], "sm2017":[21.50,31.30,36.67],"sm2010":[10.26,15.88,17.83], "duc":[12.05,16.73,19.19],"krapivin":[8.50,9.99,10.48],"nus":[9.61,13.43,14.65]}
all = {"inspec":[26.17,33.81,36.17], "sm2017":[22.81,32.51,37.18],"sm2010":[12.95,17.07,20.09], "duc":[13.05,17.31,19.13],"krapivin":[11.78,12.93,12.58],"nus":[15.24,18.33,17.95]}
ab = {"inspec":[28.06,35.80,37.43], "sm2017":[21.63,32.23,37.52],"sm2010":[12.95,17.95,20.69], "duc":[22.51,26.97,26.28],"krapivin":[12.91,14.36,13.58],"nus":[14.11,17.72,17.95]}
re = {"inspec":[27.85,34.36,36.40], "sm2017":[20.37,31.21,36.63],"sm2010":[13.05,18.27,20.35], "duc":[23.31,26.65,26.42],"krapivin":[12.35,14.31,13.31],"nus":[14.39,18.46,19.41]}
methods = [textrank, singlerank, topicrank, multipartite, YAKE, EmbedRank, SIFRank, PD, all, ab, re]
name = ["textrank", "singlerank", "topicrank", "multipartite", "YAKE", "EmbedRank", "SIFRank", "PD", "all", "ab", "re"]
#cal avg
data_avg = dict()
for i, m in enumerate(methods):
    l5, l10, l15 = [],[],[]
    for ds, f1 in m.items():
        l5.append(f1[0])
        l10.append(f1[1])
        l15.append(f1[2])
    avg5 = round(sum(l5)/len(l5),2)
    avg10 = round(sum(l10)/len(l10),2)
    avg15 = round(sum(l15)/len(l15),2)
    data_avg[i] = [avg5, avg10, avg15]

#cal rank
ds_name = textrank.keys()
np.set_printoptions(precision=2)
m_array = []
for i, m in enumerate(methods):
    array = []
    for ds, f1 in m.items():
        array.append(f1)
    m_array.append(array)
m_array = np.array(m_array)
total_rank = []
for i, n in enumerate([5,10,15]):
    rank = np.zeros((11,6))
    df = pd.DataFrame(m_array[:, :, i], columns=ds_name, index=name)
    for n, ds in enumerate(ds_name):
        rank[:,n]=list(df[ds].rank(method="first",  ascending=False).values)
    avg = np.nanmean(rank, axis=1)
    print ("avg: ", avg)
    std = np.std(rank, axis=1)
    print ("std: ", std)