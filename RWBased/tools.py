#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[64]:


def readCSV(cellLine_file, IC50_file, AUC_file, target_file):
#     cellLineInfoDf = pd.read_csv('cell_line_info.csv')
    cellLineInfoDf = pd.read_csv(cellLine_file)
#     ICDf = pd.read_csv('IC50.csv')
    ICDf = pd.read_csv(IC50_file)
#     AUCDf = pd.read_csv('AUC.csv')
    AUCDf = pd.read_csv(AUC_file)
#     drugPairs = pd.read_csv('drug_pairs.csv', header=None)
    drugPairs = pd.read_csv(target_file, header=None)
    
    
    drugPairs.drop(columns = [0], inplace =True) # 去掉原本的编号列
    tarList = list()
    
    for ii, row in drugPairs.iterrows():
        tarList.append(row.values)
    # print(tarList)
    
    return tarList, cellLineInfoDf, AUCDf, ICDf
#     print(tarList[0])
#     print(cellLineInfoDF.info())
#     print()
#     print(ICInfoDF.info())
#     print()
#     print(AUCDF.info())


# In[3]:


def getCorrespondenceInfo(curTar, cellLineInfoDf, AUCDf, ICDf):
    import pandas as pd
    import numpy as np
    # 从IC50.CSV 和 AUC50.csv取出当前对比组的数据
    drugSensDf = pd.DataFrame()
    drugSensDf = pd.concat([drugSensDf, ICDf['cosmic_id']], axis=1)
#     drugSensDf = drugSensDf[ (drugSensDf['cosmic_id']!=1298167) & (drugSensDf['cosmic_id']!=930299) ]
#     print(drugSensDf)
    for ii, xx in enumerate(curTar):
#         print(xx)
        curTarIC_list = ICDf[xx].tolist()        
        curTarAUC_list = AUCDf[xx].tolist()
        
        curDict = {}
        curDict['D'+str(ii+1)+'IC50'] = curTarIC_list
        curDict['D'+str(ii+1)+'AUC'] = curTarAUC_list

        curTarDf = pd.DataFrame(curDict)
        
        drugSensDf = pd.concat([drugSensDf, curTarDf], axis=1)

    
#     print(drugSensDf.info())
    
    drugSensDf.dropna(how='any', inplace = True)
    drugSensDf.sort_values('cosmic_id', inplace= True)
    drugSensDf.index = range(len(drugSensDf))
    
#     print(drugSensDf.info())
    cellLineInfoDf = cellLineInfoDf[cellLineInfoDf['cosmic_id'].isin(drugSensDf['cosmic_id'])].dropna(how='any')
    cellLineInfoDf.sort_values('cosmic_id', inplace = True)
    cellLineInfoDf.index = range(len(cellLineInfoDf))
    
    # 拿到对应的tissue后，drugSensDf去掉cell line列
    cellLineList = cellLineInfoDf['sample_name'].values
    tissueList = cellLineInfoDf['Tissue'].values
#     print(type(cellLine))
    drugSensWithoutID = drugSensDf.drop(columns=['cosmic_id'])

    
#     print(cellLineList.shape)
#     print(tissueList.shape)
    
    sensValList = drugSensWithoutID.values
#     print(sensValList.shape)
    
#     import pickle
#     with 
    
    
    return sensValList, cellLineList, tissueList


def getCenters(labelList, dataList):
    centerDist = {}
    for kk in np.unique(labelList):
        sum1 = 0
        count = 0
        for ii, xx in enumerate(labelList):
            if xx == kk:
                sum1 += dataList[ii]
                count += 1
        centerDist[kk] = sum1/count
#     print('Num CLusters:', len(centerDist))
    return centerDist


def computePBM(data, labelList):

	K = len(np.unique(labelList))
	# print('K=',K)
	centerK1 = np.sum(data, axis=0)/len(data)
	# E1 = np.sum(np.sqrt(np.sum(np.subtract(data,centerK1)**2, axis=1)))
	E1 = 0
	for sample in data:
		E1 += np.linalg.norm(np.subtract(sample, centerK1))

	centers = getCenters(labelList, data)

	EK = 0
	for xx in np.unique(labelList):
		ci = centers[xx]
		distance = 0
		for jj, ee in enumerate(labelList):
			if ee == xx:
				sample = data[jj]
				distance += np.linalg.norm(np.subtract(sample, ci))
		EK += distance


	DK = 0
	centerList = list(centers.values())
	for ii in range(len(centerList)):
		for jj in range(ii+1, len(centerList)):
			curDiff = np.linalg.norm(np.subtract(centerList[ii], centerList[jj]))
			if curDiff >= DK:
				DK = curDiff

	PBM = ((1/K)*(E1/EK)*DK)**2
	return PBM 



def computeRMSSTD(data, labelList):

#     P = data.shape[1]
#     print('P=', P)
    fenzi_sum = 0
    fenmu_sum = 0

    centers = getCenters(labelList, data)

    for xx in np.unique(labelList):
        fenzi = 0
        fenmu = 0
        count = 0
        ci = centers[xx]

        for jj, ee in enumerate(labelList):
            if ee == xx:
                sample = data[jj]
                fenzi += np.linalg.norm(np.subtract(sample, ci))**2
                count += 1
        fenmu = 4*(count-1)
        
        fenzi_sum += fenzi
        fenmu_sum += fenmu
        
    overrallSum = fenzi_sum/fenmu_sum
    
    overrallSum = np.sqrt(overrallSum)
#     print('RMSSTD=', overrallSum)
#     print('-'*5)
    return overrallSum


def computeSScore(data, label):
	from sklearn.metrics import silhouette_score
	sscore = silhouette_score(data, label)
	return sscore




def saveInfo( drugName, sensValList, labelArray, sscore, rmsstd):
    import os
    import pickle

    # print('DrugName:',drugName)
    fileName = '_'.join(drugName)
    print('File Name:', fileName)
    
    # Save Data and LabelArray
    import os
    saveDataLabelFolder = 'DataAndLabel'
    
    if os.path.exists(saveDataLabelFolder):
        pass
    else:
        os.mkdir(saveDataLabelFolder)
    
    f0 = open(saveDataLabelFolder+'/'+fileName, 'wb')
#     pickle.dump(dataList, f0)
    pickle.dump(sensValList, f0)
    pickle.dump(labelArray, f0)
#     pickle.dump(cellLineList, f0)
    f0.close()
    
    
    # Save sscore
    saveSScoreFolder = 'SScore'
    
    if os.path.exists(saveSScoreFolder):
        pass
    else:
        os.mkdir(saveSScoreFolder)
    
    f0 = open(saveSScoreFolder+'/'+ fileName +'_SScore', 'wb')
    pickle.dump(sscore, f0)
    f0.close()
    
    
    # save RMSSTD
    saveRMSSTDFolder = 'RMSSTD'
    
    if os.path.exists(saveRMSSTDFolder):
        pass
    else:
        os.mkdir(saveRMSSTDFolder)
    
    f0 = open(saveRMSSTDFolder+'/'+ fileName +'_RMSSTD', 'wb')
    pickle.dump(rmsstd, f0)
    f0.close()
    
    
    
    return
