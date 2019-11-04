#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
def MakeFolder(newFolder):
    """
        Method to make a new folder given the path of the desired folder;
        if the folder already exists, then it does nothing

    :param newFolder:
    :return:
    """

    if newFolder[-1] != '/':
        newFolder = newFolder + '/'

    # Make output folders if they don't exist
    if os.path.exists(newFolder):
        pass
    elif ~os.path.exists(newFolder):
        os.mkdir(newFolder)
        os.mkdir(newFolder + 'Figs/')
        os.mkdir(newFolder + 'SavedData/')
        os.mkdir(newFolder + 'Results/')
        os.mkdir(newFolder + 'Exports/')
        os.mkdir(graphsFolder)


    return newFolder


# In[4]:


def IterateOverTargets(ids, curDrugName):
    """

        Method to iterate over each tarets in the target list
        and execute POET for the associated compounds and cell lines

    :param targList:
    :param compList:
    :param Ntargets: # N组药物对比 len(compList) = len(targList)
    :return:
    """

    # SEt up main loop with global variables that will change with every entry in the
    # wrapperFile
    global curTarg              # current Target under consideration 
    global numDrugs             # number of drugs that are target agents for curTarg 
    global drugName             # list of drug names that are agents for curTarg
    global allVarsInclLabels    # list of labels for variables being clustered
    global localResultsFolder   # local folder, based on curTarg, where output is
                                # directed

    curTarg = str(ids)
    numDrugs = len(curDrugName)
    allVarsInclLabels = list()
    drugName = list() 

    # Iterate thru each compound associatd with current target
    # to get drug name and assign string labels for each variable
    # Note:  drugSensitivityType refers to the list of variables
    #        that exist to describe the measured response of a cell line
    #        to a compound; examples are AUC and log(IC50)

    for jj in range(numDrugs):
        drugName.append(curDrugName[jj])
        for kk in range(len(drugSensitivityType)): 
            allVarsInclLabels.append(drugName[jj] + drugSensitivityType[kk]) 

    # Define folder for current target and see if it exists; if not make it
    localResultsFolder = resultsFolder + curTarg + '/'
    if os.path.exists(localResultsFolder):
        pass
    elif ~os.path.exists(localResultsFolder):
        os.mkdir(localResultsFolder)

        ###########################################################################
        # Execute CellLine Master Method for current target and associated compounds
        # CellLine_Master_Method() is the method that creates a graph and partitions
        # it recursively
        ###########################################################################
#         PrintStartTime(str(ii) + ' ; \t Starting POET for ' + curTarg + ' with ' + str(numDrugs) + ' compounds:' + str(compList[ii]))
#         curNumVerts = CellLine_Master_Method()
#         curNumVerts=[0]

        ###########################################################################
        # Print results of POET for current target
        ###########################################################################
#         PrintEndTime(str(ii) + ' ; \t' + curTarg + ' ; \t' + str(len(drugName)) + ' drugs; \t' + str(sum(curNumVerts)) + ' cell lines \t' + str(len(curNumVerts)) + ' subpops ; \t')

        # DrawHierarchicalTree(figFolder + expTag + 'HierarchicalTree_' + str(ii+1) + '.png', resultsFolder+str(ii+1)+'/'+expTag+str(ii+1)+'_')
#         DrawHierarchicalTreeWithOutcomes(figFolder + expTag + 'HierarchicalTree_' + str(ii+1) + '.png', resultsFolder+str(ii+1)+'/'+expTag+str(ii+1)+'_', '')

    return


# In[6]:


def convertSensList(sensValList, cellLineList, tissueList, drugName, varUsed):
    import numpy as np
    import pickle
    
    numDrug = len(drugName)
    numFeatures = sensValList.shape[1]
    drugSensValsX = list()
    
    
    
    for ii in range(numDrug):
        drugSensValsX.append([])
        index = np.arange(ii*varUsed, ii*varUsed+varUsed)
        tempVars = list()
        for sample in sensValList:
            curSampleVar = list()
            for jj in index:
                curSampleVar.append(sample[jj]) # 组成[AUC,IC50]
            drugSensValsX[ii].append(curSampleVar)
           
    
    
    cellLinesX = cellLineList.tolist()
    studyLineAbbX = tissueList.tolist()
    
    f1 = open(graphsFolder + expTag + curTarg + '_' + 'RawGraphData' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize),'wb')
    pickle.dump(cellLinesX,f1) 
    pickle.dump(drugSensValsX,f1) 
    pickle.dump([],f1) 
    pickle.dump(studyLineAbbX,f1) 
    f1.close()
    
    return drugSensValsX, cellLinesX, studyLineAbbX


# In[9]:


def CreateVertexProperty(g, valList, propType):
    """
        Method to create a vertex property holding a value in valList for
        every vertex in the graph

    :param g:
    :param valList:
    :param propType:
    :return:
    """

    newProp = g.new_vertex_property(propType)

    for ii in range(0,g.num_vertices()):
        newProp[g.vertex(ii)] = valList[ii]

    return newProp


# In[10]:


def CreatePatientDataVertexProperty(g, valList,type='object'):
    """

        Create a vertex property that holds heterogeneous(异质的) types of data in a list

    :param g:
    :param valList:
    :return:
    """

    # Create new property to hold raw clinical values
    patientDataVertexProp = g.new_vertex_property(type)

    # Iterate thru each vertex and add raw clinical values to each node
    for ii in range(len(valList)):

        patientDataVertexProp[g.vertex(ii)] = valList[ii]

    return patientDataVertexProp


# In[11]:


def PerformTStatNorm(vals): 
    """
        Method to perform traditional t-stat normalization where the values in
        vals are converted to a Normal(0,1) distribution by subtracting the mean
        and dividing by the STDDEV
    """

    arrayVals = np.array(vals)
    valsNorm = (arrayVals - np.mean(arrayVals))/np.std(arrayVals)

    return list(valsNorm)


# In[12]:


def PerformChiSquareNorm(valList):

    """

        Method to perform a chisquare calculation on every individual value in a list
        versus the remaining values in the list

    :param valList:
    :return:
    """

    chisq = list()
    for ii in range(len(valList)):


        # Made change to how chisquare is calcualted byincluding in the expected
        # set the value that is in the observed set; avoids the problem of singleton
        # occurrences in valList
        # tempValList = copy.copy(valList)
        # tempValList.remove(valList[ii])

        csq, p = CalcChiSquareP2CDistance(valList[ii], valList)
        chisq.append(csq)

    return chisq


# In[13]:



import numba as nb
# @nb.jit()
def CreateMultivariateEmdProperty(g, nmz):
    import multiprocessing
    from multiprocessing import Process
    
    import threading
    from graph_tool.all import Graph
    """
        Method to create an EMD property for a graph using the tStat property
        that exists at every vertex

    :param g:
    :return:
    """


    numVars = g.graph_properties['numVars'] # =4
    nmzClinVals = g.vertex_properties['nmzClinVals']  
    global emdProp
    emdProp = g.new_edge_property('vector<float>')


    
    
    # Create 3d array to hold emd values
    emd3D = np.zeros((g.num_vertices(),g.num_vertices(),numVars))
    
    nmzArray = np.array(nmz).transpose()

    from itertools import combinations
    combins = [c for c in  combinations(range(nmzArray.shape[0]), 2)]
    selfVer = [(ii,ii) for ii in range(nmzArray.shape[0])]
    combinsX = combins + selfVer
    
    combinsX=sorted(combinsX,key=lambda x:(x[0],x[1]))

    g.add_edge_list(combinsX)
    
    testTime = time.time()
    for ee in g.edges():
        sourceV = g.vertex_index[ee.source()]
        
        targetV = g.vertex_index[ee.target()]
        curValue = np.abs(np.subtract(nmzArray[sourceV],nmzArray[targetV]))
        emdProp[ee] = curValue.tolist()
        emd3D[sourceV, targetV, :] = curValue
        emd3D[targetV, sourceV, :] = curValue

    emdMats = list()
    

    for ii in range(numVars):
        emdMats.append(list(emd3D[:,:,ii].ravel()))

    return emdProp, emdMats


# In[14]:


@nb.jit()
def CreateMultivariateSimProp(g, propType, emdMatsList):

    """
        Method to develop a multivariate measure of graph similarity that
        is an extension of the univariate approach.  In this case, we
        construct a quasi-Gaussian (QG) multivariate model that uses
        a covariance that is based on the EMD distances derived between
        each pair of vertices for a specific variable.

    :param g:
    :param mu:
    :param propType:
    :return:
    """
    
    matSize = int(math.sqrt(len(emdMatsList[0]))) #len(emdMatsList[0]) = 862*862
    emdMats = list()
        
    emdMatsDict = {}

    for ii in range(len(emdMatsList)): #len(emdMastList) = 4
        emdMats.append(np.array(emdMatsList[ii]).reshape((matSize, matSize)))
    
    dictTime = time.time()
    for ii, feature in enumerate(emdMats):

        for jj, values in enumerate(feature):
            emdMatsDict[(ii,jj)] = values
    # Get emd values for every edge
    emd = g.edge_properties['emd']
    simProp = g.new_edge_property(propType)


    testTime = time.time()
    
    for ee in g.edges():
        # Get vector of EMDs for curEdge
        curEmds = emd[ee] # (4,1)

        sourceV = g.vertex_index[ee.source()]

        targetV = g.vertex_index[ee.target()]

        numVars = len(curEmds) # numVars = 4
        cov = np.zeros((numVars,numVars)) 

        # Iterate thru curEmds to fill covariance diagonals
        for kk in range(numVars):

            cov[kk,kk] = math.pow((curEmds[kk] + np.mean(emdMatsDict[(kk, sourceV)])+ np.mean(emdMatsDict[(kk,targetV)]))/3.0, 2)

            # Calculate off-diagonal similarity values 
            for ll in range(kk+1,numVars): # kk+1 到 numVars-1
                cov[kk,ll] = (curEmds[kk] + np.mean(emdMatsDict[(kk,sourceV)])+ np.mean(emdMatsDict[(kk,targetV)])) * (curEmds[ll]  + np.mean(emdMatsDict[(ll, sourceV)])+ np.mean(emdMatsDict[(ll, targetV)]))/9.0
                cov[ll,kk] = cov[kk,ll]
        
        simProp[ee] = math.exp(-1 * np.dot(np.array(curEmds), np.dot(np.linalg.pinv(cov), np.reshape(np.array(curEmds), (numVars,1)))))

    return simProp


# In[15]:


def PivotList(x):

    """
        Method to exchange dimensions of list of lists:  eg, list containing different
        variable values can be turned into list of patient lists, where each list contains
        multiple variables

    :param x:
    :return:
    """

    
    dim0 = len(x)

    if type(x[0]) == type([1]): 
        newList = list()
        dim1 = len(x[0])
        for ii in range(dim1): 
            curList = list()
            for jj in range(dim0): 
                curList.append(x[jj][ii])
            newList.append(curList)

    else:
        newList = list()
        for ii in range(dim0):
            newList.append([x[ii]])
    
    return newList


# In[16]:


def MakeMultivariatePatientNetworkGraph(valuesList, patientIdList):
    """
        Method to create multi-patient network model with measures of similarity
        between patients established using frequency counts of unique conditions
        that are shared pairwise between patients

    :param valuesList: lists of lists where each sub-list corresponds to a different
                       variable, and has as many values in it as the number of
                       patients (vertices)

    :param patientIdList: list of patientId values

    :return:
    """
    g = graph_tool.Graph(directed=False)

    #
    # Make vertices for graph
    #

    # Iterate over all patients in patientSubset and add vertex for each
    for ii in range(len(patientIdList)):
        g.add_vertex()

    # Create vertex prop for patient ID
    patientIdProp = CreateVertexProperty(g, patientIdList, 'string')

    g.vertex_properties['idProp'] = patientIdProp

    nmz = list()                    # list of normalized variables
    nmzValuesList = list()          # list of ???
#     # binary list of variables used, based on whether there is diversity in values
    varsUsedList = list()
    for ii in range(len(allVarsInclType)): #len(allVarsInclType) = 4

        if len( [ jj for jj,xx in enumerate(valuesList[ii]) if xx == valuesList[ii][0]] ) == len(valuesList[ii]):
#             # print 'Variable ' + str(ii) + ':  WARNING! ALL VALUES OF VARIABLE ' + allVarsInclLabels[ii] + ' ARE EQUAL'
            varsUsedList.append(0) #features值没有差异
            # sys.exit()
        else:
            nmzValuesList.append(valuesList[ii])
            if allVarsInclType[ii] == 'continuous': 
                
                nmz.append(PerformTStatNorm(valuesList[ii]))
                # print 'Variable ' + str(ii) + ':  ' + allVarsInclLabels[ii] + ' is continuous'
            elif allVarsInclType[ii] == 'categoricalFromContinuous':
                
                nmz.append(PerformTStatNorm(PerformChiSquareNorm(valuesList[ii])))
                # print 'Variable ' + str(ii) + ':  ' + allVarsInclLabels[ii] + ' is categoricalFromContinuous'
            elif allVarsInclType[ii] == 'categorical': 
                chisq = PerformChiSquareNorm(valuesList[ii])
                if len( [ jj for jj,xx in enumerate(chisq) if xx == chisq[0]] ) == len(valuesList[ii]):
                    nmz.append([0.0] * len(chisq))
                else:
                    nmz.append(PerformTStatNorm(chisq))                # nmz.append(PerformTStatNorm(PerformChiSquareNorm(valuesList[ii])))
                # print 'Variable ' + str(ii) + ':  ' + allVarsInclLabels[ii] + ' is categorical'
            elif allVarsInclType[ii] == 'binary': # 二元类型变量
                chisq = PerformChiSquareNorm(valuesList[ii])
                if len( [ jj for jj,xx in enumerate(chisq) if xx == chisq[0]] ) == len(valuesList[ii]):
                    nmz.append([0.0] * len(chisq))
                else:
                    nmz.append(PerformTStatNorm(chisq))
                # nmz.append(PerformTStatNorm(PerformChiSquareNorm(valuesList[ii])))
                # print 'Variable ' + str(ii) + ':  ' + allVarsInclLabels[ii] + ' is binary'
            varsUsedList.append(1)

    varsUsed = g.new_graph_property('object')
    varsUsed[g] = varsUsedList
    g.graph_properties['varsUsed'] = varsUsed

    
    clinVals = CreatePatientDataVertexProperty(g, PivotList(valuesList))
    g.vertex_properties['clinVals'] = clinVals
#     print('vertex prop: ', clinVals[g.vertex(0)])
    graphNameProp = g.new_graph_property('string')
    graphNameProp[g] = 'g0'
    g.graph_properties['graphName'] = graphNameProp

    #
    # EDGE CONSTRUCTION
    #

    # Proceed to edge construction

    if sum(varsUsedList) > 0 and len(patientIdList) > 2:
        # At least 1 variables had diversity(

        # Make graph property to hold number of variables based on demo, obs, and cdp vars
        numVars = g.new_graph_property('int')

        numVars[g] = len(nmzValuesList)
        g.graph_properties['numVars'] = numVars

        # Make vertex property to hold normalized clinical values
        nmzClinVals = CreatePatientDataVertexProperty(g, PivotList(nmz))
        g.vertex_properties['nmzClinVals']  = nmzClinVals

        # Make new graph property to hold emd matrices
        emdMats = g.new_graph_property('object')

       
        emdTime = time.time()
        emdProp, ggMatsList = CreateMultivariateEmdProperty(g, nmz)
        
        
        EMDtime = time.time()-emdTime
        EMDTimeList.append(EMDtime)
#         print('EMD Time:', time.time()-emdTime)
        
        
        emdMats[g] = ggMatsList
        g.edge_properties['emd'] = emdProp
        g.graph_properties['emdMats'] = emdMats       
        
        simTime = time.time()
        simProp = CreateMultivariateSimProp(g, 'float', ggMatsList)
        SimTime = time.time() - simTime

        simTimeList.append(SimTime)

        g.edge_properties['simProp'] = simProp
    
    else:
        # NO variables had diversity; halt construction
        print('No variables had diversity, halting graph construction')
        ggMatsList = []
        pass

    
    return g, ggMatsList


# In[17]:


def MakeCellLineNetwork():
    """

        Method to create a cell line network (CLN) using the raw graph data values that have been
        processed, curated, and stored beforehand

    :param drugNum:
    :return:
    """

    
    # Save final lists of data for current processing
    f1 = open(graphsFolder + expTag + curTarg + '_' + 'RawGraphData' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize),'rb')
    # f1 = open(graphsFolder + expTag + 'RawGraphData','r')
    cellLineX = pickle.load(f1) # cell line(cell line info.csv 和 AUC.csv, IC50.csv 均存在的cell line)
    sensValsX = pickle.load(f1) # drug sens (feature)
    # binSensVals = cPickle.load(f1) 
    valuesList = pickle.load(f1) # 一个空list
    studyLineAbbX = pickle.load(f1) #表中的Tissue列？
    f1.close()


    # Make one list of lists with variables and then pivot
    sensValsList = list()
    for ii in range(len(drugName)):
        for jj in range(len(drugSensitivityType)):
            curVals = list()
            for kk in range(len(cellLineX)):
                curVals.append(sensValsX[ii][kk][jj])
            sensValsList.append(curVals)
    
    # Set graph-making variables to all floats
    global allVarsInclType
    allVarsInclType = list()
    global allVarsInclCats
    allVarsInclCats = list()
    for ii in range(len(drugName)*len(drugSensitivityType)):
        allVarsInclType.append('continuous')
        allVarsInclCats.append([])
    global allVarsInclLabels
    allVarsInclLabels = list()
    for ii in range(len(drugName)):
        for jj in range(len(drugSensitivityType)):
            allVarsInclLabels.append(drugName[ii]+'_'+drugSensitivityType[jj])
            

    #
    # CONSTRUCT GRAPH
    #


    g0, curMatsList = MakeMultivariatePatientNetworkGraph(sensValsList, cellLineX)
#     MakeMultivariatePatientNetworkGraph(sensValsList, cellLineX)

    # Add studyLineAbbX as vertex property
    studyLineAbbXProp = g0.new_vertex_property('string')
    for ii in range(g0.num_vertices()):
        studyLineAbbXProp[g0.vertex(ii)] = studyLineAbbX[ii]
    g0.vertex_properties['studyLineAbb'] = studyLineAbbXProp

    # Save graph to file
#     g0.save(graphsFolder + expTag + curTarg + '_' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize)+ 'CellLineNetwork.gt')
    
    return g0


# In[20]:


def ExtractListsFromVertices(vertexProp, g): 
    """
        Method to extract the lists at each vertex of a vertex property,
        vertexProp, belonging to a graph, g, and to return a list of lists,
        where each sub-list is a list of values from each vertex

    :param vertexProp:
    :param g:
    :return:
    """

    # Make a list to hold lists from each vertex
    varList = list()

    # Iterate thru each vertex and extract list associated with vertexProp
    for ii in range(g.num_vertices()):
        varList.append(vertexProp[g.vertex(ii)])

    return varList


# In[21]:


def PerformRwLaplacian(x):
    import numpy as np
#     print(type(x))
#     print('-'*6, 'Get SymLaplacian', '-'*6)
    
    for ii in range(len(x)):
        x.itemset((ii, ii), 1)
    
    l = -x
    
    # Calculate sum of rows of x
    gamma = np.sum(x,axis=1).tolist()

    gamma = sum(gamma, [])

    degree = np.diag(gamma)

    
    degreeInv = np.linalg.inv(degree)
    
    # compute Laplacian matrix
    for ii in range(len(gamma)):
        l.itemset((ii, ii), gamma[ii] - x.item((ii, ii)))
 
    l = np.dot(degreeInv, l)

    return l


# In[22]:


def PartitionGraphAndSegment(g, parentLabel):
    import time
    """

        Method to partition a graph, g, having label, parentLabel
        into 2 graphs using the Fiedler eigenvector

    :param g:  图
    :param parentLabel: 图的名称
    :return:
    """

    #
    # DERIVE FIEDLER EIGENVECTOR AND SEGMENT CLN
    #
    adj = adjacency(g, g.edge_properties['simProp']).todense()

    
    lap = PerformRwLaplacian(adj) # Laplacian矩阵

#     print()
    u,s,v = np.linalg.svd(lap)

    fVec = u[:,len(s)-2]

    fVal = s[len(s)-2]
    

    # Convert fVec into binary vector where vals < 0 --> 0 and vals > 0 --> 1
    binFVec = list()
    for jj in range(len(fVec)):
        if fVec[jj] > 0.0:
            binFVec.append(1)
        else:
            binFVec.append(0)
    if sum(binFVec) > math.floor(len(binFVec)/2.0): # why? math.floor取最接近的整数
        for ii in range(len(binFVec)):
            if binFVec[ii] == 0:
                binFVec[ii] = 1
            else:
                binFVec[ii] = 0

    # Extract clinical values for class0 and class1
    clinValsProp = g.vertex_properties['clinVals']

    clinVals = ExtractListsFromVertices(clinValsProp, g)
    # Segregate clinVals into two groups, class0 and class1
    clinVals0 = [ clinVals[ii] for ii,xx in enumerate(binFVec) if xx == 0 ]
    clinVals1 = [ clinVals[ii] for ii,xx in enumerate(binFVec) if xx == 1 ]

    # Extract id property and segregate
    idProp = g.vertex_properties['idProp'] 
    ids = ExtractListsFromVertices(idProp,g)
    ids0 = [ ids[ii] for ii,xx in enumerate(binFVec) if xx == 0 ]
    ids1 = [ ids[ii] for ii,xx in enumerate(binFVec) if xx == 1 ]

    # Extract studyLineAbb property
    studyLineAbbProp = g.vertex_properties['studyLineAbb']
    studyLineAbbVals = list()
    for ii in range(g.num_vertices()):
        studyLineAbbVals.append(studyLineAbbProp[g.vertex(ii)])
    studyLineAbbVals0 = [ studyLineAbbVals[ii] for ii,xx in enumerate(binFVec) if xx == 0 ]
    studyLineAbbVals1 = [ studyLineAbbVals[ii] for ii,xx in enumerate(binFVec) if xx == 1 ]

    # print 'Fiedler eigenvalue is ' + str(fVal)
    # print 'Partitioning found ' + str(len(ids0)) + ' vertices in first partition of ' + parentLabel
    # print 'Partitioning found ' + str(len(ids1)) + ' vertices in second partition of ' + parentLabel
    
#     print('Partitioning found', str(len(ids0)), ' vertices in first partition of ', parentLabel)
#     print('Partitioning found', str(len(ids1)), ' vertices in second partition of ', parentLabel)
    #
    # Calculate silhouette score for partition
    #

    # Get adjacency from g
    adj = np.array(adjacency(g, g.edge_properties['simProp']).todense())
    # Make sure diagonal values are = 1.0
    for ii in range(g.num_vertices()):
        adj.itemset((ii,ii), 1.0)
    sscore = silhouette_score(1.0 - adj, np.array(binFVec), metric='precomputed')
#     print('Silhouette score of new graphs is ', str(sscore))
    # print 'Silhouette score of new graphs is ' + str(sscore)

    # Create new graphs for class0 and class1

    
#     print()
#     print('-'*10, 'PivoList(clinVals0)', '-'*10)
#     print('clinVals0: ', np.array(clinVals0).shape) # shape = (#cell line, #features )
#     print()
#     print('clinVals0: ', np.array(PivotList(clinVals0)).shape) # shape = (#features, #cell line)
    
    # 第一个参数: drug sens (行为feature, 列为cell line) 第二个参数: cell line名称
    g0, curMatsList0 = MakeMultivariatePatientNetworkGraph(PivotList(clinVals0), ids0)
    g1, curMatsList1 = MakeMultivariatePatientNetworkGraph(PivotList(clinVals1), ids1)

    # Get number ov variables actually used in graph construction

    varsUsed0 = g0.graph_properties['varsUsed']
    numVarsUsed0 = sum(varsUsed0)
    varsUsed1 = g1.graph_properties['varsUsed']
    numVarsUsed1 = sum(varsUsed1)

    # Add graph name to each graph
    graphName0 = g0.new_graph_property('string', parentLabel + '0')
    g0.graph_properties['graphName'] = graphName0
    graphName1 = g1.new_graph_property('string', parentLabel + '1')
    g1.graph_properties['graphName'] = graphName1

    # Add parent name to each graph
    parentName0 = g0.new_graph_property('string')
    parentName0[g0] = parentLabel
    g0.graph_properties['parentName'] = parentName0
    parentName1 = g1.new_graph_property('string')
    parentName1[g1] = parentLabel
    g1.graph_properties['parentName'] = parentName1

    # Add studyLineAbb property to g0 and g1
    studyLineAbbProp0 = g0.new_vertex_property('string')
    for ii in range(g0.num_vertices()):
        studyLineAbbProp0[g0.vertex(ii)] = studyLineAbbVals0[ii]
    g0.vertex_properties['studyLineAbb'] = studyLineAbbProp0
    
    studyLineAbbProp1 = g1.new_vertex_property('string')
    for ii in range(g1.num_vertices()):
        studyLineAbbProp1[g1.vertex(ii)] = studyLineAbbVals1[ii]
    g1.vertex_properties['studyLineAbb'] = studyLineAbbProp1

    # print
    # print 'Graph ' + graphName0[g0] + ' has ' + str(len(ids0)) + ' vertices using ' + str(numVarsUsed0) + ' variables'
    # print 'Graph ' + graphName1[g1] + ' has ' + str(len(ids1)) + ' vertices using ' + str(numVarsUsed1) + ' variables'
    # print
    # print'***************************************************************************'
    # print
    
    return g0, g1, binFVec, sscore, varsUsed0, varsUsed1


# In[23]:


def CompleteRecursivelyPartitionGraph(g):
# def CompleteRecursivelyPartitionGraph():
    # 迭代切图
    """

        Method to recursively partition a graph-based network until stopping criterion are met

        Unlike RecursivelyPartitionGraph, this method partitions all resulting
        partitions until they meet a stopping criterion

    :param g:
    :return:
    """

#     print()
#     print('-'*10, 'CompleteRecursivelyPartitionGraph', '-'*10)
    
    # Get total number of variables used for each graph
    numVars = g.graph_properties['numVars'] # numVars = 4

    # Set up variables to be captured per iteration
    iterCtr = 0
    iterBinPartList = list()
    iterSscoreList = list()

    # gList is the list of graphs generated from each partition
    gList = list() # 图的list
    gListCtr = 0
    gList.append(g)
    gListGraphNameList = list()
    gListGraphNameList.append('g0')
    gListNumVertsList = list() 
    gListNumVertsList.append(g.num_vertices())
    gListVarsUsedList = list() 
    gListVarsUsedList.append([1]*numVars)

    # Set up closure list to keep track of whether a specific sub-pop
    # is closed or not; if a value is 0, then no more partitioning should occur
    gListClosure = list()
    gListClosure.append(1) 

    # Perform graph partitioning until resulting graphs have too few vertices or no more
    # covariates exist that have any diversity remaining
    continuePartitioning = True
#     PartitionGraphAndSegment(gList[gListCtr], gListGraphNameList[gListCtr])

    while gListCtr < len(gList) and continuePartitioning == True:

        #
        # Perform partition 
        #
        g0, g1, binPart0, sscore0, varsUsed0, varsUsed1 = PartitionGraphAndSegment(gList[gListCtr], gListGraphNameList[gListCtr])


        # 2-10-17:  modify acceptance criteria for new subpops:  the new criteria uses 2 factors:
        #           1)  is the separation between the 2 new subpops greater than the threshold:  sscoreTh
        #           2)  are the sizes of the 2 new subpops both greater than the threshold:  minSubpopSize

        if sscore0 > sscoreTh and g0.num_vertices() >= minSubpopSize and g1.num_vertices() >= minSubpopSize:

            # Close off current graph in closure list
            gListClosure[gListCtr] = -1

            # Add graphs to gList
            gList.append(g0)
            gList.append(g1)

            # Assess closure of resulting partitions for future partitioning
            if g0.num_vertices() < minNumVerts or float(sum(varsUsed0)) == 0.0:
                gListClosure.append(0) 
            else:
                gListClosure.append(1) 

            if g1.num_vertices() < minNumVerts or float(sum(varsUsed1)) == 0.0:
                gListClosure.append(0)
            else:
                gListClosure.append(1)

            #
            # Keep track of bookkeeping for new graph additions
            #

            # Derive position of new graphs to be added to gList
            # This should be the length of gList - 2 and lenth of gList - 1
            newGraphInds1 = len(gList) - 2
            newGraphInds2 = len(gList) - 1


            # Append binPart0
            iterBinPartList.append((newGraphInds1, newGraphInds2, binPart0))

            # Append number of patients in resulting graphs
            gListNumVertsList.append(g0.num_vertices())
            gListNumVertsList.append(g1.num_vertices())

            # Keep list of variables used in each graph
            gListVarsUsedList.append(varsUsed0)
            gListVarsUsedList.append(varsUsed1)

            # Add graph names for new graphs
            lenGraphName = len(list(gListGraphNameList))
            graphName0 = g0.graph_properties['graphName']
            graphName1 = g1.graph_properties['graphName']
            gListGraphNameList.append(graphName0)
            gListGraphNameList.append(graphName1)

            # Save graphs
            graphName0 = g0.graph_properties['graphName']
#             g0.save(graphsFolder + expTag + curTarg + '_' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize)+'_' + 'MultivariateGraph.gt')
            graphName1 = g1.graph_properties['graphName']
#             g1.save(graphsFolder + expTag + curTarg + '_' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize)+'_'+graphName1 + 'MultivariateGraph.gt')

        else:
            # Current partition is unacceptable, reject partition and move forward
            # Close off current graph in closure list
            gListClosure[gListCtr] = 0

        # Increment gListCtr by 2 to account for the 2 new graphs added to gList
        # gListCtr += 2

        # Determine whether to continue partitioning based on closure
        # value of graphs remaining in gList
        # if any(gListClosure[gListCtr-1:len(gListClosure)]) == 1:
        if any(gListClosure[gListCtr+1:len(gListClosure)]) == 1:
            # There is a graph that is not closed; now find out which is the first one
            # and move gListCtr to it
            openGraphInds = [ jj for jj,xx in enumerate(gListClosure) if xx == 1 ]
            gListCtr = openGraphInds[0]
            # Advance iteration counter
            iterCtr += 1
        else:
            continuePartitioning = False

    # Save all variables describing partition
    f0 = open(savedDataFolder + expTag + curTarg + '_' + 'PartitioningResults' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize), 'wb')
    pickle.dump(iterBinPartList, f0)
    pickle.dump(gListGraphNameList, f0)
    pickle.dump(gListNumVertsList, f0)
    pickle.dump(gListVarsUsedList, f0)
    pickle.dump(gListClosure, f0)
    pickle.dump(gList, f0)
    f0.close()

    return gList, gListClosure, iterBinPartList, iterSscoreList


# In[25]:


def saveTime(makeNetworkTime, cutTime, totalTime):
    import os
    saveTimeFolder = 'RWB_TimeInfo'
    
    if os.path.exists(saveTimeFolder):
        pass
    else:
        os.mkdir(saveTimeFolder)
    
    fileName = '_'.join(drugName)
    
    
    f = open(saveTimeFolder+ '/' +fileName + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize), 'wb')
    pickle.dump(makeNetworkTime, f)
    pickle.dump(cutTime, f)
    pickle.dump(totalTime, f)


# In[27]:


def ExtractSubpopulations(gList, gListClosure):
    """

        Method to extract subpopulation information from results of
        recursive graph segmentation

    :param gList:
    :param gListClosure:
    :return:
    """

    # Rebuild gList from all the partitions
    # gList, gListClosure, iterBinPartList, iterSscoreList = RebuildGList(drugNum)

    # Iterate thru gList and extract data about vars used and verts
    numVerts = list()
    varsUsed = list()
    numVarsUsed = list()
    gName = list()
    gListInd = list()
    clinVals = list()
    cellLines = list()

    for ii in range(len(gListClosure)):
        curVarsUsed = gList[ii].graph_properties['varsUsed']
        curClinVals = list()
        # Check to see whether graph is closed or could not be split,
        # in either case, it is a graph that is a final constituent
        # graph of the population
        if gListClosure[ii] == 0: 
            # Current graph is a constituent graph; so keep track of it and its parameters
            gListInd.append(ii)
            numVerts.append(gList[ii].num_vertices())
            numVarsUsed.append(sum(curVarsUsed))
            gName.append(gList[ii].graph_properties['graphName'])
            varsUsed.append(gList[ii].graph_properties['varsUsed'])

            # Get clinical values out of current graph
            clinValsProp = gList[ii].vertex_properties['clinVals']
            curClinVals = ExtractListsFromVertices(clinValsProp, gList[ii])
            # for jj in range(len(curClinValsStr)):
            #     curClinVals.append(curClinValsStr[jj].dataValueList)

            # Get names of patients in current graph and keep track of this
            idProp = gList[ii].vertex_properties['idProp']
            curPatients = ExtractListsFromVertices(idProp, gList[ii])
            clinVals.append(curClinVals)
            cellLines.append(curPatients)

        else:
            # This graph is not a constituent graph
            pass


    # Write results to file
    f = open(localResultsFolder + expTag + curTarg + '_' + 'SubpopResults' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize), 'wb')
    pickle.dump(numVerts, f)
    pickle.dump(numVarsUsed, f) 
    pickle.dump(gName, f) 
    pickle.dump(varsUsed, f) 
    pickle.dump(clinVals, f) 
    pickle.dump(cellLines, f) 
    pickle.dump(gListClosure, f)
    pickle.dump(gListInd, f)
    f.close()

    return numVerts, numVarsUsed, gName, varsUsed, clinVals, cellLines, gListInd


# <h2>Save data and compute silhouette  </h2>

# In[29]:


def saveDataListAndLabel(localResultsFolder, expTag, curTarg, sensValList, cellLineList):
    import pickle
    import numpy as np
    from collections import Counter
    f = open(localResultsFolder + expTag + curTarg + '_' + 'SubpopResults' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize), 'rb')
    numVerts = pickle.load(f)
    numVarsUsed = pickle.load(f)
    gName = pickle.load(f)
    varsUsed = pickle.load(f)
    clinVals = pickle.load(f)
    cellLines = pickle.load(f)
    gListClosure = pickle.load(f)
    gListInd = pickle.load(f)
    f.close()

    
    label = 0

    print('Number of clusters: ', len(gListInd))

    
    labelArray = np.zeros(len(sensValList))
    cellLineArray = np.empty(len(sensValList), dtype='str')

    
    for pp in clinVals: 
        for mm in pp: 
            for jj, vv in enumerate(sensValList):
                if Counter(mm) == Counter(vv):
                    labelArray[jj] = label
        label += 1
    
    for pp in cellLines:
        for mm in pp:
            for jj, vv in enumerate(cellLineList):
                if mm == vv:
                    cellLineArray[jj] == vv


    import os
    saveDataLabelFolder = 'RWB_DataAndLabel'
    
    if os.path.exists(saveDataLabelFolder):
        pass
    else:
        os.mkdir(saveDataLabelFolder)

#     print('DrugName:',drugName)
    fileName = '_'.join(drugName)
#     print('File Name:', fileName)
    
    f0 = open(saveDataLabelFolder+'/'+fileName + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize), 'wb')
    pickle.dump(sensValList, f0)
    pickle.dump(labelArray, f0)
    pickle.dump(cellLineArray, f0)
    f0.close()
    
    return sensValList, labelArray, cellLineArray


# In[31]:


def computeSilhouette(savedDataFolder, localResultsFolder, expTag, curTarg, dataList, labelList):
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    import dunn
    from s_dbw import S_Dbw
    f0 = open(savedDataFolder + expTag + curTarg + '_' + 'PartitioningResults' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize), 'rb')
    
    pickle.load(f0)
    pickle.load(f0)
    pickle.load(f0)
    pickle.load(f0)
    pickle.load(f0)
    gList = pickle.load(f0)
    f0.close()
    
#     print(gList)
    originalG = gList[0]
    originalDrugSensProp = originalG.vertex_properties['clinVals']
    originalDrugSens = ExtractListsFromVertices(originalDrugSensProp, originalG)
#     print(np.array(originalDrugSens).shape)

    f = open(localResultsFolder + expTag + curTarg + '_' + 'SubpopResults' + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize), 'rb')
    numVerts = pickle.load(f)
    numVarsUsed = pickle.load(f)
    gName = pickle.load(f)
    varsUsed = pickle.load(f)
    clinVals = pickle.load(f)
    cellLines = pickle.load(f)
    gListClosure = pickle.load(f)
    gListInd = pickle.load(f)
    f.close()
    
    from collections import Counter
    labelVector = np.zeros((len(originalDrugSens),1))
#     print(labelVector.shape)
    curLabel = 0
    
#     print(np.array(clinVals).shape)
    for ii, clinCluster in enumerate(clinVals):
        for jj, curSens in enumerate(clinCluster):
#             if ii == 0:
#                 print(Counter(curSens))
#                 print()
            for kk, oriSens in enumerate(originalDrugSens):
#                 if kk ==0:
#                     print(Counter(oriSens))
                if Counter(curSens) == Counter(oriSens):
                    labelVector[kk] = curLabel
        curLabel += 1
    
#     print('Check # clusters: ', np.unique(labelVector).shape[0]==len(gListInd))
#     print('Check length: ', len(labelVector) == len(originalDrugSens))
    
    
    
    # Get adjacency from g
    adj = np.array(adjacency(originalG, originalG.edge_properties['simProp']).todense())
    # Make sure diagonal values are = 1.0
    for ii in range(originalG.num_vertices()):
        adj.itemset((ii,ii), 1.0)
        
    fileName = '_'.join(drugName)
    f0 = open(fileName + '_adjMatrix', 'wb')
    pickle.dump(adj, f0)
    f0.close()    
    
    
    # Silhouette Score
    sscore = silhouette_score(1.0 - adj, np.array(labelVector.ravel()), metric='precomputed')
    print('Silhouette=', sscore)

    # Cal-Ha Score
    ch_score = calinski_harabasz_score(dataList, labelList)
    print('Cal-Har=', ch_score)
    
    # DB score
    db_score = davies_bouldin_score(dataList, labelList)
    print('Davies-Bouldin=', db_score)
    
    # dunn score
    uniqueLabel = np.unique(labelList)
    clusterList = list()
    for xx in uniqueLabel:
        ids = np.where(labelList==xx)
        curCluster = dataList[ids[0]]
        clusterList.append(curCluster)
    print(len(clusterList))
        
    dunnS = dunn.dunn(clusterList)
    print('Slow Dunn=', dunnS)
    

    # PBM
    pbm = tools.computePBM(dataList,labelList)
    print('PBM=',pbm)
    
    # SDB_W
    sdbw_score = S_Dbw(dataList, labelList)
    print('SDB_W=', sdbw_score)
    
    # RMSSTD
    rmsstd = tools.computeRMSSTD(dataList, labelList)
    print('RMSSTD=', rmsstd)
    
    import os
    saveSScoreFolder = 'RWB_SScore'
    
    if os.path.exists(saveSScoreFolder):
        pass
    else:
        os.mkdir(saveSScoreFolder)
    
    fileName = '_'.join(drugName)
    
    f0 = open(saveSScoreFolder+'/'+ fileName + '_mV_' + str(minNumVerts) + '_ssT_'+ str(sscoreTh) + '_msz_' + str(minSubpopSize), 'wb')
    pickle.dump(sscore, f0)
    pickle.dump(ch_score,f0)
    pickle.dump(db_score,f0)
    pickle.dump(dunnS, f0)
    pickle.dump(pbm, f0)
    pickle.dump(sdbw_score, f0)
    pickle.dump(rmsstd, f0)
    f0.close()
    
    return sscore


# In[ ]:





# In[33]:


from initRWBFileInfo import *
import time
import pickle 
import os
import math
from collections import Counter 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare 
from scipy.stats import ttest_ind as ttest
import csv
from sklearn.metrics import silhouette_score
import tools
from graph_tool.all import *
def main(ids, curDrugs, cellLineInfoDf, AUCDf, ICDf, minNumV, ssTh, minSubSize):

    print('RW Start')
    totalSTime = time.time()

    global EMDTimeList
    EMDTimeList = list()
    global simTimeList
    simTimeList = list()
    
    global minNumVerts
    global sscoreTh
    global minSubpopSize
    
    minNumVerts = minNumV
    sscoreTh = ssTh
    minSubpopSize = minSubSize


    
    MakeFolder(opFolder)

    
    IterateOverTargets(ids, curDrugs)
    
    
    
    sensValList, cellLineList, tissueList = tools.getCorrespondenceInfo(curDrugs, cellLineInfoDf, AUCDf, ICDf)
    drugSensValsX, cellLinesX, studyLineAbbX = convertSensList(sensValList, cellLineList, tissueList, drugName, len(drugSensitivityFile))
    
    print('Cell Lines Number', len(sensValList))
    
    startTime = time.time()
    g = MakeCellLineNetwork()

    makeNetworkTime = time.time() - startTime
    print('MakeCellLine Time:',makeNetworkTime)
    
    
    cutSTime = time.time()
    gList, gListClosure, iterBinPartList, iterSscoreList = CompleteRecursivelyPartitionGraph(g)

    cutTime = time.time()-cutSTime
    print('Cut Time: ', cutTime)
    
    runningTime = time.time() - totalSTime
    print('Total Running Time:', time.time()-totalSTime)
    
    saveTime(makeNetworkTime, cutTime, runningTime)
    
    numVerts, numVarsUsed, gName, varsUsed, clinVals, cellLines, gListInd = ExtractSubpopulations(gList, gListClosure)

    
    
    dataList, labelList, cellLineList = saveDataListAndLabel(localResultsFolder, expTag, curTarg, sensValList, cellLineList)
    
    sscore = computeSilhouette(savedDataFolder, localResultsFolder, expTag, curTarg, dataList, labelList)


    
#     fileName = '-'.join(curDrugs)
#     f0 = open(fileName+'_CompareTime', 'wb')
#     pickle.dump(EMDTimeList, f0)
#     pickle.dump(simTimeList, f0)
#     pickle.dump(makeNetworkTime, f0)
#     pickle.dump(cutTime, f0)
#     pickle.dump(runningTime, f0)
    
#     print('Running Time:', runningTime)
#     print('RW End')
#     print()
    print()
    return dataList, labelList, cellLineList


# In[ ]:


def MakeIc50Subpop(clinVals, numVerts):
    """

        Method to plot the mean values for logIc50 for each subpopulation along
        with results of significance testing between the first drug and the others

    :param meanIc50:
    :param clinVals:
    :param numVerts:
    :return:
    """

    # Extract  logIc50 data for the 3 drugs, in each subpop
    meanIc50 = [ [] for ii,xx in enumerate(range(numDrugs)) ]
    stdIc50 = [ [] for ii,xx in enumerate(range(numDrugs)) ]
    for ii in range(len(clinVals)):
        for jj in range(numDrugs):
            meanIc50[jj].append(np.mean(PivotList(clinVals[ii])[2*jj + 1]))
            stdIc50[jj].append(np.std(PivotList(clinVals[ii])[2*jj + 1]))

    # Get number of subpops
    Nsubpops = len(meanIc50[0])
    colorStr = ['b','r','g', 'm', 'c', 'k', 'y', 'b', 'r', 'g', 'm']

    # Calculate tTEsts
    tTests = list()
    for ii in range(len(clinVals)):
        curTestVals = PivotList(clinVals[ii])[1]
        curTtest = list()
        for jj in range(1,numDrugs):
            curBaseVals = PivotList(clinVals[ii])[2*jj + 1]
            ###### USE TWO SIDED TEST
            # curT, curProb = ttest(curTestVals, curBaseVals)
            # if curProb < 0.01:
            #     curTtest.append('**')
            # elif curProb < 0.05:
            #     curTtest.append('*')
            # else:
            #     curTtest.append('')
            ###### USE ONE SIDED TEST
            curT, curProb = ttest(curTestVals, curBaseVals)
            if curProb/2.0 < 0.01 and curT < 0:
                curTtest.append('**')
            elif curProb/2.0 < 0.05 and curT < 0 :
                curTtest.append('*')
            else:
                curTtest.append('')
        tTests.append(curTtest)

    # Make xtr labels
    xtickStrs = list()
    for ii in range(Nsubpops):
        xtickStrs.append(str(ii+1) + ' (' + str(numVerts[ii])+')')

    # Determine spacing parameters for numDrugs
    barWidth = 0.90 / float(numDrugs)

    # Make figure for different logIc50 responses in each subpop for each drug
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    for ii in range(numDrugs):
        ax1.bar(np.arange(0.60+barWidth*ii,Nsubpops+0.5,1), meanIc50[ii], barWidth, color=colorStr[ii], label=drugName[ii])

    plt.grid()
    ax1.set_xticks(np.arange(1,Nsubpops+1))
    ax1.set_xticklabels(xtickStrs, rotation='vertical')
    ax1.set_title('Average Log(IC50) Responses for Each Sub-population \n * = p < 0.05; ** = p < 0.01')
    ax1.set_xlabel('Sub-population Number & Size \n ' + str(sum(numVerts)) + ' Used')
    ax1.set_ylabel('Log(IC50)')

    # Add significance testing results
    x1,x2,y1,y2 = plt.axis()
    plt.axis([x1,x2,y1,y2])
    plt.axis([x1,x2,y1-1.0,y2])
    x1,x2,y1,y2 = plt.axis()
    delta = 1.0 / float(numDrugs)
    for jj in range(Nsubpops):
        for kk in range(0,numDrugs-1):
            plt.text(jj+0.80,y1+kk*delta,tTests[jj][kk],color=colorStr[kk+1])

    plt.legend()
    fig.tight_layout()
    plt.savefig(localResultsFolder + expTag + curTarg + '_'  + 'MultiDrugLogIc50.png')
#     plt.close()

    return


# In[ ]:


def EvaluateSubpopDist():
    """

        Method to evaluate the distribution of tissues from which the cell lines
        originate from, within each subpopulation.

        This method has been altered from its original incarnation to create a
        composite variable that captures the impact of different pharmacology
        parameters in terms of a key attribute.  IN this case, the composite
        variable represents the sensitivity of a cell line to a compound by
        dividing the log IC50 value by the AUC.

        The ordering that results is critical for the filenaming and export
        of data, since the subpops are sorted by the value of the composite
        variable.  This variable, named here indsSMinMeanCompVar, is saved
        as part of the composite variables file

    :return:
    """

    # Load subpop results
    f = open(localResultsFolder + expTag + curTarg + '_'  + 'SubpopResults', 'rb')
    numVerts = pickle.load(f)
    numVarsUsed = pickle.load(f)
    gName = pickle.load(f)
    varsUsed = pickle.load(f)
    clinVals = pickle.load(f)
    cellLines = pickle.load(f)
    gListClosure = pickle.load(f)
    f.close()

    Nsubpops = len(numVerts)

    # Open original CLN
    origG = load_graph(graphsFolder + expTag + curTarg + '_' + 'CellLineNetwork.gt')
    origTissTypeProp = origG.vertex_properties['studyLineAbb']
    origTissTypes =  ExtractListsFromVertices(origTissTypeProp, origG)

    # Get pop-level stats
    # 得到不重复的tissue list以及对应tissue出现的数量
    uniqTissTypes = sorted(list(set(origTissTypes)))
    uniqTissTypesCounts = list()
    for ii in range(len(uniqTissTypes)):
        uniqTissTypesCounts.append(origTissTypes.count(uniqTissTypes[ii]))
    

    #
    # CALCULATE COMPOSITE VARIABLE FOR EVERY DRUG
    #

    # Calculate composite variable for each subpop
#     print(np.array(PivotList(clinVals[0])).shape)
#     print(np.array(clinVals[1]).shape)
#     print()
    compVar = list()
    meanCompVar = list()
    stdCompVar = list()
    for ii in range(Nsubpops):
        compVar.append([])
        meanCompVar.append([])
        stdCompVar.append([])
        for jj in range(numDrugs):
            curAUC = PivotList(clinVals[ii])[jj*2] 
            curIc50 = PivotList(clinVals[ii])[jj*2 + 1]
            compVar[ii].append(np.divide(curIc50,curAUC)) 
            meanCompVar[ii].append(np.mean(compVar[ii][jj])) 
            stdCompVar[ii].append(np.std(compVar[ii][jj]))


    # compVar, meanCompVar, stdCompVar的原维度是(number of subPopulations, number of drugs)
    # We want the display vars to be organized by 1) subpop, and 2) drug, so pivot
    compVar = PivotList(compVar)
    meanCompVar = PivotList(meanCompVar) # 同上
    stdCompVar = PivotList(stdCompVar) # 同上

    
    # Sort values above so that the most sensitive subpops are first; if there are more

    minMeanCompVar = [ min(xx) for ii,xx in enumerate(PivotList(meanCompVar)) ] 


    sMinMeanCompVar = sorted(minMeanCompVar) #升序排序
    indsSMinMeanCompVar = [ minMeanCompVar.index(xx) for ii,xx in enumerate(sMinMeanCompVar) ]

    # Use indsSMinMeanCompVar as the ordering of the original subplots, based on max sensitivity

    # Determine spacing parameters for numDrugs
    barWidth = 0.90 / float(numDrugs)
    colorStr = ['b','r','g', 'm', 'c', 'k', 'y', 'b', 'r', 'g', 'm']

    # Make figure for different logIc50 responses in each subpop for each drug
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
#     print(np.array(meanCompVar).shape) (2, 17)
    for ii in range(numDrugs):
        curSMeanCompVar = [ meanCompVar[ii][xx] for jj,xx in enumerate(indsSMinMeanCompVar) ] 
        curStdCompVar = [ stdCompVar[ii][xx] for jj,xx in enumerate(indsSMinMeanCompVar) ]
        ax1.bar(np.arange(0.55+ii*barWidth,Nsubpops+0.55), curSMeanCompVar, barWidth, yerr=curStdCompVar, color=colorStr[ii])

    ax1.set_xticks(np.arange(1,Nsubpops+1))
    ax1.set_xticklabels(np.arange(1,Nsubpops+1))
    ax1.set_xlabel('Sub-population Number')
    ax1.set_ylabel('Avg. Composite Sensitivity \n (log(IC50)/AUC)')
    plt.grid()
    plt.savefig(figFolder + expTag + curTarg + '_' + 'CompositeDrugSensitivity')
    # Save composite variables
    f = open(resultsFolder + expTag + curTarg + '_CompositeSensitivityVariables', 'wb')
    pickle.dump(compVar, f)
    pickle.dump(indsSMinMeanCompVar, f)
    f.close()

    ############################
    
    # Iterate thr each subpop graph and extract composition of each subpop
    subpopTissCounts = list()
    for ii in range(Nsubpops):
        if Nsubpops == 1:
            curG = load_graph(graphsFolder + expTag + curTarg + '_' + 'CellLineNetwork.gt')
        else:
            curG = load_graph(graphsFolder + expTag + curTarg + '_' + gName[ii] + 'MultivariateGraph.gt')
        tissProp = curG.vertex_properties['studyLineAbb']
        tissTypes = ExtractListsFromVertices(tissProp,curG)
        subpopTissCounts.append([])
        for jj in range(len(uniqTissTypes)):
            subpopTissCounts[ii].append(tissTypes.count(uniqTissTypes[jj]))

    # Make plots for each subpop that demonstrate the tissue distribution in each,
    # and observing the ordering of subpops based on the most sensitive to the most
    # resistant
    # for ii in indsSMinMeanCompVar:
    for ii in range(len(indsSMinMeanCompVar)):
        # Here ii is the literal index into the subpop data in the order that the subpops were generated
        # sortSubpopNum is the logical index of the subpop, based on its ordering based on the composite variable
        # the latter should be used for "naming" the subpop in the figure titles and images since it corresponds
        # to the ordering of the subpops in the images
        dataSubpopInd = indsSMinMeanCompVar[ii]
        # Open figure
        fig = plt.figure()

        # Make bar plots of tissue distributions with sorted values
        ax1 = fig.add_subplot(211)
        plt.bar(list(range(len(uniqTissTypesCounts))), uniqTissTypesCounts, color='b', label='Original')
        plt.bar(list(range(len(uniqTissTypesCounts))), subpopTissCounts[dataSubpopInd], color='r', label='Subpop '+str(ii+1))
        ax1.set_ylabel('Counts')
        ax1.set_xlabel('Tissue Type')
        ax1.set_xticks(np.arange(0.4,len(uniqTissTypesCounts)+1))
        ax1.set_xticklabels(uniqTissTypes, rotation='vertical')
        ax1.set_title('Distribution of Tissue Types in Subpopulation ' + str(ii+1) + ' and\n Original Cell Line Network (N = ' + str(sum(numVerts)) + ')')
        plt.grid()
        plt.legend(loc=0)

        # Plot boxplots for each covariate used in graph
        ax2 = fig.add_subplot(212)
        plt.boxplot(PivotList(clinVals[ii]))
        ax2.set_xticks(np.arange(1.0,len(allVarsInclLabels)+1))
        ax2.set_xticklabels(allVarsInclLabels, rotation=20, horizontalalignment='center')
        ax2.set_ylabel('Output Values \n(AUC, log(IC50)')
        plt.grid()
        ax2.set_title('Subpop ' + str(ii+1) + ' (N = ' + str(numVerts[dataSubpopInd]) + ')')

        # Format fig and save and then close
        plt.tight_layout(pad=1.0)
        plt.savefig(localResultsFolder + expTag + curTarg + '_' + 'SubpopTissDist' + str(ii+1) + '.png')
#         plt.close()

    # Make figure showing IC50 value for all different subpops
    MakeIc50Subpop([clinVals[xx] for jj,xx in enumerate(indsSMinMeanCompVar)], [numVerts[xx] for jj,xx in enumerate(indsSMinMeanCompVar)])

    return


# In[ ]:


import pickle
def ExportSubpopResults(suffix=[]):
    """

        Method to write POET results to file

    :param suffix:
    :return:
    """

    # Rebuild gList from all the partitions
    # gList, gListClosure = RebuildGList()

    # Write results to file
    f = open(localResultsFolder + expTag + curTarg + '_' + 'SubpopResults', 'rb')
    numVerts = pickle.load(f)
    numVarsUsed = pickle.load(f)
    gName = pickle.load(f)
    varsUsed = pickle.load(f)
    clinVals = pickle.load(f)
    cellLines = pickle.load(f)
    gListClosure = pickle.load(f)
    f.close()

#     print(numVerts)
    # Load data from composite variable analysis; most importantly
    # retrieve the indexed order of most sensitive --> most resistant drug
    f = open(resultsFolder + expTag + curTarg + '_CompositeSensitivityVariables', 'rb')
    pickle.load(f)
    sortSubpopInds = pickle.load(f)
    f.close()

    # Get list of cell lines names in each subpop
    gListInd = [ii for ii,xx in enumerate(gListClosure) if xx == 0]
    numSubpops = len(gListInd)

    # Open up output file and write subpop data
    with open(exportsFolder + expTag + curTarg + '_' + 'ExportedSubpopData' + suffix + '.csv', 'w') as csvfile:
        subpopWriter = csv.writer(csvfile, delimiter=',', dialect='excel')
        # subpopWriter = csv.writer(opFolder + expTag + 'ExportedSubpopData', dialect='excel', delimiter=' ')

        # Write a row for each subpopulation
        # Format is:  subpopNum, Nsubpop, subpopIds
        for ii in sortSubpopInds:
        # for ii in range(numSubpops):
            if numSubpops == 1:
                curG = load_graph(graphsFolder + expTag + curTarg + '_' + 'CellLineNetwork.gt')
            else:
                curG = load_graph(graphsFolder + expTag + curTarg + '_' + gName[ii] + 'MultivariateGraph.gt')

            tissProp = curG.vertex_properties['studyLineAbb']
            tissTypes = ExtractListsFromVertices(tissProp,curG)

            subpopWriter.writerow([str(ii+1)] + [str(numVerts[ii])] + cellLines[ii] + tissTypes)

    # Load composite variables which are organized by numDrugs x numSubpops
    f = open(resultsFolder + expTag + curTarg + '_CompositeSensitivityVariables', 'rb')
    compVar = PivotList(list(pickle.load(f)))
    f.close()
    print(np.array(compVar))
    
    # Export composite sensitivity variable separately
    with open(exportsFolder + expTag + curTarg + '_' + 'ExportedCompositeVariables' + suffix + '.csv', 'w') as csvfile:
        subpopWriter = csv.writer(csvfile, delimiter=',', dialect='excel')

        # Write a row for each subpopulation
        # Format is:  subpopNum, numVerts,  numDrugs, flattened compVars for current subpop
        # for ii in range(len(numVerts)):
        for ii in sortSubpopInds:
            curSubpopStr = []
            for vv in range(numDrugs):
                for ww in range(numVerts[ii]):
                    curSubpopStr = curSubpopStr + [str(compVar[ii][vv][ww])]
            subpopWriter.writerow([str(ii+1)] + [str(numVerts[ii])] + [str(numDrugs)] + curSubpopStr)

    return cellLines


# In[ ]:


def ExportAllSubpopResults(suffix=[]):
    """

        Method to write POET results to file

    :param suffix:
    :return:
    """

    import os
    import glob
    import csv

    with open(exportsFolder + expTag + curTarg + '_' + 'ExportedAllSubpopsData' + suffix + '.csv', 'w') as csvfile:
        subpopWriter = csv.writer(csvfile, delimiter=',', dialect='excel')
        allFiles = sorted(glob.glob(graphsFolder + expTag + curTarg + '*.gt'))

        for ii in range(len(allFiles)):

            if ii == 0:
                head,tail = os.path.split(allFiles[ii])
                slashInds = [ jj for jj,xx in enumerate(tail) if xx == '_' ]
                MInd = [ jj for jj,xx in enumerate(tail) if xx == 'M']
                curGname = 'g'
            else:
                head, tail = os.path.split(allFiles[ii])
                slashInds = [jj for jj, xx in enumerate(tail) if xx == '_']
                MInd = [jj for jj, xx in enumerate(tail) if xx == 'M']
                curGname = tail[slashInds[3]+1:MInd[1]]

            curG = load_graph(allFiles[ii])
            curNumVerts = curG.num_vertices()
            tissProp = curG.vertex_properties['studyLineAbb']
            tissTypes = ExtractListsFromVertices(tissProp, curG)
            curCellLineIdProp = curG.vertex_properties['idProp']
            curCellLineIds = ExtractListsFromVertices(curCellLineIdProp, curG)
            subpopWriter.writerow([str(ii + 1)] + [str(curGname)] + [str(curNumVerts)] + curCellLineIds + tissTypes)
  

    return curCellLineIds

