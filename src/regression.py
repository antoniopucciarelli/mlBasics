import numpy as np 

class TreeNode:
    def __init__(self, examples, printout=False):
        self.examples = examples        
        self.getLabel()
        self.left = None
        self.right = None
        self.split_point = None

        if printout:
            for ii, example in enumerate(examples):
                print('example ', ii, 'bpd = ', example['bpd'])

    def getLabel(self, label='bpd'):
        features = []
        for key in self.examples[0].keys():
            if key != label:
                features.append(key)
        self.features = features 
        self.label = label

    def featuresMSE(self, parentData):
        MSEdict   = {} # stores the mean squared error of a feature
        leftDict  = {} # stores the left dictionary of a feature
        rightDict = {} # stores the right dictionary of a feature
        splitDict = {} # stores the splitting index of a feature

        # setting up storing value for each feature
        for key in self.features:
            MSEdict[key]   = np.Inf
            leftDict[key]  = None 
            rightDict[key] = None
            splitDict[key] = None 

        # studying each feature
        for feature in self.features:
            # sorting feature based on a descendent sorting 
            featureList = sorted(parentData, key=lambda x: x[feature])

            # the sorted feature is divided into bins
            # every bin defines the splitting index for the study variables
            for ii in range(len(featureList) - 1):
                # computing splitting index as the mean between the 2 adiacent data of the study feature
                deltaFeature = (featureList[ii+1][feature] + featureList[ii][feature]) / 2

                # setting up storing variables 
                leftData  = [] # stores the left data after the splitting 
                rightData = [] # stores the right data after the splitting 

                # every vector inside the dataset is split wrt the splitting factor for the study feature
                for jj in range(len(featureList)):
                    # splitting data wrt the study feature
                    if featureList[jj][feature] < deltaFeature:
                        # appending data to the left dictionary list
                        leftData.append(featureList[jj]) 
                    else:
                        # appending data to the right dictionary list 
                        rightData.append(featureList[jj])

                # computing feature properties
                # getting feature properties
                leftFeature = np.array([value[self.label] for value in leftData])
                # computing feature means
                leftMean = sum(leftFeature) / len(leftFeature)
                # computing mean squared error for the defind feature
                leftMSE = sum((leftFeature - leftMean)**2) / len(leftFeature)

                # getting feature properties
                rightFeature = np.array([value[self.label] for value in rightData])
                # computing feature means
                rightMean = sum(rightFeature) / len(rightFeature)
                # computing mean squared error for the defined feature
                rightMSE = sum((rightFeature - rightMean)**2) / len(rightFeature)

                # computing the mean squared error for the whole node
                MSE = (leftMSE * len(leftFeature) + rightMSE * len(rightFeature)) / (len(leftFeature) + len(rightFeature))

                # comparing MSE between the previous computed and the new one
                # if the MSE is lower than the output data is updated
                if MSE < MSEdict[feature]:
                    MSEdict[feature] = MSE 
                    leftDict[feature] = leftData
                    rightDict[feature] = rightData
                    splitDict[feature] = deltaFeature

        return MSEdict, leftDict, rightDict, splitDict
    
    def getBestFeature(self, MSEdict, leftDict, rightDict, splitDict, printout=False):
        # filtering best feature among the whole computed nodes
        bestFeature = min(MSEdict, key=MSEdict.get)

        if printout:
            print('feature = ', bestFeature)
            print('splitIndex = ', splitDict[bestFeature])
            print('MSE = ', MSEdict[bestFeature])
            print('left branch = ', leftDict[bestFeature])
            print('right branch = ', rightDict[bestFeature])

        return bestFeature, MSEdict[bestFeature], leftDict[bestFeature], rightDict[bestFeature], splitDict[bestFeature]
   
    def split(self):
        # setting up splitting operations for the whole dataset till one of the final nodes has just one element inside it
        
        # computing starting node features properties
        MSEdict, leftDict, rightDict, splitDict = self.featuresMSE(self.examples)
        # getting best representative feature
        feature_, MSEdict_, leftDict_, rightDict_, splitDict_ = self.getBestFeature(MSEdict, leftDict, rightDict, splitDict)

        # setting up storing dictionary for the whole nodes
        levelNum = 0
        levels = {
            levelNum: [
                {
                    'id': 0, # node id
                    'leftChild': 0, # child left node (<)
                    'rightChild': 1, # child right node (>=)
                    'feature': feature_, # study feature of the node
                    'splitIndex': splitDict_, # splitting index of the node 
                    'lenNode': len(self.examples), # number of elements in each node
                    'average': None # average of the node -> used only for the last layers
                }
            ]
        }

        # merging left and right dictionary lists into a single list
        nodes = [leftDict_, rightDict_]

        # looping over the maximum number of elements in each node
        while len(nodes) > 0:
            # updating levels
            levelNum = levelNum + 1
            
            # setting up temporary storing variables
            tempNodes = []
            levelDict = []

            # looping over all possible nodes in the study layer
            for ii, node in enumerate(nodes):
                if len(node) == 1:
                    # storing label values into an array
                    nodeLabelVal = np.array([value[self.label] for value in node])
                    # averaging label properties
                    average = sum(nodeLabelVal) / len(node)
                    # appending node properties to a level dictionary list
                    levelDict.append(
                        {
                            'id': ii,
                            'leftChild': None,
                            'rightChild': None,
                            'feature': self.label,
                            'splitIndex': None,
                            'lenNode': len(node),
                            'average': average
                        }
                    )
                else:
                    # computing node properties for each feature
                    MSEdict, leftDict, rightDict, splitDict = self.featuresMSE(node)
                    # getting best representative feature
                    feature_, MSEdict_, leftDict_, rightDict_, splitDict_ = self.getBestFeature(MSEdict, leftDict, rightDict, splitDict)
    
                    # appending data to the storing dictionaries
                    tempNodes.append(leftDict_)
                    tempNodes.append(rightDict_)
    
                    # appending node properties to a level dictionary list
                    levelDict.append(
                        {
                            'id': ii,
                            'leftChild': len(tempNodes)-2,
                            'rightChild': len(tempNodes)-1,
                            'feature': feature_,
                            'splitIndex': splitDict_,
                            'lenNode': len(node),
                            'average': None
                        }
                    )

            # copying nodes properties for new layer study
            nodes = tempNodes.copy()

            # updating level properties
            levels.update(
                {
                    levelNum: levelDict
                }
            )

        # saving level property
        self.levels = levels
                        
class RegressionTree:
    def __init__(self, examples):
        # Don't change the following two lines of code.
        self.root = TreeNode(examples)
        self.train()

    def train(self):
        # Don't edit this line.
        self.root.split()

    def predict(self, example):
        # the first node has 0 ID
        nodeID = 0
        
        # looping all over the nodes
        for levelNum, nodes in self.root.levels.items():

            # getting the best predictive feature of the study node
            studyFeature = nodes[nodeID]['feature']

            # from the splitting feature -> understand if it is the last layer by feature == label
            if studyFeature != self.root.label:                
                # if it is not the last layer
                # compute child node belonging
                if example[studyFeature] < nodes[nodeID]['splitIndex']:            
                    # belongs to the left branch of the study node
                    nodeID = nodes[nodeID]['leftChild']
                else:
                    # belongs to the right branch of the study node
                    nodeID = nodes[nodeID]['rightChild']
            else:
                # if it is the last layer
                # return the average data
                average = np.round(nodes[nodeID]['average'], 4)
                
                return average
