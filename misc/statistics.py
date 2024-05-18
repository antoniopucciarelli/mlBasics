import numpy as np 

def mean(x):
    xLen = len(x)
    if xLen > 0:
        return sum(x) / xLen
    else:
        return 0 
        
def median(x):
    xLen = len(x)
    
    sortedVec = np.sort(x)
    
    if xLen % 2 == 0:
        medianVal = (sortedVec[int(xLen / 2) - 1] + sortedVec[int(xLen / 2)]) / 2
    else:
        medianVal = sortedVec[int(xLen / 2)]

    return float(medianVal)

def mode(x):

    valueVec = []
    counterVec = []

    for dataChecker in x:
        if dataChecker not in valueVec:
            valueVec.append(dataChecker)
            counter = 0
            for data in x:
                if data == dataChecker:
                    counter = counter + 1
                    
            counterVec.append(counter)

    modeVal = valueVec[np.argmax(counterVec)]

    return modeVal

def stdDev(x, mean):
    stdDev_ = 0
    
    for val in x:
        stdDev_ = stdDev_ + (val - mean)**2

    stdDev_ = np.sqrt(stdDev_ / (len(x) - 1))
    
    return stdDev_

def confidenceLevel(x, std, mean):
    score = 1.96 

    stdError = std / np.sqrt(len(x))

    lowConfLev = mean - score * stdError
    uppConfLev = mean + score * stdError

    return lowConfLev, uppConfLev

def get_statistics(input_list):
    meanVal = mean(input_list)
    medianVal = median(input_list)
    modeVal = mode(input_list)
    stdVal = stdDev(input_list, meanVal) 
    variance = stdVal**2 
    confLev = confidenceLevel(input_list, stdVal, meanVal)
    
    return {
        "mean": meanVal,
        "median": medianVal,
        "mode": modeVal,
        "sample_variance": variance,
        "sample_standard_deviation": stdVal,
        "mean_confidence_interval": list(confLev),
    }
