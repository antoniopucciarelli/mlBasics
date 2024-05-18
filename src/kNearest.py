import math

# Should use the `find_k_nearest_neighbors` function below.
def predict_label(examples, features, k, label_key="is_intrusive"):
    pid = find_k_nearest_neighbors(examples, features, k)

    prob = 0 
    for pidName in pid:
        prob = prob + examples[pidName][label_key]

    prob = prob / len(pid)

    if prob > 0.5:
        return 1
    else:
        return 0
    
def find_k_nearest_neighbors(examples, features, k):
    dict_ = {}
    for pidName in examples.keys():
        exampleFeatures = examples[pidName]['features']

        value = features

        distance = 0 
        for ii in range(len(exampleFeatures)):
            distance = distance + (exampleFeatures[ii] - features[ii])**2

        distance = math.sqrt(distance)
        
        dict_.update({
            pidName: distance
        })

    orderedDict = sorted(dict_.items(), key=lambda x:x[1])

    pid = []
    for ii in range(k):
        keyName = orderedDict[ii]
        pid.append(keyName[0])
    
    return pid
