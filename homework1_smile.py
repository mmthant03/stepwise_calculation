import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

m = 5
n = 400 # 400, 800, 1200, 1600, 2000
DEBUG = True

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y):
    yhat = np.zeros(y.shape)
    for r1,c1,r2,c2 in predictors:
	    yhat += X[:,r1,c1] > X[:,r2,c2]
    yhat = yhat/m > 0.5
    return fPC(y, yhat)
    

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    predictors = []
    for i in range(m):
        if DEBUG: print("Current Step : ", i)
        bestAcc = 0
        location = None
        for r1 in range(0,24):
            for c1 in range(0,24):
                for r2 in range(0,24):
                    for c2 in range(0,24):
                        if (r1,c1) == (r2,c2):
                            continue
                        if (r1,c1,r2,c2) in predictors:
                            continue
                        
                        accuracy = measureAccuracyOfPredictors(predictors + list(((r1, c1, r2, c2),)), trainingFaces, trainingLabels)
                        if accuracy > bestAcc: 
                            bestAcc = accuracy
                            location = (r1, c1, r2, c2)
        if DEBUG: 
            print("Best pixels : ", location)
            print("Best Accuracy : ", bestAcc)
        predictors.append(location)
    r1, c1, r2, c2 = location

    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()
    return predictors

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    predictors = stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
    print(predictors)