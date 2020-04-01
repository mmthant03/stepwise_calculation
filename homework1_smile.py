import numpy as np
from itertools import product
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time

m = 5
DEBUG = False
show = False

def fPC(y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors(predictors, X, y):
    yhat = np.zeros(y.shape)
    for r1,c1,r2,c2 in predictors:
        yhat += X[:,r1,c1] > X[:,r2,c2]
    yhat = yhat/len(predictors) > 0.5
    return fPC(y, yhat)

def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):
    
    start_time = time.time()
    best, loc = 0, None
    preds = [] # predictors
    pixels = [x for x in product(range(0,24), repeat = 4) 
              if (x[0],x[1]) != (x[2],x[3])]
    
    for i in range(m):
        
        if DEBUG:
            print("Current Step : ", i)
            print(f'{round((time.time() - start_time)/60, 3)} minutes elapsed')
            
        for p in pixels:
            if p in preds: continue
            acc = measureAccuracyOfPredictors(preds + [p], X=trainingFaces, y=trainingLabels)            
            best = max(acc, best)
            loc = p if best == acc else loc
            
        if DEBUG:
            print("Best pixels : ", loc)
            print("Best Accuracy : ", best)
        
        preds.append(loc)

    if show:
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        
        for r1,c1,r2,c2 in predictors:
            color = np.random.rand(3,)
            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        plt.show()
 
    return preds, best

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    trainl, testl = [], []

    print('n \t trainAccuracy \t testAccuracy')
    for a in range(1,6):
        predictors, train_acc = stepwiseRegression(trainingFaces[:400*a], trainingLabels[:400*a], testingFaces, testingLabels)
        test_acc = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        trainl.append(train_acc)
        testl.append(test_acc)
        print(f'n \t {round(train_acc,4)} \t {round(test_acc,4)}')

    if show:
        plt.plot(trainl, label="Train")
        plt.plot(testl, label="Test")
        plt.title("Model Accuracy")
        plt.legend()
        plt.show()