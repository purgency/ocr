import cv2
import numpy as np

InputA = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[1,0,1,0,0,0],[1,1,0,0,1,0],[1,0,1,0,0,1]])
InputB = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,1,0,1,1,0],[0,0,0,0,1,1],[0,0,1,0,1,1]])       


whid = np.random.uniform(0.0, 1.0, (3,6)) #weights from input neurons to hidden layer neurons
wout = np.random.uniform(0.0, 1.0, (1,3)) #weights from hidden layer neurons to output neurons

bhid = np.random.rand(3)
bout = np.random.rand(1)
learningrate = 0.1
	
#transferfunc = lambda h: np.tanh(h) #tanh
#ableitungtransfunc = lambda h: 1 - np.power(transferfunc(h),2)
transferfunc = lambda h: 1.0/(1.0+np.exp(-h)) #sigmoid
ableitungtransfunc = lambda h: transferfunc(h) * (1.0-transferfunc(h))
#transferfunc = lambda h: np.log(1+np.exp(h)) #relu
#ableitungtransfunc = lambda h: 1/(1 + np.exp(-h))

for x in range(300): #number of training episodes
#----------------------------------Take input, give output (forwardpropagation):
    if(np.random.random() >= 0.5):
        Input = InputA[np.random.randint(0,1)]
        Solution = 0.0
        print("Input is A")
    else:
        Input = InputB[np.random.randint(0,1)]
        Solution = 1.0
        print("Input is B")
    
    h1 = np.dot(whid,Input)+bhid
    shid = transferfunc(h1)
    #print(shid)

    h2 = np.dot(wout,shid)+bout
    #print(h2)
    sout = h2
#----------------------------------Calculate error, learn from error/update weights (backpropagation):
    delta2 = Solution - sout #difference between the correct output and the produced output sout (this is the error)
    #print(delta2)
    delta1 = ableitungtransfunc(h1) * np.dot(delta2, wout)
    #print(delta1)

    wout += learningrate *np.outer(delta2, shid)
    #print(wout)
    whid += learningrate *np.outer(delta1, Input)
    #print(whid)
    bhid += learningrate * delta1
    #print(bhid)
    bout += learningrate * delta2
    #print(bout)	
    print("Solution is " + str(Solution))	
    print('Given Output is ' + str(sout) + ' Error is ' + str(delta2))
    print ('\n')