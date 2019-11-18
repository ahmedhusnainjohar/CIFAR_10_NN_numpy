import numpy as np 
import os 


# ############################################################
#                     LOADING THE BATCHs                     #
# ############################################################
def LOAD_DATA(path, batch_num):
    """gets the path string of the parent folder in which all batches live. batch_num is tag of batch to read. For test batch use batch_num = 0.
    
    Parameters
    ----------
    path : string
        path of the main folder in which the batches are present
    batch_num : int
        if 0 --> load test batch otherwise batch 1,2,3,4,5
    
    Returns
    -------
    LABELS : list
        list of integers
    IMAGES : numpy.ndarray
        shape = (10000, 3072)
    IMAGE_NAMES : list of strings
        list of all images names
    """
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    if batch_num == 1:
        N = "data_batch_1"
    elif batch_num == 2:
        N = "data_batch_2"
    elif batch_num == 3:
        N = "data_batch_3"
    elif batch_num == 4:
        N = "data_batch_4"
    elif batch_num == 5:
        N = "data_batch_5"
    elif batch_num == 0:
        N = "test_batch"
        

    batch_path = os.path.join(path,N)

    # Data_loaded = unpickle(file=r"C:\Users\Ahmed Husnain Johar\Downloads\Compressed\New folder (2)\cifar-10-batches-py\data_batch_1")

    Data_loaded = unpickle(file=batch_path)



    KEYS = []
    for i in Data_loaded:
        KEYS.append(i)

    LABELS = Data_loaded[KEYS[1]]
    IMAGES = Data_loaded[KEYS[2]]
    IMAGE_NAMES = Data_loaded[KEYS[3]]



    return LABELS, IMAGES, IMAGE_NAMES



def cvt_to_one_hot_encoding(nC, LABELS):
    """takes as input the no. of classes and list of LABELS and converts them to one-hot encoded form
    
    Parameters
    ----------
    nC : int
        no. of classes i.e. 10 in case of CIFAr-10
    LABELS : list of ints
        list of labels
    
    Returns
    -------
    encoded : numpy.ndarray
        shape = (nC, len(LABELS)) where len(LABELS) ---> no. of examples
    """
    encoded = np.eye(nC)[LABELS]
    return encoded.T



def pre_process(images):
    """normalize the image pixel values between 0 -- 1
    
    Parameters
    ----------
    images : numpy.ndarray
        images batch
    
    Returns
    -------
    processed_imgs : numpy.ndarray
        normalized images. No change in shape 
    """
    processed_imgs = images/255.0
    return processed_imgs.T




def initialize_weights(n_L, ndim_in):
    """initialize weights automatically
    
    Parameters
    ----------
    n_L : list of ints
        no. of neurons in each layer
    ndim_in : int
        no. of pixels in each image example i.e. 3072
    
    Returns
    -------
    weights : dict
        keys : "W1", "W2" ..... (automatically generated)
    
    """
    weights = {}

    for i in range(len(n_L)):
        if i == 0:
            weights["W"+str(i+1)] = np.random.randn(n_L[i], ndim_in)
            # weights["B"+str(i+1)] = np.random.randn(n_L[i], ndim_in)
        elif i != 0:
            weights["W"+str(i+1)] = np.random.randn(n_L[i], n_L[i-1])
            # weights["B"+str(i+1)] = np.random.randn(n_L[i], n_L[i-1])
    return weights




def sigmoid(x):
    #  (nL1, m)
    return 1. / (1. + np.exp(-x))



def fwd_propagate(weights, images):
    """fwd_propagate
    
    Parameters
    ----------
    weights : dict
        keys : "W1", "W2" ....
    images : numpy.ndarray
        images batch shape = (3072, m) where m -- > no. of examples
    """
    KEYS = list(weights.keys()) # get keys 
    Z = {}
    A = {}

    for i in range(len(KEYS)):
        if i == 0:
            Z["Z"+str(i+1)] = np.dot(weights[KEYS[i]] , images)
            # print(Z["Z"+str(i)].shape)
            A["A"+str(i+1)] = sigmoid(Z["Z"+str(i+1)])
            # print(A["A"+str(i)].shape)
        elif i != 0:
            Z["Z"+str(i+1)] = np.dot(weights[KEYS[i]] , A["A"+str(i)])
            # print(Z["Z"+str(i)].shape)
            A["A"+str(i+1)] = sigmoid(Z["Z"+str(i+1)])
            # print(A["A"+str(i)].shape)

    return Z, A



def loss(Y_act, Y_pred):
    # nc, m
    m = Y_act.shape[1]
    loss = np.sum(-Y_act*np.log(Y_pred)-(1- Y_act)*np.log(1-Y_pred)) / m
    return loss


labels, images, _ = LOAD_DATA(path=r"C:\Users\Ahmed Husnain Johar\Downloads\Compressed\New folder (2)\cifar-10-batches-py", batch_num=1)


labels_enc = cvt_to_one_hot_encoding(nC=10, LABELS=labels)

imgs = pre_process(images=images)


WEIGHTS = initialize_weights(n_L=[50, 100, 10], ndim_in=3072)

# for k,v in WEIGHTS.items():
#     print(k, v.shape)

scores, activ = fwd_propagate(weights=WEIGHTS, images=imgs)

predictions = activ[ list(activ.keys())[-1] ]


l_ = loss(Y_act=labels_enc, Y_pred=predictions)

print(l_)