import cv2
from model import bodypose_model, PoseEstimationWithMobileNet
from util import*
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import argparse

def Net_Prediction(model, image, device, backbone = 'Mobilenet'):
    scale_search = [1]
    stride = 8
    padValue = 128
    heatmap_avg = np.zeros((image.shape[0], image.shape[1], 19))
    paf_avg = np.zeros((image.shape[0], image.shape[1], 38))
    
    for m in range(len(scale_search)):
        scale = scale_search[m]
        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
        
        im = np.transpose(np.float32(imageToTest_padded), (2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)
   
        with torch.no_grad():
            stages_output = model(data)
            _paf = stages_output[-1].cpu().numpy()
            _heatmap = stages_output[-2].cpu().numpy()  
            
        
        heatmap = np.transpose(np.squeeze(_heatmap), (1, 2, 0))  
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)     
        
        paf = np.transpose(np.squeeze(_paf), (1, 2, 0))  
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg += heatmap / len(scale_search)
        paf_avg += paf / len(scale_search)
        
    return heatmap_avg, paf_avg

if __name__ == '__main__':

    scale = 0.34 
    threshold= 0.2 
    device = torch.device("cpu")   
    
    model = PoseEstimationWithMobileNet().to(device)
    model.load_state_dict(torch.load('weights/MobileNet_bodypose_model', map_location=lambda storage, loc: storage))
    
    model.eval()
    print('model is successfully loaded...')
    
    test_image = 'images/test.jpg'
    image = cv2.imread(test_image)
    imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    since = time.time()
    
    heatmap, paf = Net_Prediction(model, imageToTest, device, backbone = 'Mobilenet')
    
    t1 = time.time()
    print("model inference in {:2.3f} seconds".format(t1 - since))
    
    all_peaks = peaks(heatmap, threshold)
    
    t2 = time.time()
    print("find peaks in {:2.3f} seconds".format(t2 - t1))
    
    connection_all, special_k = connection(all_peaks, paf, imageToTest)
        
    t2 = time.time()
    print("find connections in {:2.3f} seconds".format(t2 - t1))
        
    candidate, subset = merge(all_peaks, connection_all, special_k)
            
    t3 = time.time()
    print("merge in {:2.3f} seconds".format(t3 - t2))
        
    canvas = drawpose(image, candidate, subset, scale)
            
    print("total inference in {:2.3f} seconds".format(time.time() - since))
    
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()