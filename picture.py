import cv2
from model import PoseEstimationWithMobileNet
from util import*
import torch
import matplotlib.pyplot as plt


scale = 0.34 
threshold= 0.2 
device = torch.device("cpu")   
    
model = PoseEstimationWithMobileNet().to(device)
model.load_state_dict(torch.load('weights/MobileNet_bodypose_model.pth', map_location=lambda storage, loc: storage))
    
model.eval()
test_image = 'images/test.jpg'
image = cv2.imread(test_image)
imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
heatmap, paf = Net_Prediction(model, imageToTest, device, backbone = 'Mobilenet')
all_peaks = peaks(heatmap, threshold)
connection_all, special_k = connection(all_peaks, paf, imageToTest)
candidate, subset = merge(all_peaks, connection_all, special_k)
canvas = drawpose(image, candidate, subset, scale)
            
plt.imshow(canvas[:, :, [2, 1, 0]])
#plt.axis('off')
plt.show()