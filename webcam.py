import cv2
from model import PoseEstimationWithMobileNet
from util import*
import torch


scale = 0.34 
threshold= 0.2 
device = torch.device("cpu")   
    
model = PoseEstimationWithMobileNet().to(device)
model.load_state_dict(torch.load('weights/MobileNet_bodypose_model.pth', map_location=lambda storage, loc: storage))
    
model.eval()

vid = cv2.VideoCapture(0)

while(True):
        
    ret, frame = vid.read()
    imageToTest = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    heatmap, paf = Net_Prediction(model, imageToTest, device, backbone = 'Mobilenet')
    all_peaks = peaks(heatmap, threshold)
    connection_all, special_k = connection(all_peaks, paf, imageToTest)
    candidate, subset = merge(all_peaks, connection_all, special_k)
    canvas = drawpose(frame, candidate, subset, scale)   
    cv2.imshow('BDG', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()