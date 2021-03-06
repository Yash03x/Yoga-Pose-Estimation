import numpy as np
import cv2
import torch
import math
from scipy.ndimage.filters import gaussian_filter

def padRightDownCorner(img, stride, padValue):
    
    h = img.shape[0]
    w = img.shape[1]
    pad = 4 * [None]
    pad[0] = 0 
    pad[1] = 0 
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)
    img_padded = np.pad(img, ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), 'constant', constant_values=padValue) 

    return img_padded, pad

def peaks(heatmap, threshold): 
    
    thre1 = threshold    
    all_peaks = []
    peak_counter = 0    
    for part in range(heatmap.shape[2]-1):
        map_ori = heatmap[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=3)
        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]
        peaks_binary = np.logical_and.reduce((one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) 
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
        
    return all_peaks

def connection(all_peaks, paf, image):

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
    thre2 = 0.1 
    connection_all = []
    special_k = []
    mid_num = 10
    for k in range(len(mapIdx)):
        score_mid = paf[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1] 
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k] 
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                     vec = np.subtract(candB[j][:2], candA[i][:2]) 
                     norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                     vec = np.divide(vec, norm) 
                     startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                     vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                     vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])
                     score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1]) 
                     score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * image.shape[0] / (norm+1e-16) - 1, 0)
                     criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                     criterion2 = score_with_dist_prior > 0
                     if criterion1 and criterion2:
                         connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
                        
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break
            
            connection_all.append(connection)
            
        else:
            
            special_k.append(k)
            connection_all.append([])
    
    return connection_all, special_k

def merge(all_peaks, connection_all, special_k):
    
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
    
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1
    
            for i in range(len(connection_all[k])): 
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): 
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
    
                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
    
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    return candidate, subset

def drawpose(canvas, candidate, subset, scale=1):
    stickwidth = 1
    dotwidth = 2
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x/scale), int(y/scale)), int(dotwidth), colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]/scale
            X = candidate[index.astype(int), 1]/scale
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

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
