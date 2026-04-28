import numpy as np
import cv2
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch.nn as nn
import torch
from transformers.image_transforms import center_to_corners_format
import os

def get_boxes_from_detr(img,detr_model,image_processor):
        inputs=image_processor(images=img,return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs.to(device)
        target_sizes=torch.tensor([img.shape[:2]])
        print("Getting bounding boxes and features from detr...")
        out=detr_model(**inputs)
        last_hidden_state=out.last_hidden_state
        pred_boxes=out.pred_boxes
        ############Taken from huggingface post_process_object_detection##################
        pred_boxes=center_to_corners_format(pred_boxes)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(pred_boxes.device)
        pred_boxes = pred_boxes * scale_fct[:, None, :]
        ##################################################################################
#        print("@@@@@@@@@@@@@@@@@@@@@@@")
#        print("lhs shape:",last_hidden_state.shape)
#        print("-----------------------------------")
#        print("pred_boxes shape:",pred_boxes.shape)
#        print("@@@@@@@@@@@@@@@@@@@@@@@")
        res_tmp=image_processor.post_process_object_detection(out, threshold=0.2, target_sizes=target_sizes)[0]
        res_fin=[]
        for score, label, box in zip(res_tmp["scores"], res_tmp["labels"], res_tmp["boxes"]):
                box=box.cpu().detach().numpy()
                score=score.cpu().detach().numpy()
                label=label.cpu().numpy()
                res_fin.append(([box[0],box[1],box[2]-box[0],box[3]-box[1]],score,label))
        feats_and_ltrb_of_boxes=[]
        for i in range(last_hidden_state.shape[1]):
                feats_and_ltrb_of_boxes.append([last_hidden_state[0][i],pred_boxes[0][i]]) #[N_queries x [hidden dim features,ltrb bbox coords]]
 #       print("+++++++++++++++++++++++++++++++")
 #       print(len(feats_and_ltrb_of_boxes))
 #       print(feats_and_ltrb_of_boxes[0][0].shape)
 #       print(feats_and_ltrb_of_boxes[0][1].shape)
 #       print("+++++++++++++++++++++++++++++++")
 #       print(res_fin)
        return res_fin, feats_and_ltrb_of_boxes

ann_lst=os.listdir('test_data/annotations')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model_detr= DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
model_detr.to(device)


c=0
sum_time=0

for f in ann_lst:
	img_tmp=cv2.imread('test_data/images_sixteenth/'+f[:-4]+'.jpg')
	t1=time.time_ns()//1000000
	results,current_detection_features=get_boxes_from_detr(img_tmp,model_detr,image_processor) #results: ltwh
	t2=time.time_ns()//1000000
	sum_time+=t2-t1
	f1=open("test_data/detections_deformable_sixteenth/"+f,"w")
	scale=4
	for r in results:
		r00=[r[0][0]*scale,r[0][1]*scale,r[0][0]*scale+r[0][2]*scale,r[0][1]*scale+r[0][3]*scale]
		r0=[str(i) for i in r00]
		f1.write(",".join(r0)+"\n")
		img_tmp=cv2.rectangle(img_tmp,(int(r00[0])//scale,int(r00[1])//scale),(int(r00[2])//scale,int(r00[3])//scale),(0,255,0),2)
	cv2.imwrite("test_data/img_det_deformable_sixteenth/"+f[:-4]+'.jpg',img_tmp)
	f1.close()
	c+=1
	print("Processed image",c)
print("Average inference time:",sum_time/c,"ms")

