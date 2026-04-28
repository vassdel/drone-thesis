import numpy as np
import cv2
from transformers import AutoImageProcessor, DetrForObjectDetection
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch.nn as nn
import torch
from ultralytics import RTDETR
import os

ann_lst=os.listdir('test_data/annotations')

model=RTDETR("runs/detect/rt_detr_custom/weights/best.pt")

c=0
sum_time=0
sum_time_ninth=0

for f in ann_lst:
	img_tmp=cv2.imread('test_data/images/'+f[:-4]+'.jpg')
	t1=time.time_ns()//1000000
	results=model.predict(img_tmp,conf=0.1)
	t2=time.time_ns()//1000000
	sum_time+=t2-t1
	f1=open("test_data/detections_ultralytics_rtdetr_custom/"+f,"w")
	scale=1
	result=results[0]
	for b in result.boxes:
		r=b.xyxy
		r00=[r[0][0]*scale,r[0][1]*scale,r[0][2]*scale,r[0][3]*scale]
		r0=[str(i.item()) for i in r00]
		f1.write(",".join(r0)+"\n")
		img_tmp=cv2.rectangle(img_tmp,(int(r00[0])//scale,int(r00[1])//scale),(int(r00[2])//scale,int(r00[3])//scale),(0,255,0),2)
	cv2.imwrite("test_data/img_det_ultralytics_rtdetr_custom/"+f[:-4]+'.jpg',img_tmp)
	f1.close()
	c+=1
	print("Processed image",c)
for f in ann_lst:
	img_tmp=cv2.imread('test_data/images_ninth/'+f[:-4]+'.jpg')
	t1=time.time_ns()//1000000
	results=model.predict(img_tmp,conf=0.1)
	t2=time.time_ns()//1000000
	sum_time_ninth+=t2-t1
	f1=open("test_data/detections_ultralytics_rtdetr_custom_ninth/"+f,"w")
	scale=3
	result=results[0]
	for b in result.boxes:
		r=b.xyxy
		r00=[r[0][0]*scale,r[0][1]*scale,r[0][2]*scale,r[0][3]*scale]
		r0=[str(i.item()) for i in r00]
		f1.write(",".join(r0)+"\n")
		img_tmp=cv2.rectangle(img_tmp,(int(r00[0])//scale,int(r00[1])//scale),(int(r00[2])//scale,int(r00[3])//scale),(0,255,0),2)
	cv2.imwrite("test_data/img_det_ultralytics_rtdetr_custom_ninth/"+f[:-4]+'.jpg',img_tmp)
	f1.close()
	print("Processed image",c)
print("Average inference time:",sum_time/c,"ms")
print("Average inference time (1/9):",sum_time_ninth/c,"ms")
