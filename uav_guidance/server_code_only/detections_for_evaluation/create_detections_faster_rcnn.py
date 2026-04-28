import numpy as np
import cv2
from transformers import AutoImageProcessor, DetrForObjectDetection
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch.nn as nn
import torch
from ultralytics import YOLO
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image

ann_lst=os.listdir('test_data/annotations')

transform = transforms.Compose([transforms.ToTensor(),])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
model=model.eval().to(device)

c=0
sum_time=0
sum_time_ninth=0
thr=0.1

for f in ann_lst:
	img_tmp=Image.open('test_data/images/'+f[:-4]+'.jpg').convert('RGB')
	img_tmp = transform(img_tmp).to(device)
	img_tmp1=cv2.imread('test_data/images/'+f[:-4]+'.jpg')
	img_tmp=img_tmp.unsqueeze(0)
	img_tmp=img_tmp.to(device)
	t1=time.time_ns()//1000000
	with torch.no_grad():
		results=model(img_tmp)
	t2=time.time_ns()//1000000
	sum_time+=t2-t1
	f1=open("test_data/detections_faster_rcnn/"+f,"w")
	scale=1
	result=results[0]["boxes"].detach().cpu().numpy()
	conf=results[0]["scores"].detach().cpu().numpy()
	result_conf=result[conf>=thr]
	for r in result_conf:
		r00=[r[0]*scale,r[1]*scale,r[2]*scale,r[3]*scale]
		r0=[str(i.item()) for i in r00]
		f1.write(",".join(r0)+"\n")
		img_tmp1=cv2.rectangle(img_tmp1,(int(r00[0])//scale,int(r00[1])//scale),(int(r00[2])//scale,int(r00[3])//scale),(0,255,0),2)
	cv2.imwrite("test_data/img_det_faster_rcnn/"+f[:-4]+'.jpg',img_tmp1)
	f1.close()
	c+=1
	print("Processed image",c)
for f in ann_lst:
        img_tmp=Image.open('test_data/images_ninth/'+f[:-4]+'.jpg').convert('RGB')
        img_tmp = transform(img_tmp).to(device)
        img_tmp1=cv2.imread('test_data/images_ninth/'+f[:-4]+'.jpg')
        img_tmp=img_tmp.unsqueeze(0)
        img_tmp=img_tmp.to(device)
        t1=time.time_ns()//1000000
        with torch.no_grad():
                results=model(img_tmp)
        t2=time.time_ns()//1000000
        sum_time_ninth+=t2-t1
        f1=open("test_data/detections_faster_rcnn_ninth/"+f,"w")
        scale=3
        result=results[0]["boxes"].detach().cpu().numpy()
        conf=results[0]["scores"].detach().cpu().numpy()
        result_conf=result[conf>=thr]
        for r in result_conf:
                r00=[r[0]*scale,r[1]*scale,r[2]*scale,r[3]*scale]
                r0=[str(i.item()) for i in r00]
                f1.write(",".join(r0)+"\n")
                img_tmp1=cv2.rectangle(img_tmp1,(int(r00[0])//scale,int(r00[1])//scale),(int(r00[2])//scale,int(r00[3])//scale),(0,255,0),2)
        cv2.imwrite("test_data/img_det_faster_rcnn_ninth/"+f[:-4]+'.jpg',img_tmp1)
        f1.close()
        print("Processed image",c)
print("Average inference time:",sum_time/c,"ms")
print("Average inference time (1/9):",sum_time_ninth/c,"ms")
