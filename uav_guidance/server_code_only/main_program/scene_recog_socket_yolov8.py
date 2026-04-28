import socket
import numpy as np
import cv2
from transformers import AutoImageProcessor, DetrForObjectDetection
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch.nn as nn
import torch
from transformers.image_transforms import center_to_corners_format
from termcolor import colored
from ultralytics import YOLO,RTDETR
from threading import Thread
import select

# import zstandard

def int_to_int8_lst(num):
        r_lst=[]
        while num//255>=0 and num>0:
                r_lst.append(num%255)
                num=num//255
        return r_lst

def int_8_lst_to_int(lst):
        num=0
        for i in range(len(lst)):
                num+=lst[i]*255**i
        return num

class tr_pred_lstm(nn.Module):
	def __init__(self,out_steps):
		super().__init__()
		self.lstm=nn.LSTM(input_size=10,hidden_size=64,batch_first=True)
		self.linear=nn.Linear(64,out_steps*2) #predict m steps of the trajectory ([x1,y1,..,xm,ym])
	def forward(self,x):
		x,_=self.lstm(x)
		x=self.linear(x)
		return x

def detect_up(x,y,det_lst):
	min_dist=100000000
	min_track=[0,0] #neighbor x,y
	for v1 in det_lst:
		xv1=v1[0][0]+0.5*v1[0][2] #x center of potential neighbor
		yv1=v1[0][1]+0.5*v1[0][3] #y center of potential neighbor
		if yv1-y<=-xv1+x and yv1-y<=xv1-x and (xv1,yv1)!=(x,y):
			dist=(yv1-y)**2+(xv1-x)**2
			if dist<min_dist:
				min_dist=dist
				min_track=[xv1,yv1]
	return min_track

def detect_down(x,y,det_lst):
        min_dist=100000000
        min_track=[0,0]
        for v1 in det_lst:
                xv1=v1[0][0]+0.5*v1[0][2] #x center of potential neighbor
                yv1=v1[0][1]+0.5*v1[0][3] #y center of potential neighbor
                if yv1-y>=-xv1+x and yv1-y>=xv1-x and (xv1,yv1)!=(x,y):
                        dist=(yv1-y)**2+(xv1-x)**2
                        if dist<min_dist:
                                min_dist=dist
                                min_track=[xv1,yv1]
        return min_track

def detect_left(x,y,det_lst):
        min_dist=100000000
        min_track=[0,0]
        for v1 in det_lst:
                xv1=v1[0][0]+0.5*v1[0][2] #x center of potential neighbor
                yv1=v1[0][1]+0.5*v1[0][3] #y center of potential neighbor
                if yv1-y<-xv1+x and yv1-y>xv1-x:
                        dist=(yv1-y)**2+(xv1-x)**2
                        if dist<min_dist:
                                min_dist=dist
                                min_track=[xv1,yv1]
        return min_track

def detect_right(x,y,det_lst):
        min_dist=100000000
        min_track=[0,0]
        for v1 in det_lst:
                xv1=v1[0][0]+0.5*v1[0][2] #x center of potential neighbor
                yv1=v1[0][1]+0.5*v1[0][3] #y center of potential neighbor
                if yv1-y>-xv1+x and yv1-y<xv1-x:
                        dist=(yv1-y)**2+(xv1-x)**2
                        if dist<min_dist:
                                min_dist=dist
                                min_track=[xv1,yv1]
        return min_track

def scale_data(dt,min,max):
	if max!=min:
		return (dt-min)/(max-min)
	else:
		return dt-min

def unscale_data(dt,min,max):
	return (max-min)*dt+min

def check_pwd(conn):
	recv_data_tot=b''
	pwd="tmp_pwd"
	t_init=time.time()
	while 1:
		read_ready,_,_=select.select([conn],[],[],0)
		if read_ready:
			recv_data=conn.recv(512)
			if(recv_data==b''):
				return 0
			recv_data_tot+=recv_data
			if b'end*end*end' in recv_data_tot:
				recv_data_lst=recv_data_tot.split(b'end*end*end')[:-1]
				if recv_data_lst[0]==b'tmp_pwd':
					print("Pwd ok")
					conn.sendall(b'okend*end*end')
					return 1
				else:
					return 0
		if time.time()-t_init>30:
			return 0

def det_transmission_and_target_selection():
	global t_select_run
	global rec_opt
	global g_current_frame
	global interesting_track_id
	global def_shape
	while t_select_run:
		time.sleep(0.5)
		try:
			rec_opt=0
			serv_port=8081
			s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			s.bind(('',serv_port))
			s.listen(5)
			print("Listening on port 8081")
			while (t_select_run):
				time.sleep(0.05)
				accept_ready,_,_=select.select([s],[],[],1)
				if accept_ready:
					conn,addr=s.accept()
					pwd_ok=check_pwd(conn)
					skip=0
					if pwd_ok==0:
						conn.close()
						skip=1
					else:
						print(addr,"connected")
					recv_data_tot=b''
					while (t_select_run and skip!=1):
						if rec_opt==0:
							img_r1=cv2.imencode('.jpg',g_current_frame)[1]
							a=img_r1.astype(np.uint8).tobytes()+b'end*end*end'
							conn.sendall(a)
						else:
							conn.sendall(("Server received "+str(rec_opt)+"end").encode('utf-8'))
							if rec_opt>0:
								interesting_track_id=rec_opt
							else:
								interesting_track_id=-1
						br_ind=0
						while t_select_run:
							time.sleep(0.005)
							read_ready,_,_=select.select([conn],[],[],0)
							if read_ready:
								recv_data=conn.recv(1024)
								if(recv_data==b''):
									br_ind=1
									break
								recv_data_tot+=recv_data
								if b'end*end*end' in recv_data_tot:
									tmp=b''
									recv_data_lst=recv_data_tot.split(b'end*end*end')[:-1]
									if recv_data_tot.split(b'end*end*end')[-1]!=b'':
										tmp=recv_data_tot.split(b'end*end*end')[-1]
									rec_opt=np.frombuffer(recv_data_lst[-1],dtype=int)[0]
									print(rec_opt)
									print("-----------------------------")
									recv_data_tot=tmp
									break
						if br_ind==1:
							break
					conn.close()
			s.close()
		except Exception as e:
			print("There was an error:",e)
			try:
				conn.close()
			except Exception as e:
				print("There was an error closing conn (det_transmission_and_target_selection):",e)
			try:
				s.close()
			except Exception as e:
				print("There was an error closing s (det_transmission_and_target_selection):",e)

def get_boxes_feats_yolov8(img, model):
	r,f=model(img, conf=0.7, return_class_logits=True)
	f_lst=[]
	res=[]
	i=0
	for b in r.boxes:
		r=b.xyxy
		r00=[r[0][0].item(),r[0][1].item(),r[0][2].item(),r[0][3].item()]
		r1=[r[0][0].item(),r[0][1].item(),r[0][2].item()-r[0][0].item(),r[0][3].item()-r[0][1].item()]
		conf=b.conf
		cls=b.cls
		res.append((r1,conf.item(),cls.item()))
		f_lst.append([f[0][i],torch.tensor(r00)])
		i+=1
	for i in range(len(f_lst)):
		f_lst[i][0]=f_lst[i][0][8:]
		mask=f_lst[i][0]>torch.topk(f_lst[i][0], 2).values[1].item()*1.5
		scale=(torch.topk(f_lst[i][0], 2).values[1].item()/torch.topk(f_lst[i][0], 2).values[0].item())*1.5-1 #max<=1.5*2nd_max
		mask=mask*scale
		mask+=1
		f_lst[i][0]=f_lst[i][0]*mask
	return res, f_lst

def iou_ltrb(ltrb1,ltrb2):
	xi_lt=max(ltrb1[0],ltrb2[0])
	yi_lt=max(ltrb1[1],ltrb2[1])
	xi_rb=min(ltrb1[2],ltrb2[2])
	yi_rb=min(ltrb1[3],ltrb2[3])
	if xi_lt>xi_rb or yi_lt>yi_rb:
		return 0
	i=(xi_rb-xi_lt)*(yi_rb-yi_lt)
	u=(ltrb1[2]-ltrb1[0])*(ltrb1[3]-ltrb1[1])+(ltrb2[2]-ltrb2[0])*(ltrb2[3]-ltrb2[1])-i
	return i/u

def get_feats_of_iou_closest(feats_pos, xc, yc, w, h):
	iou_max=0
	d_sq_min=100000000
	best_fit=[]
	for i in range(len(feats_pos)):
		curr_iou=iou_ltrb([xc-w/2, yc-h/2, xc+w/2, yc+h/2],feats_pos[i][1])
		if curr_iou>iou_max:
			iou_max=curr_iou
			best_fit=feats_pos[i]
		if iou_max==0:
			if (xc-(feats_pos[i][1][2]+feats_pos[i][1][0])/2)**2+(yc-(feats_pos[i][1][3]+feats_pos[i][1][1])/2)**2<d_sq_min:
				d_sq_min=(xc-(feats_pos[i][1][2]+feats_pos[i][1][0])/2)**2+(yc-(feats_pos[i][1][3]+feats_pos[i][1][1])/2)**2
				best_fit=feats_pos[i]
	return best_fit,iou_max

def create_similarity_map(previous_f, curr_f, track_locs_new, track_locs_old, tmp_counter):
	# previous_f: all features and locations from previous frame->[f_b_0, f_b_1,.. f_b_num_q-1], f_b_i=[hidden dim features,ltrb bbox coords]
	# curr_f: similar, for current frame
	# track_locs: lst of track locations-ith position represents the ith desired track's position in ltwh format, if -1 we do not have data
	#
	# Returns np.array of shape len(track_locs)xlen(track_locs), ret[i][j] represents the similarity of the ith current interesting track with
	# the jth interesting track at the initial frame.
	ret=[]
	for i in range(len(track_locs_new)):
		no_data_flag=0
		sim_i=[]
		if track_locs_new[i]==-1:
			no_data_flag=1
		else:
			xc=track_locs_new[i][0]+0.5*track_locs_new[i][2]
			yc=track_locs_new[i][1]+0.5*track_locs_new[i][3]
			curr_i_th_feats,_=get_feats_of_iou_closest(curr_f,xc,yc,track_locs_new[i][2],track_locs_new[i][3]) # !!!! Applying a threshold to filter non-detected tracks could improve results
			#print(i,": ",xc,yc,track_locs_new[i][2],track_locs_new[i][3])
			#print("-------------------------")
			t=np.array(curr_i_th_feats[0].cpu().detach())
		for j in range(len(track_locs_old)):
			if no_data_flag==1 or track_locs_new[j]==-1:
				sim_i.append(0)
			else:
				tmp_counter[i][j]+=1
				xcj=track_locs_old[j][0]+0.5*track_locs_old[j][2]
				ycj=track_locs_old[j][1]+0.5*track_locs_old[j][3]
				old_j_feats,_=get_feats_of_iou_closest(previous_f,xcj,ycj,track_locs_old[j][2],track_locs_old[j][3]) # !!!! Applying a threshold to filter non-detected tracks could improve results
				t1=np.array(old_j_feats[0].cpu().detach())
				sim_i.append(np.dot(t,t1)/(np.linalg.norm(t)*np.linalg.norm(t1)))
		ret.append(sim_i)
	return np.array(ret)

def relocate_target(current_feat_and_pos, last_t_feats, tr_predictions, n_steps_ago, w, h, x_offset, y_offset):
#	for i in range(len(tr_predictions)):
#		if i%2==0:
#			tr_predictions[i]-=x_offset
#		else:
#			tr_predictions[i]-=y_offset
	sxx=[57.5, 173.1, 359.5, 621.9, 965.7, 1422.8]
	syy=[15.4, 60.7, 136.8, 240.0, 371.6, 538.6]
	sxy=[4.0, 12.6, 25.1, 39.8, 56.0, 67.8]
	if n_steps_ago<=6:
		s_idx=n_steps_ago-1
	else:
		s_idx=5
	curr_sx=np.sqrt(sxx[s_idx])
	curr_sy=np.sqrt(syy[s_idx])
	curr_r=sxy[s_idx]/(curr_sx*curr_sy)
	if n_steps_ago<=6:
		estimated_current_position=[tr_predictions[2*n_steps_ago-2],tr_predictions[2*n_steps_ago-1]] #[x_est, y_est] from the trajectory prediction (nth step - max 6)
	else:
		n_ahead=n_steps_ago-6
		x_step=tr_predictions[10]-tr_predictions[8]
		y_step=tr_predictions[11]-tr_predictions[9]
		x_e=tr_predictions[10]+n_ahead*x_step
		y_e=tr_predictions[11]+n_ahead*y_step
		estimated_current_position=[x_e,y_e]
	max_sim=0
	best_fit=[]
	print("Relocating Target...")
	for i in range(len(current_feat_and_pos)):
		t=np.array(current_feat_and_pos[i][0].cpu().detach())
		t1=np.array(last_t_feats[0].cpu().detach())
		cos_sim=np.dot(t,t1)/(np.linalg.norm(t)*np.linalg.norm(t1)) #cosine similariy between feature vectors
		x_dist_from_est=(current_feat_and_pos[i][1][2]+current_feat_and_pos[i][1][0])/2-estimated_current_position[0]
		y_dist_from_est=(current_feat_and_pos[i][1][3]+current_feat_and_pos[i][1][1])/2-estimated_current_position[1]
		x_dist_from_est=x_dist_from_est.cpu().detach()
		y_dist_from_est=y_dist_from_est.cpu().detach()
		p_targ=1/(2*np.pi*curr_sx*curr_sy*np.sqrt(1-curr_r**2))*np.exp(-1/(2*(1-curr_r**2))*((x_dist_from_est/curr_sx)**2-2*curr_r*(x_dist_from_est/curr_sx)*(y_dist_from_est/curr_sy)+(y_dist_from_est/curr_sy)**2))
		last_t_feats[1]=last_t_feats[1].cpu().detach()
		dim_sim=(abs(w-(last_t_feats[1][2]-last_t_feats[1][0]))/max(w,(last_t_feats[1][2]-last_t_feats[1][0]))+abs(h-(last_t_feats[1][3]-last_t_feats[1][1]))/max(h,(last_t_feats[1][3]-last_t_feats[1][1])))/2
		tot_sim=0.25*cos_sim+0.6*p_targ+0.15*dim_sim
#		print(tr_predictions)
#		print(x_dist_from_est)
#		print(y_dist_from_est)
#		print(tot_sim,"---",max_sim)
		if tot_sim>max_sim:
			max_sim=tot_sim
			best_fit=current_feat_and_pos[i]
#	for i in range(len(tr_predictions)):
#		if i%2==0:
#			tr_predictions[i]+=x_offset
#		else:
#			tr_predictions[i]+=y_offset
	return (best_fit,max_sim)

def lin_interp_older_hist(hist,hist_timestamps,avg_timestep):
	arr=np.array(hist,dtype=float)
	arr_hist_timestamps=np.array(hist_timestamps,dtype=float)
	i=arr.shape[0]-1
	time_head=arr_hist_timestamps[0]
	while i>0:
		while_flag=0
		if arr_hist_timestamps[i]==arr_hist_timestamps[i-1]:
			arr_hist_timestamps[i]+=1
		while arr_hist_timestamps[i]-arr_hist_timestamps[i-1]<avg_timestep:
			while_flag=1
			step_as_fraction_of_t_diff=avg_timestep/(arr_hist_timestamps[i]-arr_hist_timestamps[i-1])
			last_deleted_timestamp=arr_hist_timestamps[i-1]
			last_deleted_step=arr[i-1]
			arr_hist_timestamps=np.delete(arr_hist_timestamps,i-1)	# if detection steps<avg_timestep, delete intermediate detection steps
			arr=np.delete(arr,i-1,axis=0)
			i-=1 #i is decreased in order to point at the same element since element i-1 was deleted
			if i==0:
				arr_hist_timestamps=np.insert(arr_hist_timestamps,i,arr_hist_timestamps[i]-avg_timestep)
				arr=np.insert(arr,i,arr[i]-step_as_fraction_of_t_diff*(arr[i]-last_deleted_step),axis=0) #####!!!!!!x
				while_flag=0
				break
		if while_flag==1:
			uncovered_step_as_fraction_of_new_t_diff=(avg_timestep-(arr_hist_timestamps[i]-last_deleted_timestamp))/(last_deleted_timestamp-arr_hist_timestamps[i-1])
			arr_hist_timestamps=np.insert(arr_hist_timestamps,i,arr_hist_timestamps[i]-avg_timestep)
			arr=np.insert(arr,i,last_deleted_step-uncovered_step_as_fraction_of_new_t_diff*(last_deleted_step-arr[i-1]),axis=0)
		if arr_hist_timestamps[i]-arr_hist_timestamps[i-1]>avg_timestep:
			step_as_fraction_of_t_diff=avg_timestep/(arr_hist_timestamps[i]-arr_hist_timestamps[i-1]) # step as fraction of the time difference at this point,
			arr_hist_timestamps=np.insert(arr_hist_timestamps,i,arr_hist_timestamps[i]-avg_timestep)  # used to keep the same fraction of track movement
			arr=np.insert(arr,i,arr[i]-step_as_fraction_of_t_diff*(arr[i]-arr[i-1]),axis=0)
			# no decrement of i, we have 1 more step
		else:
			i-=1
	if arr.shape[0]>10:
		arr=arr[-10:]
		arr_hist_timestamps=arr_hist_timestamps[-10:]
	avg_old_step=arr[1]-arr[0]
	if arr.shape[0]>=3:
		avg_old_step+=arr[2]-arr[1]
		avg_old_step=avg_old_step/2
	oldest_pos=arr[0] #oldest data line in sequence
	for i in range(10-len(hist)):
		arr=np.insert(arr,0,oldest_pos-(i+1)*avg_old_step,axis=0) # might create negative values
		arr_hist_timestamps=np.insert(arr_hist_timestamps,0,-1)
	return arr

def find_neighbors_from_detections(detections,interest_point_x,interest_point_y):
	neigh_up=detect_up(interest_point_x,interest_point_y,detections)
	neigh_down=detect_down(interest_point_x,interest_point_y,detections)
	neigh_right=detect_right(interest_point_x,interest_point_y,detections)
	neigh_left=detect_left(interest_point_x,interest_point_y,detections)
	one_step_data_line=[neigh_up[0],neigh_up[1],neigh_left[0],neigh_left[1],interest_point_x,interest_point_y,neigh_right[0],neigh_right[1],neigh_down[0],neigh_down[1]]
	return one_step_data_line

while 1:
	time.sleep(0.5)
	try:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model=YOLO("runs/detect/yolov8x_custom7/weights/best.pt")
		serv_port=8080
		def_shape=(480,640,3) #(1080,1920,3)

		rec_opt=0
		g_current_frame=np.array([[[0,0,0]]])
		interesting_track_id=-1
		t_select_run=1
		t_select_thread=Thread(target=det_transmission_and_target_selection, daemon=True)
		t_select_thread.start()

		s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('',serv_port))
		s.listen(5)

		while (1):
			try:
				rec_opt=0
				g_current_frame=np.array([[[0,0,0]]])
				target_x=-800
				target_y=-800
				tracker = DeepSort(max_age=3,n_init=2)
				interesting_track_id=-1

				skip=0
				conn,addr=s.accept()
				pwd_ok=check_pwd(conn)
				if pwd_ok==0:
					conn.close()
					print("unknown connection from",addr[0],"closed")
					skip=1
				else:
					print(addr,"connected")
				if skip!=1:
					track_predict_model=tr_pred_lstm(6) #model to predict 6 steps ahead using the last 10 steps
					track_predict_model.load_state_dict(torch.load('model_epoch_98.pt'))
					track_predict_model.eval()
					track_predict_model.to(device)

					tr_pred=[] #track predictions, initially empty
					interesting_history=[] #track_history of max 10 steps in the form required for trajectory prediction
					hist_times=[]
					prev_time=0
					sum_timestep=0 #in ms
					timestep_cnt=0
					last_seen=0 # timestamp when interesting track was seen
					recv_data_tot=b''

					frame_id=-1    		#Used for similarity map testing
					init_feats=[] 		#Used for similarity map testing
					id_lst=[]     		#Used for similarity map testing
					count_lst=[]  		#Used for similarity map testing
					init_locs=[]  		#Used for similarity map testing
					sim_matrix_lst=[] 	#Used for similarity map testing
					sim_stop_at_frame=13	#Used for similarity map testing

					while (1):
						recv_data=b''
						recv_data=conn.recv(20000)
						recv_data_tot+=recv_data
						if(recv_data==b''):
							break
						if rec_opt==-1:
							tr_pred=[] #track predictions, initially empty
							interesting_history=[] #track_history of max 10 steps in the form required for trajectory prediction
							hist_times=[]
							prev_time=0
							sum_timestep=0 #in ms
							timestep_cnt=0
							last_seen=0 # timestamp when interesting track was seen

							frame_id=-1             #Used for similarity map testing
							init_feats=[]           #Used for similarity map testing
							id_lst=[]               #Used for similarity map testing
							count_lst=[]            #Used for similarity map testing
							init_locs=[]            #Used for similarity map testing
							sim_matrix_lst=[]       #Used for similarity map testing
							sim_stop_at_frame=13    #Used for similarity map testing

							target_x=-800
							target_y=-800

						if b'end*end*end' in recv_data_tot:
							tmp=b''
							recv_data_lst=recv_data_tot.split(b'end*end*end')[:-1]
							if recv_data_tot.split(b'end*end*end')[-1]!=b'':
								tmp=recv_data_tot.split(b'end*end*end')[-1]
							for r in recv_data_lst:
								print("---------------------------------------------------------------------------")
								img_frame=np.ones(def_shape)
								frame_id+=1
								t_init=time.time_ns()//1000000
								recv_data_tot=np.frombuffer(r,dtype=np.uint8)
								len0=recv_data_tot[-4].astype(int)								#
								len1=recv_data_tot[-3].astype(int)                                      			#
								len2=recv_data_tot[-2].astype(int)                                      			#
								len3=recv_data_tot[-1].astype(int)								#
								print(len0, len1,len2,len3)									#
								recv_data_tot=recv_data_tot[:-4]								#
								timestamp=int_8_lst_to_int(recv_data_tot[-len0-len1-len2-len3:-len1-len2-len3].tolist())	# Getting frame info from end of array
								y_top=int_8_lst_to_int(recv_data_tot[-len1-len2-len3:-len2-len3].tolist())			#
								x_left=int_8_lst_to_int(recv_data_tot[-len2-len3:-len3].tolist())				#
								scaling_factor=int_8_lst_to_int(recv_data_tot[-len3:].tolist())					#
								recv_data_tot=recv_data_tot[:-len0-len1-len2-len3]						#

								if prev_time!=0:
									timestep_cnt+=1
									sum_timestep+=timestamp-prev_time
								prev_time=timestamp
								recv_data_tot=cv2.imdecode(recv_data_tot, cv2.IMREAD_COLOR)
								print("+++Received frame",4*frame_id,"with shape",recv_data_tot.shape,"and timestamp",timestamp)

								if scaling_factor!=1: # If no target is selected, the image is transmitted whole and resized. It needs to be reshaped back to original shape.
									recv_data_tot=cv2.resize(recv_data_tot,(recv_data_tot.shape[1]*scaling_factor, recv_data_tot.shape[0]*scaling_factor))

								img_frame[y_top:y_top+recv_data_tot.shape[0],x_left:x_left+recv_data_tot.shape[1],:]=recv_data_tot
								recv_data_tot=img_frame

								t_i_detr=time.time_ns()//1000000
								results,current_detection_features=get_boxes_feats_yolov8(recv_data_tot,model)
								t_f_detr=time.time_ns()//1000000
								img_r=recv_data_tot.copy()
			#					print("++++++++++++++++++++++++++")
			#					print(results)
			#					print("++++++++++++++++++++++++++")
								print("Found",len(results),"detections, ready to update tracks")
								tracking_time_init=time.time_ns()//1000000
								tracks = tracker.update_tracks(results,recv_data_tot)
								tracking_time_fin=time.time_ns()//1000000
								t_i_lstm=0
								t_f_lstm=0
								new_locs=[]  #Used for similarity map testing
								new_feats=[] #Used for similarity map testing
								#################################################
			#					if frame_id==3:					#
			#						init_feats=current_detection_features	#
			#					if frame_id>3 and frame_id<sim_stop_at_frame:	#
			#						new_feats=current_detection_features	#Used for similarity map testing
			#						for i in range(len(id_lst)):		#
			#							new_locs.append(-1)		#
								#################################################
								total_tracks=0
								tracks_seen=[]
								reloc=1
								for track in tracks:
									if rec_opt==-1:
										break
									if not track.is_confirmed():
										continue
									total_tracks+=1
									track_id = track.track_id
									tracks_seen.append(track_id)
									#################################################################
			#						if frame_id==3:							#
			#							id_lst.append(track_id)					#
			#							init_locs.append((track.to_ltwh()).tolist())		#
			#							count_lst.append(0)					#
			#						if frame_id>3 and frame_id<sim_stop_at_frame:			#Used for similarity map testing
			#							if track_id in id_lst:					#
			#								idx=id_lst.index(track_id)			#
			#								new_locs[idx]=(track.to_ltwh()).tolist()	#
			#								count_lst[idx]+=1				#
									#################################################################
									if int(track_id)==interesting_track_id:
										reloc=0
										print("[Found target track id]", track_id)
										track_ltwh=track.to_ltwh()
										target_x=track_ltwh[0]+0.5*track_ltwh[2]
										target_y=track_ltwh[1]+0.5*track_ltwh[3]
										#print(type(track_ltwh))
										t_i_lstm=time.time_ns()//1000000
										last_w=track_ltwh[2]
										last_h=track_ltwh[3]
										last_target_features_tmp,match_iou=get_feats_of_iou_closest(current_detection_features,track_ltwh[0]+0.5*track_ltwh[2],track_ltwh[1]+0.5*track_ltwh[3],track_ltwh[2],track_ltwh[3])
										if match_iou>=0.5:
											last_seen=timestamp
											last_target_features=last_target_features_tmp
											data_line=find_neighbors_from_detections(results,track_ltwh[0]+0.5*track_ltwh[2],track_ltwh[1]+0.5*track_ltwh[3])

										#	for i in range(len(data_line)): # Add offset for the predictions and the return values
										#		if i%2==0:
										#			data_line[i]+=x_left
										#		else:
										#			data_line[i]+=y_top

											interesting_history.append(data_line)
											hist_times.append(timestamp) #timestamps used to handle uneven detections
											if len(interesting_history)>10:
												interesting_history=interesting_history[1:]
												hist_times=hist_times[1:]
											if len(interesting_history)>1:
												intr_hist_arr=lin_interp_older_hist(interesting_history,hist_times,sum_timestep/timestep_cnt)
												max_h=intr_hist_arr.max()
												min_h=intr_hist_arr.min()
												hist_scaled=scale_data(intr_hist_arr,min_h,max_h)
												tr_pred=track_predict_model(torch.Tensor(hist_scaled).to(device))[-1,:]
												tr_pred=unscale_data(tr_pred,min_h,max_h)
												print("Predicted future track([x1,y1,..,x6,y6]):",[round(n,1) for n in tr_pred.tolist()])
												for i in range(tr_pred.shape[0]//2):
									#				img_r=cv2.circle(img_r, (round(tr_pred[2*i].item()-x_left),round(tr_pred[2*i+1].item()-y_top)), radius=2, color=(0, 0, 255), thickness=-1)
													img_r=cv2.circle(img_r, (round(tr_pred[2*i].item()),round(tr_pred[2*i+1].item())), radius=2, color=(0, 0, 255), thickness=-1)
											t_f_lstm=time.time_ns()//1000000
									ltrb = track.to_ltrb()
									#print(track_id,":",ltrb)
									if int(track_id)!=interesting_track_id:
										img_r=cv2.rectangle(img_r,(int(ltrb[0]),int(ltrb[1])),(int(ltrb[2]),int(ltrb[3])),(0,255,0),2)
									else:
										img_r=cv2.rectangle(img_r,(int(ltrb[0]),int(ltrb[1])),(int(ltrb[2]),int(ltrb[3])),(0,0,255),2)
									cv2.putText(img_r,str(track_id),(int(ltrb[0]),int(ltrb[1])),0,0.6,(0,255,0))
								g_current_frame=img_r
								print("Found",total_tracks,"confirmed tracks. Tracking time:", tracking_time_fin-tracking_time_init)
								print("Id lst:", id_lst)
								print("tracks seen:",tracks_seen)
								#################################################################################################################
			#					if frame_id==3:													#
			#						sim_matrix_lst=np.zeros((len(id_lst),len(id_lst)))
			#						tmp_counter=np.zeros((len(id_lst),len(id_lst)))								#
			#					if frame_id>3 and frame_id<sim_stop_at_frame:									#
			#						new_sim_mat=create_similarity_map(init_feats, new_feats, new_locs, init_locs,tmp_counter)				#
			#						sim_matrix_lst+=new_sim_mat										#Used for similarity map testing
			#						if frame_id==sim_stop_at_frame-1:									#
			#							counts=np.zeros((len(id_lst),len(id_lst)))
			#							for i in range(sim_matrix_lst.shape[0]):							#
			#								for j in range(sim_matrix_lst.shape[0]):						#
			#									sim_matrix_lst[i][j]=sim_matrix_lst[i][j]/min(count_lst[i],count_lst[j])	#
			#									counts[i][j]=min(count_lst[i],count_lst[j])
			#							print(sim_matrix_lst)										#
			#							print(counts.astype(np.int32))
			#							print(tmp_counter.astype(np.int32))
			#							np.save("sim_matrix_test_yolov8_custom.npy",sim_matrix_lst)					#
								#################################################################################################################
								if last_seen!=0 and last_seen<timestamp and rec_opt>0 and reloc==1: # track lost and we need to relocate target
										#print("Lost target, relocating...")
										seen_n_frames_ago=int((timestamp-last_seen)/(sum_timestep/timestep_cnt)) #how many avg timesteps have passed since last seen
										skip_reloc=0
										if seen_n_frames_ago>30:
											tr_pred=[] #track predictions, initially empty
											interesting_history=[] #track_history of max 10 steps in the form required for trajectory prediction
											hist_times=[]
											prev_time=0
											sum_timestep=0 #in ms
											timestep_cnt=0
											last_seen=0 # timestamp when interesting track was seen
											frame_id=-1             #Used for similarity map testing
											init_feats=[]           #Used for similarity map testing
											id_lst=[]               #Used for similarity map testing
											count_lst=[]            #Used for similarity map testing
											init_locs=[]            #Used for similarity map testing
											sim_matrix_lst=[]       #Used for similarity map testing
											sim_stop_at_frame=13    #Used for similarity map testing
											target_x=-800
											target_y=-800
											skip_reloc=1
										if skip_reloc==0:
											target,confidence=relocate_target(current_detection_features,last_target_features,tr_pred,seen_n_frames_ago,last_w,last_h,x_left,y_top)
											if len(target)>0:
												d_sq_min=100000000
												xc_t=(target[1][0]+target[1][2])/2
												yc_t=(target[1][1]+target[1][3])/2
												target_x=xc_t
												target_y=yc_t
												for track in tracks: # scan all tracks to find the closest to our relocated target
													track_ltwh=track.to_ltwh()
													dist_sq_from_relocated=(track_ltwh[0]+track_ltwh[2]/2-xc_t)**2+(track_ltwh[1]+track_ltwh[3]/2-yc_t)**2
													if dist_sq_from_relocated<d_sq_min:
														d_sq_min=dist_sq_from_relocated
														interesting_track_id=int(track.track_id) # new interesting_track_id
														target_x=track_ltwh[0]+track_ltwh[2]/2
														target_y=track_ltwh[1]+track_ltwh[3]/2
														new_ltrb = track.to_ltrb()
												if d_sq_min<100000000:
													print(colored("Target relocated at "+str(round(target_x,1))+","+str(round(target_y,1))+". New track id: "+str(interesting_track_id),"green"))
													img_r=cv2.rectangle(img_r,(int(new_ltrb[0]),int(new_ltrb[1])),(int(new_ltrb[2]),int(new_ltrb[3])),(255,0,0),2)
											else:
												target_x=-800
												target_y=-800
								cv2.imwrite("server_track_res/"+str(frame_id)+".jpg",img_r)

								recv_data_tot=b''
								t_fin=time.time_ns()//1000000
								print("Frame processed in",t_fin-t_init,"ms ---- YOLO:",t_f_detr-t_i_detr,"ms --- LSTM:",t_f_lstm-t_i_lstm,"ms")
								s_lst=[]
								if len(tr_pred)==12:
									for i in range(6):
										s_lst.append(tr_pred[2*i].item())
										s_lst.append(tr_pred[2*i+1].item())
										s_lst.append(timestamp+(i+1)*(sum_timestep/timestep_cnt))
								else:
									s_lst=[target_x,target_y,timestamp]
								if rec_opt==-2:
									s_lst=[int(def_shape[1]/2),int(def_shape[0]/2), timestamp]
									target_x=int(def_shape[1]/2)
									target_y=int(def_shape[0]/2)
								elif rec_opt<0:
									s_lst=[-800,-800, timestamp]
									target_x=-800
									target_y=-800
								a=np.array(s_lst) # Send target position at valid timestamp
								a=a.astype(int)
								a=a.tobytes()+b'end*end*end'
								conn.sendall(a)
								if rec_opt==-1:
									break
							recv_data_tot+=tmp

					conn.close()
					print("connection closed")
					print("============================================")
			except Exception as e:
				print("Error:",e)
				t_select_run=0
				t_select_thread.join()
				print("============================================")
				conn.close()
		t_select_run=0
		t_select_thread.join()
	except Exception as e:
		s.close()
		print("There was an error:",e)
