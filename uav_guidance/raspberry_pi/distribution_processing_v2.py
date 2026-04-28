import threading
from f_4g_comm import *
import cv2
import time
from openvino.inference_engine import IECore
import torch
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import torch.nn as nn
import socket
from termcolor import colored
from pymavlink import mavutil
import select
import rtsp
import traceback2 as traceback
import ffmpeg
import os
import gi
import subprocess
#import multiprocessing as mp

gi.require_version('Gst','1.0')
from gi.repository import Gst,GLib

os.environ["MAVLINK20"]='1'

Gst.init(None)

def set_mode(connection,mode):
	connection.set_mode(connection.mode_mapping()[mode])

def set_velocity(velocity_x, velocity_y, connection):
    connection.mav.set_position_target_local_ned_send(
        0,
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b111111000111, 0, 0, 0, velocity_x, velocity_y, 0,0, 0, 0, 0, 0)

def check_guided(conn):
	msg = conn.recv_match(type = 'HEARTBEAT', blocking = False)
	l=["GUIDED","POSHOLD","LOITER","LAND"]
	r=-1
	while msg:
		mode = mavutil.mode_string_v10(msg)
		if mode in l:
			if mode=="GUIDED":
				r=1
			else:
				r=0
		msg = conn.recv_match(type = 'HEARTBEAT', blocking = False)
	return r

def send_pwd_and_wait_ok(conn):										##############
	conn.sendall(b'tmp_pwdend*end*end')
	t_init=time.time()
	recv_data_tot=b''
	while 1:
		read_ready,_,_=select.select([conn],[],[],0)
		if read_ready:
			recv_data=conn.recv(512)
			if(recv_data==b''):
				return 0
			recv_data_tot+=recv_data
			if b'end*end*end' in recv_data_tot:
				recv_data_lst=recv_data_tot.split(b'end*end*end')[:-1]
				if recv_data_lst[0]==b'ok':
					return 1
				else:
					return 0
		if time.time()-t_init>2:
			return 0

def send_pwd_and_wait_ok_4g(ser):										##############
	r=send_data_4g(ser,b'tmp_pwdend*end*end',256)
	t_0=time.time()
	while r==0:
		if time.time()-t_0>10:
			raise Exception("Cannot send auth to server")
		time.sleep(1)
		r=send_data_4g(ser,b'tmp_pwdend*end*end',256)
	t_init=time.time()
	recv_data_tot=b''
	while 1:
		recv_data=recv_data_4g(ser,timeout=60)
		if(recv_data==b''):
			return 0
		recv_data_tot+=recv_data
		if b'end*end*end' in recv_data_tot:
			recv_data_lst=recv_data_tot.split(b'end*end*end')[:-1]
			if recv_data_lst[0]==b'ok':
				return 1
			else:
				return 0


def int_to_int8_lst(num):
        r_lst=[]
        if num==0:
        	r_lst=[0]
        while num//255>=0 and num>0:
                r_lst.append(num%255)
                num=num//255
        return r_lst

def int_8_lst_to_int(lst):
        num=0
        for i in range(len(lst)):
                num+=lst[i]*255**i
        return num

def create_frame_info(timestamp_ms,y_top,x_left,scaling_factor):
        s0=int_to_int8_lst(timestamp_ms)
        s1=int_to_int8_lst(y_top)
        s2=int_to_int8_lst(x_left)
        s3=int_to_int8_lst(scaling_factor)
        l=s0+s1+s2+s3+[len(s0),len(s1),len(s2),len(s3)]
       # print("########################################################",y_top,x_left,l)
        return np.array(l)

def init_exec(model_xml,model_bin):
	print("Reading model and loading on ncs2...")
	ie=IECore()
	net=ie.read_network(model=model_xml,weights=model_bin)
	nexec=ie.load_network(network=net, device_name="MYRIAD")
	print("Done")
	return nexec

def get_yolov5n_output(img,n_exec):
#	print("Geting results from YOLO model...")
	img_blob = cv2.dnn.blobFromImage(img, 1/255 , (512,288), swapRB=True, crop=False)
	t1=time.time_ns()//1000000
	outs=n_exec.infer({'images':img_blob})
#	print("++++++++++++++++++++++++++++++++++++")
	print("Inference time:",time.time_ns()//1000000-t1,"ms")
	return outs['output0'][0]

def get_yolo_boxes_nms_from_raw_2(raw_inp,xc,yc,exp_shape=[480,640,3]):
	print("Processing YOLO output...")
	min_conf=0.4
	mask=raw_inp[:,4]>min_conf
	conf_filt_inp=raw_inp[mask]
	sc_y=288/exp_shape[0]
	sc_x=512/exp_shape[1]

	t_yp_init=time.time_ns()//1000000
	conf_arr=[]
	box_arr=[]
	r_box_arr=[]
	class_est_arr=[]
	d_sq_min=10000000
	bbox_b=(-1000,-1000,10,10) #ltwh box for tracker
	for i in range(conf_filt_inp.shape[0]):
		conf=conf_filt_inp[i][4]
		conf_arr.append(conf)
		exp_class=np.argmax(conf_filt_inp[i][5:])
		class_est_arr.append(exp_class)
		lt_x=conf_filt_inp[i][0]-0.5*conf_filt_inp[i][2]
		lt_y=conf_filt_inp[i][1]-0.5*conf_filt_inp[i][3]
		r_box_arr.append(([lt_x/sc_x,lt_y/sc_y,conf_filt_inp[i][2]/sc_x,conf_filt_inp[i][3]/sc_y],conf,exp_class)) # ( [left,top,w,h] , confidence, de>
		box_arr.append([lt_x/sc_x,lt_y/sc_y,conf_filt_inp[i][2]/sc_x,conf_filt_inp[i][3]/sc_y])
		if (lt_x/sc_x+0.5*conf_filt_inp[i][2]/sc_x-xc)**2+(lt_y/sc_y+0.5*conf_filt_inp[i][3]/sc_y-yc)**2<d_sq_min:
			d_sq_min=(lt_x/sc_x+0.5*conf_filt_inp[i][2]/sc_x-xc)**2+(lt_y/sc_y+0.5*conf_filt_inp[i][3]/sc_y-yc)**2
			bbox_b=(int(lt_x/sc_x),int(lt_y/sc_y),int(conf_filt_inp[i][2]/sc_x),int(conf_filt_inp[i][3]/sc_y))
	conf_arr=np.array(conf_arr)
	box_arr=np.array(box_arr)
	class_est_arr=np.array(class_est_arr)
	idx_arr=cv2.dnn.NMSBoxes(box_arr,conf_arr,0.4,0.7)

	res=[]
	if len(idx_arr)!=0:
		for i in range(idx_arr.shape[0]):
			res.append(r_box_arr[i])
	t_yp_fin=time.time_ns()//1000000
	print("Yolo processing time:",t_yp_fin-t_yp_init,"ms")
	return (bbox_b, res)
'''
class frame_getter:
	def __init__(self,stream_url,shape=[1080,1920,3]):
		self.running=1
		self.shape=shape
		self.url=stream_url
		self.frame=np.array([[0,0,0],[0,0,0]])
		self.thread=threading.Thread(target=self.update, daemon=True)
		self.thread.start()
		self.frame_ready=0
	def update(self):
		while self.running:
			ffmpeg_cmd=[
				"ffmpeg",
				"-i", self.url,
				"-fflags", "nobuffer",
				"-flags","low_delay",
				"-tune","zerolatency",
				"-f", "rawvideo",
				"-pix_fmt","bgr24",
#				"-framedrop"
				"-"
			]
			process=subprocess.Popen(ffmpeg_cmd,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=2*10**7)
			while self.running:
				recv_b=process.stdout.read(self.shape[0]*self.shape[1]*self.shape[2])
				if not recv_b:
					break
				self.frame=np.frombuffer(recv_b,np.uint8).reshape(self.shape)
				self.frame_ready=1

	def get_last_frame(self):
		ret=self.frame_ready
		self.frame_ready=0
		return ret,self.frame

	def close(self):
		self.running=0
		self.thread.join()
		self.s.release()
'''
'''
class frame_getter:
	def __init__(self,stream_url):
		self.url=stream_url
		self.running=1
		self.p_c, c_c=mp.Pipe()
		self.p=mp.Process(target=self.update, args=(c_c, self.url))
		self.p.daemon=True
		self.p.start()

	def update(self, ch_c,url):
		timeout=0.15
		s=cv2.VideoCapture(url,apiPreference=cv2.CAP_FFMPEG,params=[cv2.CAP_PROP_READ_TIMEOUT_MSEC,120])
		t_no_frame_init=time.time()
		frame_ready=0
		frame=np.array([[0,0,0],[0,0,0]])
		while 1:
			if not s.grab():
				time.sleep(0.001)
			else:
				t_no_frame_init=time.time()
				frame_ready,frame=s.retrieve()
			print("in while...")
			if ch_c.poll()==1:
				print("Poll==1")
				ti=time.time()
				req=ch_c.recv()
				if req==1:
					ch_c.send([frame_ready,frame])
					frame_ready=0
				elif req==-1:
					s.release()
					break
				print(time.time()-ti)
			if time.time()-t_no_frame_init>timeout:
				s.release()
				s=cv2.VideoCapture(self.url,apiPreference=cv2.CAP_FFMPEG,params=[cv2.CAP_PROP_READ_TIMEOUT_MSEC,120])
				t_no_frame_init=time.time()
		ch_c.close()

	def get_last_frame(self):
		self.p_c.send(1)
		ret,frame=self.p_c.recv()
		return ret,frame

'''

class frame_getter:
	def __init__(self,stream_url):
		self.url=stream_url
		self.running=1
		self.frame_ready=0
		self.frame=np.array([[0,0,0],[0,0,0]])
		self.thread=threading.Thread(target=self.update, daemon=True)
		self.thread.start()

	def update(self):
		timeout=0.15
		self.s=cv2.VideoCapture(0,apiPreference=cv2.CAP_ANY,params=[cv2.CAP_PROP_BUFFERSIZE,1]) #self.url,apiPreference=cv2.CAP_FFMPEG,params=[cv2.CAP_PROP_READ_TIMEOUT_MSEC,60])
		t_no_frame_init=time.time()
		while self.running:
			if not self.s.grab():
				time.sleep(0.001)
			else:
				t_no_frame_init=time.time()
				self.frame_ready,self.frame=self.s.retrieve()
			if time.time()-t_no_frame_init>timeout:
				self.s.release()
				self.s=cv2.VideoCapture(0,apiPreference=cv2.CAP_ANY,params=[cv2.CAP_PROP_BUFFERSIZE,1]) #self.url,apiPreference=cv2.CAP_FFMPEG,params=[cv2.CAP_PROP_READ_TIMEOUT_MSEC,60])
				t_no_frame_init=time.time()
		self.s.release()

	def get_last_frame(self):
		ret=self.frame_ready
		self.frame_ready=0
		return ret,self.frame

	def close(self):
		self.running=0
		self.thread.join()
'''
class frame_getter:
	def __init__(self,url):
		self.url=url
		self.frame=[]
		self.frame_ready=0
		self.thread=threading.Thread(target=self.update, daemon=True)
		self.thread.start()

	def on_message(self, bus, message, pipeline):
		msg_type = message.type
		print(msg_type)
		if msg_type == Gst.MessageType.EOS:
			pipeline.set_state(Gst.State.NULL)
		elif msg_type == Gst.MessageType.ERROR:
			err, debug = message.parse_error()
			pipeline.set_state(Gst.State.NULL)
		elif msg_type == Gst.MessageType.STATE_CHANGED:
			old_state, new_state, pending_state = message.parse_state_changed()

	def on_buffer(self, sink, data):
		sample = sink.emit('pull-sample')
		if sample:
			buf = sample.get_buffer()
			caps = sample.get_caps()

			# Get width and height from caps
			structure = caps.get_structure(0)
			width = structure.get_value('width')
			height = structure.get_value('height')
			# Extract data from buffer
			success, mapinfo = buf.map(Gst.MapFlags.READ)
			if not success:
				return Gst.FlowReturn.ERROR

			try:
				# Create numpy array from buffer data
				yuv = np.frombuffer(mapinfo.data, np.uint8).reshape((height * 3 // 2, width))
				self.frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
				self.frame_ready=1
			finally:
				buf.unmap(mapinfo)
		return Gst.FlowReturn.OK

	def update(self):
		# Create a pipeline
		pipeline = Gst.parse_launch(
			'rtspsrc location=rtsp://192.168.144.25:8554/main.264 latency=10 buffer-mode=0 !'
			' decodebin ! videoconvert ! appsink name=sink emit-signals=True sync=false'
		)

		# Get the appsink element and connect to the callback
		sink = pipeline.get_by_name('sink')
		frame_data = {'frame': None, 'loop': GLib.MainLoop()}
		sink.connect('new-sample', self.on_buffer, frame_data)

		# Start the pipeline
		bus = pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect('message', self.on_message, pipeline)
		pipeline.set_state(Gst.State.PLAYING)

		try:
			while True:
				print("waiting msg...")
				msg = bus.timed_pop_filtered(10000000, Gst.MessageType.ANY)
				if msg:
					print("Message!!!")
					bus.emit('message', msg)
				time.sleep(0.1)
		except KeyboardInterrupt:
			pass

	def get_last_frame(self):
		ret=self.frame_ready
		self.frame_ready=0
		return ret,self.frame
'''

def find_new_factor(factor,base,change_list,last_d_error):
	if len(change_list)==0:
		change_list.append(base/15)
		return factor+base/15
	if last_d_error>0:
		del change_list[:-1]
		change_list[0]=-change_list[0]
		return factor+change_list[0]
	else:
		if change_list[-1]>0:
			change_list.append(min(3,len(change_list))*base/15)
			return factor+(min(3,len(change_list)-1))*base/15
		else:
			change_list.append(-min(3,len(change_list))*base/15)
			return factor-(min(3,len(change_list)-1))*base/15

class distributor_processor:
	def __init__(self,modelxml,modelbin,cam_url):
		self.following_mode=0 #0 for getting only initial location from server, 3 for autonomous function (centered detection as target)
		self.last_frame=[]
		self.new_target_est=[-800,-800]
		self.expected_shape=[480,640,3]
		self.host="147.102.74.191"
		self.port=8080
		self.xct=-800
		self.yct=-800
		self.last_v_rec_time=time.time()
		self.avg_len=0
		self.got_new_info=0
		self.frame_id=-1
		self.track_seen=0
		self.running=1
		self.vx=-1
		self.vy=-1
		self.t_last_det=time.time()
		self.t_last_update=0
		self.reloc_attempts=0
		self.n_exec=init_exec(modelxml,modelbin)
		self.getter=frame_getter(cam_url)
		self.wifi=0
		if self.following_mode!=3:
			if self.wifi:
				self.sending_thread=threading.Thread(target=self.transmit_frames_wifi, daemon=True)
				self.sending_thread.start()
			else:
				self.sending_thread=threading.Thread(target=self.transmit_frames_4G, daemon=True)
				self.sending_thread.start()
		self.new_pos_speed_idx=0
#		self.following_thread=threading.Thread(target=self.adjust_speed_simple, daemon=True)
#		self.following_thread.start()

	def transmit_frames_wifi(self):
		while self.running:															#
			time.sleep(0.2)														#
			try:															#
				self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)							#
				self.s.connect((self.host,self.port))										#
				auth_ok=send_pwd_and_wait_ok(self.s)										#
				if auth_ok:													#
					print("Auth ok")											#
				else:														#
					raise Exception("Cannot connect to server")
				frame_id=-1
				c=0
				recv_data_tot=b''
				sent_frames=0
				got_responses=0
				t_last_frame=time.time()
				while self.running:
					if frame_id<self.frame_id:
						t_last_frame=time.time()
						c+=1
						frame_id=self.frame_id
						img_r=self.last_frame
						y_top=0
						x_left=0
						scaling_factor=1
						#if self.avg_len>0 and c>3: #if we are looking at a target, transmit cut frames
						#	y_top=int(self.yct-6*self.avg_len)
						#	x_left=int(self.xct-6*self.avg_len)
						#	y_bottom=int(self.yct+6*self.avg_len)
						#	x_right=int(self.xct+6*self.avg_len)
						#	print("Img shape:",img_r.shape, "y_top:",y_top,"y_bottom:",y_bottom, "x_left",x_left,"x_right",x_right, "self.avg_len:",self.avg_len)
						#	img_r=img_r[max(0,y_top):min(img_r.shape[0],y_bottom),max(0,x_left):min(img_r.shape[1],x_right),:]
						scaling_factor=2
						img_r=cv2.resize(img_r,(int(img_r.shape[1]/scaling_factor), int(img_r.shape[0]/scaling_factor)))
						img_r1=cv2.imencode('.jpg',img_r)[1]
						tmstmp=time.time_ns()//1000000
						a=np.append(img_r1,create_frame_info(tmstmp,max(0,y_top),max(0,x_left),scaling_factor))
						a=a.astype(np.uint8)
						a=a.tobytes()+b'end*end*end'
						_,write_ready,_=select.select([],[self.s],[],0.05)
						if write_ready:
							self.s.sendall(a)
							sent_frames+=1
						else:
							frame_id-=1
					while(self.running):
						if time.time()-t_last_frame>5:
							self.s.close()
							raise Exception("Got nothing from server in the last 5 seconds, retrying...")
						read_ready,_,_=select.select([self.s],[],[],0)
						if read_ready:
							recv_data=self.s.recv(1024)
							if(recv_data==b''):
								break
							recv_data_tot+=recv_data
							if b'end*end*end' in recv_data_tot:
								tmp=b''
								recv_data_lst=recv_data_tot.split(b'end*end*end')[:-1]
								got_responses+=len(recv_data_lst)
								if recv_data_tot.split(b'end*end*end')[-1]!=b'':
									tmp=recv_data_tot.split(b'end*end*end')[-1]
								r=recv_data_lst[-1]
								target_est=np.frombuffer(r,dtype=int) #target_est=[x1,y1,t1,..,x6,y6,t6] from server predictions
								recv_data_tot=tmp
								current_time_ms=time.time_ns()//1000000
								dist_from_current=100000
								min_i=5
								for i in range(len(target_est)//3):
									ts=target_est[3*i+2]
									if abs(current_time_ms-ts)<dist_from_current:
										dist_from_current=abs(current_time_ms-ts)
										min_i=i
								self.new_target_est=[target_est[min_i*3],target_est[min_i*3+1]]
								self.got_new_info=1
						if sent_frames-got_responses<3:
							break
				self.s.close()
			except Exception as e:
				print("There was an error:",e)
				print(traceback.format_exc())

	def transmit_frames_4G(self):
		while self.running:
			time.sleep(0.2)
			try:
				print("trying 4g connection")
				ser_tmp=serial.Serial('/dev/ttyAMA0',115200)
				baud_3000000(ser_tmp)
				ser_tmp.close()
				ser=serial.Serial('/dev/ttyAMA0',3000000)
				r0=open_tcp_conn(ser,self.host,str(self.port))
				if r0:
					print("connected, trying to get server access...")
					auth_ok=send_pwd_and_wait_ok_4g(ser)
					if auth_ok:
						print("Auth ok")
						frame_id=-1
						c=0
						recv_data_tot=b''
						sent_frames=0
						got_responses=0
						t_last_frame=time.time()
						while self.running:
							if frame_id<self.frame_id:
								c+=1
								t_last_frame=time.time()
								frame_id=self.frame_id
								img_r=self.last_frame
								y_top=0
								x_left=0
							#	scaling_factor=1
							#	if self.avg_len>0 and c>3: #if we are looking at a target, transmit cut frames
							#		y_top=int(self.yct-2.5*self.avg_len)
							#		x_left=int(self.xct-2.5*self.avg_len)
							#		y_bottom=int(self.yct+2.5*self.avg_len)
							#		x_right=int(self.xct+2.5*self.avg_len)
							#		img_r=img_r[max(0,y_top):min(img_r.shape[0],y_bottom),max(0,x_left):min(img_r.shape[1],x_right),:]
							#	elif self.xct==-800: # if we do not have a target located, resize
								scaling_factor=2
								img_r=cv2.resize(img_r,(int(img_r.shape[1]/scaling_factor), int(img_r.shape[0]/scaling_factor)))
								img_r1=cv2.imencode('.jpg',img_r)[1]
								tmstmp=time.time_ns()//1000000
								a=np.append(img_r1,create_frame_info(tmstmp,max(0,y_top),max(0,x_left),scaling_factor))
								a=a.astype(np.uint8)
								a=a.tobytes()+b'end*end*end'
								send_r=send_data_4g(ser,a,1500)
								print("++++++++++++++++++DATA SENT+++++++++++++++")
								if send_r!=1:
									close_tcp_conn(ser)
									raise Exception("Problem sending data")
								sent_frames+=1
								while 1:
									if time.time()-t_last_frame>10:
										close_tcp_conn(ser)
										raise Exception("Got nothing from server in the last 10 seconds")
									recv_data=recv_data_4g(ser,timeout=0.01,t_o_wait=0.01)
									if(recv_data==b''):
										break
									if recv_data!=-1:
										recv_data_tot+=recv_data
										if b'end*end*end' in recv_data_tot:
											tmp=b''
											recv_data_lst=recv_data_tot.split(b'end*end*end')[:-1]
											got_responses+=len(recv_data_lst)
											if recv_data_tot.split(b'end*end*end')[-1]!=b'':
												tmp=recv_data_tot.split(b'end*end*end')[-1]
											r=recv_data_lst[-1]
											target_est=np.frombuffer(r,dtype=int) #target_est=[x1,y1,t1,..,x6,y6,t6] from server predictions
											current_time_ms=time.time_ns()//1000000
											dist_from_current=100000
											recv_data_tot=tmp
											min_i=5
											for i in range(len(target_est)//3):
												ts=target_est[3*i+2]
												if abs(current_time_ms-ts)<dist_from_current:
													dist_from_current=abs(current_time_ms-ts)
													min_i=i
											self.new_target_est=[target_est[min_i*3],target_est[min_i*3+1]]
											self.got_new_info=1
									if sent_frames-got_responses<3:
										break
					else:
						raise Exception("Cannot connect to server")
				close_tcp_conn(ser)
			except Exception as e:
				print("transmit_frames_4G: There was an error:",e)
				print(traceback.format_exc())
				close_tcp_conn(ser)

	def adjust_speed_simple(self):
		conn_veh=mavutil.mavlink_connection('/dev/ttyUSB0',baud=57600)
		last_mode=0 #----------------------------------#
		while self.running:
			time.sleep(0.1)
			try:
				curr_mode=check_guided(conn_veh)                                                #-------------------------------#
				print("Checked_guided:",curr_mode)
				if curr_mode!=-1:                                                               #-------------------------------#
					last_mode=curr_mode                                                     #-------------------------------#
				print("Checking if we can start follow loop:    Avg_len:", self.avg_len,"   Last mode:", last_mode,"    Expected Shape:",self.expected_shape, "    xct:",self.xct,"    yct:",self.yct)
				if self.avg_len<=self.expected_shape[0]/3 and self.avg_len!=0 and last_mode==1: #-------------------------------#
					print("---------------------Guided Mode---------------------")
					avg_side_len=3
					current_uav_vx=0
					current_uav_vy=0
					first_loop=0
					previous_vx=0
					previous_vy=0
					avg_error_vx=0
					avg_error_vy=0
					err_count=0
					last_vx_error=0
					last_vy_error=0
					prev_error=0
					v_len=self.avg_len
					error_init_flag=1
					if v_len!=0:
						mpp=avg_side_len/v_len # Vx_real_mps = mpp*cos(theta)*Vx_seen_pps + mpp*sin(theta)*Vy_seen_pps
						theta=0		       # Vy_real_mps = -mpp*sin(theta)*Vx_seen_pps + mpp*cos(theta)*Vy_seen_pps
						mpp_base=avg_side_len/v_len
						theta_base=np.pi/2
						last_mpp_d_error=0
						last_theta_d_error=0
						mpp_change_lst=[]
						theta_change_lst=[]
						break_flag=0
						dvx_real=0
						dvy_real=0
						prev_mpp_error=0
						prev_theta_error=0
						sum_error=0
						centering_speed_x=0
						centering_speed_y=0
					else:
						break_flag=1
					while self.running:
						if break_flag==1:
							break
						if first_loop==0:
							first_loop=1
						else:
							first_loop=-1
						v_len=self.avg_len
						curr_mode=check_guided(conn_veh)        #----------------------------#
						if curr_mode!=-1:                       #----------------------------#
							last_mode=curr_mode             #----------------------------#
						if last_mode==0:                        #----------------------------#
							set_velocity(0,0,conn_veh)
							print("Breaking")		#----------------------------#
							break                           #----------------------------#
						if v_len==0:
							set_velocity(0,0,conn_veh)
							break
						positions_lst=[]
						times_lst=[]
						while len(positions_lst)<2:
							if self.new_pos_speed_idx==1:
								positions_lst.append((self.xct, self.yct))
								times_lst.append(self.last_v_rec_time)
								self.new_pos_speed_idx=0
						last_vx=-(positions_lst[-1][1]-positions_lst[0][1])/(times_lst[-1]-times_lst[0]) #pixels per second ----- uav vx=-(object vy in frame)
						last_vy=(positions_lst[-1][0]-positions_lst[0][0])/(times_lst[-1]-times_lst[0])  #pixels per second ----- uav vy=object vx in frame
						if first_loop!=1:
							print("=====================================================",err_count)
							print("pos_lst:",positions_lst)
							expected_vx=centering_speed_x
							expected_vy=centering_speed_y
							print(expected_vx,expected_vy,last_vx,last_vy)
							print("old mpp",mpp,"old theta",theta)
							sum_error+=np.sqrt((expected_vx-last_vx)**2+(expected_vy-last_vy)**2)
							if err_count==0:
								if error_init_flag!=1:
									mpp=find_new_factor(mpp,mpp_base,mpp_change_lst,last_mpp_d_error) #last_xx_d_error is 0 or stored from previous round
							elif err_count==5:
								last_mpp_d_error=sum_error/6-prev_mpp_error
								prev_mpp_error=sum_error/6
								sum_error=0
							elif err_count==6:
								if error_init_flag!=1:
									theta=find_new_factor(theta,theta_base,theta_change_lst,last_theta_d_error)
							elif err_count==11:
								last_theta_d_error=sum_error/6-prev_theta_error
								prev_theta_error=sum_error/6
								sum_error=0
								err_count=-1
								error_init_flag=0
							err_count+=1
							print("new mpp:",mpp,"new theta:",theta)
							print("=======================================================")
						previous_vx=last_vx
						previous_vy=last_vy
						last_vx=mpp*np.cos(theta)*previous_vx+mpp*np.sin(theta)*previous_vy
						last_vy=mpp*np.cos(theta)*previous_vy-mpp*np.sin(theta)*previous_vx
						print("****",positions_lst,"****")
						print("Last_vx=",last_vx)
						print("Last_vy=",last_vy)
						x_c=self.last_frame.shape[1]/2
						y_c=self.last_frame.shape[0]/2
						#if abs(positions_lst[-1][1]-y_c)<25:
						#	vx_to_set=current_uav_vx+last_vx #meters per second
						#	dvx_real=-last_vx
						#	centering_speed_x=0
						#else:
						vx_to_set=current_uav_vx+last_vx-((positions_lst[-1][1]-y_c)*mpp*np.cos(theta)-(positions_lst[-1][0]-x_c)*mpp*np.sin(theta))/4 #meters per second
						dvx_real=-(last_vx-((positions_lst[-1][1]-y_c)*mpp*np.cos(theta)-(positions_lst[-1][0]-x_c)*mpp*np.sin(theta))/4)
						centering_speed_x=(positions_lst[-1][1]-y_c)/4
						#if abs(positions_lst[-1][0]-x_c)<25:
						#	vy_to_set=current_uav_vy+last_vy #meters per second
						#	dvy_real=-last_vy
						#	centering_speed_y=0
						#else:
						vy_to_set=current_uav_vy+last_vy-(-(positions_lst[-1][0]-x_c)*mpp*np.cos(theta)-(positions_lst[-1][1]-y_c)*mpp*np.sin(theta))/4 #meters per second
						dvy_real=-(last_vy-(-(positions_lst[-1][0]-x_c)*mpp*np.cos(theta)-(positions_lst[-1][1]-y_c)*mpp*np.sin(theta))/4)
						centering_speed_y=-(positions_lst[-1][0]-x_c)/4
						print("setting velocity to:",vx_to_set,vy_to_set)
						if vx_to_set>6:
							vx_to_set=6
						elif vx_to_set<-6:
							vx_to_set=-6
						if vy_to_set>6:
							vy_to_set=6
						elif vy_to_set<-6:
							vy_to_set=-6
						t_init_f=time.time()
						#while time.time()-t_init_f<0.1:
						set_velocity(vx_to_set,vy_to_set,conn_veh)
						current_uav_vx=vx_to_set
						current_uav_vy=vy_to_set
				else:
					set_velocity(0,0,conn_veh)
			except Exception as e:
				print("Speed Setting Error:",e)
				print(traceback.format_exc())

	def process_locally(self):
		time.sleep(0.005)
		r,img_r=self.getter.get_last_frame()
		if r:
			self.last_frame=img_r.copy()
			self.frame_id+=1
		#	print("---------------------------------------------------------------------------")
			if self.following_mode==3 and time.time()-self.last_v_rec_time>2: # if seen earler than 2s ago, erase old position
				self.new_pos_speed_idx=0
				self.xct=self.last_frame.shape[1]/2
				self.yct=self.last_frame.shape[0]/2
			elif self.following_mode!=3 and time.time()-self.last_v_rec_time>2:
				self.new_pos_speed_idx=0
				self.xct=-800
				self.yct=-800
			if self.xct!=-800:
				if self.track_seen==0 and time.time()-self.t_last_det>1:
					results=get_yolov5n_output(img_r,self.n_exec)
					self.t_last_det=time.time()
					best_box,results=get_yolo_boxes_nms_from_raw_2(results,self.xct,self.yct)
					if best_box[0]!=-1000:
						self.tracker = cv2.TrackerKCF_create()
						print(img_r.shape)
						print(best_box)
						self.tracker.init(img_r,best_box)
						self.track_seen,bbox=self.tracker.update(img_r)
						if self.track_seen:
							cv2.rectangle(img_r, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (255,0,0), 2)
							cv2.imwrite("img"+str(self.frame_id)+".jpg", img_r)
							print("Vehicle seen at",(round(bbox[0]+bbox[2]/2),round(bbox[1]+bbox[3]/2)))
							self.xct=round(bbox[0]+bbox[2]/2)
							self.yct=round(bbox[1]+bbox[3]/2)
							self.last_v_rec_time=time.time()
							self.avg_len=(bbox[2]+bbox[3])/2
							self.new_pos_speed_idx=1
						else:
							self.avg_len=0
							#self.tracker.release()
					else:
						self.avg_len=0
				elif self.track_seen!=0:
					if time.time()-self.t_last_det>8 and self.following_mode!=1:
						results=get_yolov5n_output(img_r,self.n_exec)
						self.t_last_det=time.time()
						best_box,results=get_yolo_boxes_nms_from_raw_2(results,self.xct,self.yct)
						if abs(best_box[0]-self.xct)<self.last_frame.shape[1]/4 and abs(best_box[0]-self.xct)<self.last_frame.shape[0]/4:
							#self.tracker.release()
							self.tracker = cv2.TrackerKCF_create()
							print(img_r.shape)
							print(best_box)
							self.tracker.init(img_r,best_box)
							self.track_seen,bbox=self.tracker.update(img_r)
							if self.track_seen:
								cv2.rectangle(img_r, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (255,0,0), 2)
								cv2.imwrite("img"+str(self.frame_id)+".jpg", img_r)
								print("Vehicle seen at",(round(bbox[0]+bbox[2]/2),round(bbox[1]+bbox[3]/2)))
								self.xct=round(bbox[0]+bbox[2]/2)
								self.yct=round(bbox[1]+bbox[3]/2)
								self.last_v_rec_time=time.time()
								self.avg_len=(bbox[2]+bbox[3])/2
								self.new_pos_speed_idx=1
							else:
								self.avg_len=0
								self.tracker.release()
						else:
							self.track_seen,bbox=self.tracker.update(img_r)
							if self.track_seen:
								cv2.rectangle(img_r, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,0,255), 2)
								cv2.imwrite("img"+str(self.frame_id)+".jpg", img_r)
								print("Vehicle seen at",(round(bbox[0]+bbox[2]/2),round(bbox[1]+bbox[3]/2)))
								self.xct=round(bbox[0]+bbox[2]/2)
								self.yct=round(bbox[1]+bbox[3]/2)
								self.last_v_rec_time=time.time()
								self.avg_len=(bbox[2]+bbox[3])/2
								self.new_pos_speed_idx=1
							else:
								self.avg_len=0
					else:
						self.track_seen,bbox=self.tracker.update(img_r)
						if self.track_seen:
							cv2.rectangle(img_r, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,0,255), 2)
							cv2.imwrite("img"+str(self.frame_id)+".jpg", img_r)
							print("Vehicle seen at",(round(bbox[0]+bbox[2]/2),round(bbox[1]+bbox[3]/2)))
							self.xct=round(bbox[0]+bbox[2]/2)
							self.yct=round(bbox[1]+bbox[3]/2)
							self.last_v_rec_time=time.time()
							self.avg_len=(bbox[2]+bbox[3])/2
							self.new_pos_speed_idx=1
						else:
							self.avg_len=0
			else:
				self.avg_len=0
			if self.got_new_info:
				target_est=self.new_target_est
				#print("Got target estimated position:",target_est,"--- Following Mode:",self.following_mode)
				self.got_new_info=0
				if self.following_mode==1: #full server mode
					if self.track_seen==0 or time.time()-self.t_last_update>5:
						self.t_last_update=time.time()
						self.xct=target_est[0]
						self.yct=target_est[1]
						self.track_seen=0
						self.avg_len=0
				elif self.following_mode==0: #server only for initial target detection or track_seen==0
					if self.track_seen==0:
						if target_est[0]!=-800:
							self.xct=target_est[0]
							self.yct=target_est[1]
				#print("Set xct,yct: ",self.xct,self.yct)
	def close(self):
		self.getter.close()
		self.running=0
		if self.following_mode!=3:
			if(self.wifi):
				self.s.close()
				self.sending_thread.join()
				print("Closed connection thread")
			else:
				self.sending_thread.join()
		self.following_thread.join()

#time.sleep(30)
try:
	os.system("sudo ip route del default via 192.168.144.1")
except:
	print("Error at system config")
while 1:
	time.sleep(1)
	try:
		dp=distributor_processor("yolov5n_10_270x512.xml","yolov5n_10_270x512.bin","rtsp://192.168.144.25:8554/main.264")
		time.sleep(10)
		br=0
		t_in=time.time()
		while 1:
			dp.process_locally()
			if time.time()-t_in>90:
				br=1
				break
		#dp.close()
		if br==1:
			break
	except Exception as e:												#
		print("There was an error:",e)
		print(traceback.format_exc())
