import numpy as np
import socket
import matplotlib.pyplot as plt
import cv2
import time
import select

def send_pwd_and_wait_ok(conn):
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

def show_select(host,port):
	s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((host,port))
	conn_ok=send_pwd_and_wait_ok(s)
	if conn_ok:
		inp=0 # 0:send image with track ids  -----  -1:stop tracking  ----  -2:track closest to the center of the image  ----- >0: Track id to follow
		print("0:send image with track ids  -----  -1:stop tracking  ----  -2:track closest to the center of the image  ----- >0: Track id to follow")
		while(1):
			if inp==0:
				recv_data_tot=b''
				while(1):
					recv_data=s.recv(1024)
					if(recv_data==b''):
						break
					recv_data_tot+=recv_data
					if b'end*end*end' in recv_data_tot:
						tmp=b''
						recv_data_lst=recv_data_tot.split(b'end*end*end')[:-1]
						if recv_data_tot.split(b'end*end*end')[-1]!=b'':
							tmp=recv_data_tot.split(b'end*end*end')[-1]
						r=recv_data_lst[-1]
						rec_img=cv2.imdecode(np.frombuffer(r,dtype=np.uint8), cv2.IMREAD_COLOR)
						cv2.imshow("Frame",rec_img)
						k=cv2.waitKey(200)
						if k==ord('q'):
							cv2.destroyAllWindows()
							break
						print("-----------------------------")
						recv_data_tot=tmp
						s.sendall(np.array([0]).astype(int).tobytes()+b'end*end*end') #keep sending 0 every second until q is pressed
			else:
				recv_data_tot=b''
				while(1):
					recv_data=s.recv(1024)
					if(recv_data==b''):
						break
					recv_data_tot+=recv_data
					if b'end' in recv_data_tot:
						tmp=b''
						recv_data_lst=recv_data_tot.split(b'end')[:-1]
						if recv_data_tot.split(b'end')[-1]!=b'':
							tmp=recv_data_tot.split(b'end')[-1]
						r=recv_data_lst[-1]
						print(r)
						print("-----------------------------")
						recv_data_tot=tmp
						break
			inp=int(input())
			s.sendall(np.array([inp]).astype(int).tobytes()+b'end*end*end')
		else:
			raise Exception("Cannot connect to server")
while 1:
	time.sleep(0.2)
	try:
		show_select("147.102.74.191",8081)
	except Exception as e:
		print("There was an error:",e)
