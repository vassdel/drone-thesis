import serial
import time
import numpy as np
import pickle

def serial_send_kw(cm,ser_con,kw,ret_b=0,t_o=20,called_from_recv=0):
	#cm: command as bytes, ser_con: serial.Serial(...)
	#kw: list of key words to look for in response (str)
	t_o_data=1
	ser_con.write(cm)
#	print("-----command:",cm)
	r=""
	kw_found=0 #check if all keywords in kw list
	r_b=b''
	t_init=time.time()
	timeout_exit=0
	while (kw_found==0) and ("ERROR" not in r):
		if ser_con.in_waiting>0:
			r_b+=ser_con.read(ser_con.in_waiting)
			t_init=time.time() #restart countdown (to avoid interrupting reading from serial)
			r=str(r_b)
		kw_found=1
		for k in kw:
			if k not in r:
				kw_found=0
		if time.time()-t_init>t_o:
			timeout_exit=1
			r="TIMEOUT1"
			r_b=b'TIMEOUT1'
			break
#	if called_from_recv and timeout_exit==0:
#		ipd_split=r_b.split(b'IPD')
#		while b'\r\n' not in ipd_split[1]:
#			print("IPD split: ",ipd_split)
#			if ser_con.in_waiting>0:
#				r_b+=ser_con.read(ser_con.in_waiting)
#				ipd_split=r_b.split(b'IPD')
#			if time.time()-t_init>t_o_data:
#				r_b=b'TIMEOUT2'
#				return r_b
#		data_len=int(ipd_split[1].split(b'\r\n')[0])
#		data=b'\r\n'.join((b'IPD'.join(ipd_split[1:])).split(b'\r\n')[1:])
#		while(len(data)<data_len):
#			if ser_con.in_waiting>0:
#				data+=ser_con.read(ser_con.in_waiting)
#			if time.time()-t_init>t_o_data:
#				data=b'TIMEOUT2'
#				return data
#		print("returning:",data)
#		return data
	if ret_b==0:
#		print("returning:",r)
		return r
	else:
#		print("returning:",r_b)
		return r_b

def check_for_sim_pin(ser, sim_pin):
	resp=serial_send_kw(b'AT+CPIN?\r\n',ser,["READY", "OK"],t_o=3)
	if "ERROR" in resp:
		print("Error at PIN check !")
		return 0
	elif resp=="TIMEOUT1":
		resp1=serial_send_kw(b'AT+CPIN?\r\n',ser,["SIM PIN","OK"],t_o=3)
		if ("ERROR" in resp1) or (resp1=="TIMEOUT1"):
			print("Error at PIN check (SIM PIN)")
			return 0
		else:
			resp2=serial_send_kw(b'AT+CPIN='+sim_pin+b'\r\n',ser,["OK"],t_o=5)
			if ("ERROR" in resp2) or (resp2=="TIMEOUT1"):
				print("Error while enterig PIN")
				return 0
			else:
				time.sleep(15)
				return 1
	else:
		return 1

def baud_3000000(ser):
	resp=serial_send_kw(b'AT+IPR=3000000\r\n',ser,["OK"],t_o=5)
	while "ERROR" in resp:
		print("Error changing baud rate")
		time.sleep(1)
		resp=serial_send_kw(b'AT+IPR=3000000\r\n',ser,["OK"],t_o=5)
	return 1

def set_ciprxget_manual(ser):
	resp=serial_send_kw(b'AT+CIPRXGET=1\r\n',ser,["OK"])
	while "ERROR" in resp:
		print("Error at CIPRXGET=1 !")
		time.sleep(1)
		resp=serial_send_kw(b'AT+CIPRXGET=1\r\n',ser,["OK"])
	return 1

def open_tcp_conn(ser,ipaddr,port):
	check_for_sim_pin(ser,b'0183')
	resp=serial_send_kw(b'AT+NETOPEN\r\n',ser,["OK","NETOPEN:"])
	if "ERROR" in resp:
		print("Error at NETOPEN ! -- Network probably open")
	set_ciprxget_manual(ser)
	time.sleep(2)
	resp=serial_send_kw(('AT+CIPOPEN=0,"TCP","'+ipaddr+'",'+port+'\r\n').encode("utf-8"),ser,["CIPOPEN:"])
	time.sleep(1)
	if "ERROR" in resp or "TIMEOUT" in resp:
		print("Error at CIPOPEN !")
		serial_send_kw(b'AT+CIPCLOSE=0\r\n',ser,["CIPCLOSE:"],t_o=5)
		return 0
	return 1

def close_tcp_conn(ser):
	resp=serial_send_kw(b'AT+CIPCLOSE=0\r\n',ser,["CIPCLOSE:"])
	if "ERROR" in resp:
		print("Error at CIPCLOSE !")
	resp=serial_send_kw(b'AT+NETCLOSE\r\n',ser,["OK"])
	if "ERROR" in resp:
		print("Error at NETCLOSE !")
		return 0
	return 1

def send_data_4g(ser,a,max_b):
	time.sleep(0.005)
	if ser.in_waiting>0:
		t=ser.read(ser.in_waiting)
	l=len(a)
	if l<max_b:
		resp=serial_send_kw(('AT+CIPSEND=0,'+str(l)+'\r\n').encode('utf-8'),ser,[">"])
		if ("ERROR" in resp) or ("TIMEOUT" in resp):
			print("Error at CIPSEND !")
			return 0
		resp=serial_send_kw(a,ser,["CIPSEND: 0,"+str(l)+","+str(l)+'\\r\\n'])
		if ("ERROR" in resp) or ("TIMEOUT" in resp):
			print("Error while sending data !")
			return 0
	else:
		for i in range(l//max_b):
			resp=serial_send_kw(('AT+CIPSEND=0,'+str(max_b)+'\r\n').encode('utf-8'),ser,[">"])
			if ("ERROR" in resp) or ("TIMEOUT" in resp):
				print("Error at CIPSEND !")
				return 0
			resp=serial_send_kw(a[i*max_b:(i+1)*max_b],ser,["CIPSEND: 0,"+str(max_b)+","+str(max_b)+"\\r\\n"])
			if ("ERROR" in resp) or ("TIMEOUT" in resp):
				print("Error while sending data !")
				return 0
			time.sleep(0.01)
		if l%max_b!=0:
			resp=serial_send_kw(('AT+CIPSEND=0,'+str(l-(l//max_b)*max_b)+'\r\n').encode('utf-8'),ser,[">"])
			if ("ERROR" in resp) or ("TIMEOUT" in resp):
				print("Error at fin CIPSEND !")
				return 0
			resp=serial_send_kw(a[(l//max_b)*max_b:],ser,["CIPSEND: 0,"+str(l-(l//max_b)*max_b)+","+str(l-(l//max_b)*max_b)+"\\r\\n"])
			if ("ERROR" in resp) or ("TIMEOUT" in resp):
				print("Error while sending data !")
				return 0
	return 1

def recv_data_4g(ser,timeout=300,t_o_wait=0.1):
	t_init=time.time()
	while time.time()-t_init<=timeout:
		print("Sending ciprxget...")
		resp=serial_send_kw(b'AT+CIPRXGET=2,0\r\n',ser,["end*end*end"],ret_b=1,t_o=t_o_wait,called_from_recv=1)
		if resp==b'TIMEOUT1' or (b'ERROR' in resp):
			time.sleep(timeout/100)
		elif resp==b'TIMEOUT2':
			raise Exception("Timeout while reading from serial (recv_data_4g)")
		else:
			resp=resp.split(b'end*end*end')[-2].split(b'\r\n')[-1]+b'end*end*end'
			print("++++final:",resp)
			return resp
	return -1

#ser=serial.Serial('/dev/ttyAMA0',115200)
#
#ro=open_tcp_conn(ser,'147.102.74.191','8080')
#if ro:
#	print("-------------------------------------------------------------------------",type(r))
#	print(r)
#	print("-------------------------------------------------------------------------")
#close_tcp_conn(ser)
#ser.close()
