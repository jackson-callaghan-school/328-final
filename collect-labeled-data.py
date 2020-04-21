import socket, sys

host = ''
port = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind((host, port))

timestamp = 0
accel_x = 0
accel_y = 0
accel_z = 0
gyro_x = 0
gyro_y = 0
gyro_z = 0
counter = 0

filename = sys.argv[1]
label = sys.argv[2]

file = open(filename,'w')

while 1:
    try:
        message, address = s.recvfrom(8192)
        info = message.decode()
        
        #print (info)
        data = info.split(",")
        if(len(data)<13):
            continue
        
        counter+=1 
        #print(data)
        for i in range(9):
            
            data[i] = data[i].strip()
            #print(str(i) + " " + data[i])
            
        timestamp = data[0]
        accel_x = data[2]
        accel_y = data[3]
        accel_z = data[4]
        gyro_x = data[6]
        gyro_y = data[7]
        gyro_z = data[8]
        print("record = " + str(counter))
        print("timestamp = " + str(timestamp)
        + " accel_x = " + str(accel_x) 
        + " accel_y = " + str(accel_y) 
        + " accel_z = " + str(accel_z))
        print("timestamp = " + str(timestamp)
        + " gyro_x = " + str(gyro_x) 
        + " gyro_y = " + str(gyro_y) 
        + " gyro_z = " + str(gyro_z))
        
        file.write(str(timestamp) + ',' + str(accel_x) + ',' + str(accel_y)+ ',' + str(accel_z) + ','  + str(gyro_x)+ ',' + str(gyro_y) + ',' + str(gyro_z) + ',' + str(label) + '\n')
    except KeyboardInterrupt:
        print("interrupted")
        file.close()