import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
x1 = np.multiply(x, x)
y1 = np.multiply(y, y)


#================create dataset========================
rng2 = np.random.RandomState(0)
random_input = rng2.randn(100*100)
random_input = random_input
for i in range(0,100*100):
    if random_input[i]>2 :
        random_input[i] =2
    if random_input[i]<-2 :
        random_input[i] = -2
        



dataset = np.zeros((100*100 , 4),float)
z3d = np.zeros((100,100),float)
z3d_p = np.zeros((1,100*100),float)
for i in range(0,100):
    for j in range(0,100):
        z3d[i][j] = np.sinc(x1[i] + y1[j])
        dataset[i*100 + j ][3] = z3d[i][j]
        dataset[i*100 + j ][0] = x[i]
        dataset[i*100 + j ][1] = y[j]
        dataset[i*100 + j ][2] = random_input[i*100+j]

        
        





#==================== Plot the surface ================
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, z3d, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()



'''
#======== if x1<0 and x2<0 then for x_random , we use dataset1
#======== if x1<0 and x2>0 then for x_random , we use dataset2
#======== if x1>0 and x2<0 then for x_random , we use dataset3
#======== if x1>0 and x2>0 then for x_random , we use dataset4





dataset1 = np.zeros((160*160 , 2),float)
dataset2 = np.zeros((160*160 , 2),float)
dataset3 = np.zeros((160*160 , 2),float)
dataset4 = np.zeros((160*160 , 2),float)
dataset5 = np.zeros((160*160 , 2),float)
dataset6 = np.zeros((160*160 , 2),float)
dataset7 = np.zeros((160*160 , 2),float)
dataset8 = np.zeros((160*160 , 2),float)


'''



dataset1 = np.zeros((100*100 , 2),float)
dataset2 = np.zeros((100*100 , 2),float)
dataset3 = np.zeros((100*100 , 2),float)

    
for i in range(0,100*100):
    #-------x1--------------------
    dataset1[i][0] = dataset[i][0]
    dataset1[i][1] = dataset[i][3]
    #------- x2 -------------------
    dataset2[i][0] = dataset[i][1]
    dataset2[i][1] = dataset[i][3]
    #------- x random--------------
    dataset3[i][0] = dataset[i][2]
    dataset3[i][1] = dataset[i][3]



#================create subarray ====================

subarray = np.array([   [0.3, 0.3, 0.3, 0.3, 0.3       ],
                        [0.6, 0.6, 0.9, 0.6, 0.6       ],
                        [0.6, 0.9, 0.0, 0.9, 0.6       ],
                        [0.6, 0.6, 0.9, 0.6, 0.6       ],
                        [0.3, 0.3, 0.3, 0.3, 0.3       ],                              ])







#======= creat IDS and narrow_path1 and spread for x1 ==================
IDS1 = np.zeros((100 , 100),float)
for i in range(0,100*100):
    k1 = dataset1[i][1] / 0.02
    k1 += 49
    k2 = dataset1[i][0] / 0.04
    k2 += 49

    k1 = int(k1)
    k2 = int(k2)
    IDS1[100-k1][k2] = 1


plt.imshow(IDS1 , interpolation='none')
plt.show()


for i in range(100):
    for j in range(0,100):
        if IDS1[i][j] ==1:
            if i>2 and i<97 and j>2 and j<97:
                for k1 in range(-2,2):
                    for k2 in range(-2,2):
                        IDS1[i+k1][j+k2] += subarray[k1+2][k2+2]
                        



max_ids = IDS1.max()
IDS1 = IDS1/max_ids

plt.imshow(IDS1 , interpolation='none')
plt.title("IDS table for x1 " ) 
plt.show()

       
out = np.linspace(1, -1, 100)
narrow_path1 = list()

for i in range(0,100):
    t=0
    for j in range(0,100):
        t+= (out[j] * IDS1[j][i])

    narrow_path1.append(t)

   


plt.plot(x , narrow_path1)
plt.title(" narrow_path for x1 " ) 
plt.show()



spread = list()

for i in range(0,100):
    t=0
    for j in range(0,100):
        if IDS1[j][i]!= 0:
            t+=1
    if t != 0:
        spread.append(1/t)
    else:
        spread.append(0)

plt.plot(x , spread)
plt.title(" 1/spread for x1 " )
plt.show()

        



    
    
    


    





#======= creat IDS and narrow_path1 and spread for random input ==================
IDS2 = np.zeros((100 , 100),float)
for i in range(0,100*100):
    k1 = dataset3[i][1] / 0.02
    k1 += 49
    k2 = dataset3[i][0] / 0.04
    k2 += 49

    k1 = int(k1)
    k2 = int(k2)
    IDS2[100-k1][k2] = 1


plt.imshow(IDS2 , interpolation='none')
plt.show()


for i in range(100):
    for j in range(0,100):
        if IDS2[i][j] ==1:
            if i>2 and i<97 and j>2 and j<97:
                for k1 in range(-2,2):
                    for k2 in range(-2,2):
                        IDS2[i+k1][j+k2] += subarray[k1+2][k2+2]
                        

max_ids = IDS2.max()
IDS2 = IDS2/max_ids
plt.imshow(IDS2 , interpolation='none')
plt.title("IDS table for random input " ) 
plt.show()

       
out = np.linspace(1, -1, 100)
narrow_path2 = list()

for i in range(0,100):
    t=0
    for j in range(0,100):
        t+= (out[j] * IDS2[j][i])

    narrow_path2.append(t)

   
plt.plot(x , narrow_path2)
plt.title(" narrow_path for random input " ) 
plt.show()



spread2 = list()

for i in range(0,100):
    t=0
    for j in range(0,100):
        if IDS2[j][i]!= 0:
            t+=1
    if t != 0:
        spread2.append(1/t)
    else:
        spread2.append(0)

plt.plot(x , spread2)
plt.title(" 1/spread for random input " )
plt.show()




#=================model with all data =====================================
y_p = list()
for i in range(0,100):
    for j in range(0,100):
        t = (narrow_path1[i] * spread[i] + narrow_path1[j] * spread[j] + narrow_path2[i] * spread2[i]) 
        y_p.append(t)


e = 0 
for i in range(0,100*100):
    e += (y_p[i] - dataset[i][3])*(y_p[i] - dataset[i][3])

print('error of ALM modeling with x1 and x2 and R1 : ' + str(e))
        



y_p = np.array(y_p,float)
y_p = np.reshape(y_p , (100,100))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, y_p, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title('ALM output with x1 and x2 and R1')
plt.show()




    



#=================model with x1 an x2 =====================================
y_p = list()
for i in range(0,100):
    for j in range(0,100):
        t = (narrow_path1[i] * spread[i] + narrow_path1[j] * spread[j] ) 
        y_p.append(t)


e = 0 
for i in range(0,100*100):
    e += (y_p[i] - dataset[i][3])*(y_p[i] - dataset[i][3])

print('error of ALM modeling with x1 and x2 : ' + str(e))
        



y_p = np.array(y_p,float)
y_p = np.reshape(y_p , (100,100))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, y_p, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title('ALM output with x1 and x2')
plt.show()



     



  
        
    









