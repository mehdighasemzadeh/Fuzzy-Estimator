import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
from matplotlib import cm
import math
from sklearn.model_selection import train_test_split
from scipy import stats
#pip install fuzzy-c-means

x = np.linspace(-2, 2, 160)
y = np.linspace(-2, 2, 160)
x1 = np.multiply(x, x)
y1 = np.multiply(y, y)


#================create dataset========================
rng2 = np.random.RandomState(0)
random_input = rng2.randn(160*160)
random_input = random_input/2
dataset = np.zeros((160*160 , 4),float)
z3d = np.zeros((160,160),float)
z3d_p = np.zeros((1,160*160),float)
for i in range(0,160):
    for j in range(0,160):
        z3d[i][j] = np.sinc(x1[i] + y1[j])
        dataset[i*160 + j ][3] = z3d[i][j]
        dataset[i*160 + j ][0] = x[i]
        dataset[i*160 + j ][1] = y[j]
        dataset[i*160 + j ][2] = random_input[i]


        
        





#==================== Plot the surface ================
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, z3d, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()


    












#------------------create FCM----------------------------
#======================================GMDH==========================================

Dataset1 , Dataset2 = train_test_split(dataset, test_size=0.5, random_state=42)

for k in range(2,10):
    #-------part1 for GMDH --------
    fcm = FCM(n_clusters=k)
    fcm.fit(Dataset1)
    GT2 = fcm.predict(Dataset1)
    labels1 = fcm.predict(Dataset2)

    #-------part2 for GMDH ------------

    fcm = FCM(n_clusters=k)
    fcm.fit(Dataset2)
    GT1 = fcm.predict(Dataset2)
    labels2 = fcm.predict(Dataset1)

    #----------test GMDH table --------
    e = 0
    for i in range(0,len(Dataset1)):
        if GT1[i] != labels1[i] :
            e+=1

    for i in range(0,len(Dataset2)):
        if GT2[i] != labels2[i] :
            e+=1


    print('error with  ' + str(k) + ' clusters : ' + str(e) )

#-----------choose #2 cluster---------------------------------------------------














#===================create membership function=====================================
A1 = list()
X1 = list()
A2 = list()
X2 = list()



Xr_1 = list()
Xr_2 = list()
Z1   = list()
Z2   = list()


fcm = FCM(n_clusters=3)
fcm.fit(dataset)
fcm_labels = fcm.predict(dataset)
fcm_centers = fcm.centers
list_of_key =list()


for i in range(0,160*160):
    if fcm_labels[i] in list_of_key:
        pass
    else:
        list_of_key.append(fcm_labels[i])
   

X = dataset
for i in range(0,160*160):
    k = int(fcm.predict(X[i]))
    if k == list_of_key[0]:
        t = (X[i][0] - fcm_centers[k][0]) * (X[i][0] - fcm_centers[k][0]) + (X[i][1] - fcm_centers[k][1]) * (X[i][1] - fcm_centers[k][1]) + (X[i][2] - fcm_centers[k][2]) * (X[i][2] - fcm_centers[k][2]) + (X[i][3] - fcm_centers[k][3]) * (X[i][3] - fcm_centers[k][3])
        t = math.sqrt(t)
        t = 1/t
        if t > 10:
            t=10
        X1.append(X[i][0])
        A1.append(t)

        Xr_1.append(X[i][2])
        Z1.append(X[i][3])

        



    if k == list_of_key[1]:
        t = (X[i][0] - fcm_centers[k][0]) * (X[i][0] - fcm_centers[k][0]) + (X[i][1] - fcm_centers[k][1]) * (X[i][1] - fcm_centers[k][1]) + (X[i][2] - fcm_centers[k][2]) * (X[i][2] - fcm_centers[k][2])  + (X[i][3] - fcm_centers[k][3]) * (X[i][3] - fcm_centers[k][3])
        t = math.sqrt(t)
        t = 1/t
        if t > 10:
            t=10
        X2.append(X[i][0])
        A2.append(t)
        Xr_2.append(X[i][2])
        Z2.append(X[i][3])


        





plt.scatter(X1,A1)
plt.title("membership function 1 for input1")
plt.show()
plt.scatter(Z1,A1)
plt.title("membership function 1 for output")
plt.show()

plt.scatter(X2,A2)
plt.title("membership function 2 for input1")
plt.show()
plt.scatter(Z2,A2)
plt.title("membership function 2 for output")
plt.show()

plt.scatter(Xr_1,A1)
plt.title("membership function 1 for random input ")
plt.show()
plt.scatter(Z1,A1)
plt.title("membership function 1 for output")
plt.show()
plt.scatter(Xr_2,A2)
plt.title("membership function 2 for random input ")
plt.show()
plt.scatter(Z2,A2)
plt.title("membership function 2 for output")
plt.show()





       

'''
# plot result
f, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(X[:,0], X[:,1], alpha=.1)
axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')

plt.show()
fcm.fit(X)
'''




#=========================== create real membership function for modeling ==========================
#----------membership function M1--------------------
x_data = np.arange(-2, 2, 0.025)  
M1 = stats.norm.pdf(x_data, -0.25, 0.5)
plt.plot(x_data, M1)
plt.title("membership function M1")
plt.show()

#----------membership function M2--------------------
x_data = np.arange(-2, 2, 0.025)  
M2 = stats.norm.pdf(x_data, 0.25, 0.5)
plt.plot(x_data, M2)
plt.title("membership function M2")
plt.show()


#----------membership function R1--------------------
x_data = np.arange(-2, 1, 0.025)
t = 0 ;
R1 = list()
for i in range(0,60):
    R1.append(t)
    t+=0.0166

for i in range(60,120):
    R1.append(t)
    t-=0.0166
      
plt.plot(x_data, R1)
plt.title("membership function R1")
plt.show()

for i in range(0,40):
    R1.append(0)



#----------membership function R2--------------------
x_data = np.arange(-1, 2, 0.025)
t = 0 ;
R2 = list()
for i in range(0,60):
    R2.append(t)
    t+=0.0166

for i in range(60,120):
    R2.append(t)
    t-=0.0166

plt.plot(x_data, R2)
plt.title("membership function R1")
plt.show()

temp = R2
R2 = list()
for i in range(0,40):
    R2.append(0)
for i in range(40,160):
    R2.append(temp[i-40])
    



#=================create output with x1 ===========================
y_p = list()
for i in range(0,160):
    for j in range(0,160):
        output = ( M1[i] * 0.8 + M2[i] * 0.8 )/2
        y_p.append(output)


e_model = 0
for i in range(0,160*160):
    temp = ( dataset[i][3] - y_p[i] ) * ( dataset[i][3] - y_p[i] )
    e_model += temp


print('error of modeling with x : ' + str(e_model))





#=================create output with random input ===========================
y_pr = list()
for i in range(0,160):
    for j in range(0,160):
        if x[i] <= -1 :
            output = 0.8 * R1[i]
            y_pr.append(output)
        if x[i] > -1 and x[i] < 1 :
            output = 0.4 * R1[i] + 0.4 * R2[i]
            y_pr.append(output)
        if x[i] >= 1 :
            output =  0.8 * R2[i]
            y_pr.append(output)
            
        



e_model = 0
for i in range(0,160*160):
    temp =  ( dataset[i][3] - y_pr[i] ) * ( dataset[i][3] - y_pr[i] )
    e_model += temp


print('error of modeling with random input : ' + str(e_model))



y_p = np.array(y_p,float)
y_p = np.reshape(y_p , (160,160) )
x_plot , y_plot = np.meshgrid(x, y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x_plot, y_plot, y_p, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()







#=================create output with x1 and x2 ===========================
y_p = list()
for i in range(0,160):
    for j in range(0,160):
        output = (M1[i] * 0.8 + M2[i] * 0.8 + M1[j] * 0.8 )/6.4 #M2[j] * 0.8
        y_p.append(output)


e_model = 0
for i in range(0,160*160):
    temp = ( dataset[i][3] - y_p[i] ) * ( dataset[i][3] - y_pr[i] )
    e_model += temp


print('error of modeling with x1 and x2 : ' + str(e_model))


y_p = np.array(y_p,float)
y_p = np.reshape(y_p , (160,160) )
x_plot , y_plot = np.meshgrid(x, y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x_plot, y_plot, y_p, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()







#=================create output with x1 and x2 and R1===========================
y_p = list()
for i in range(0,160):
    for j in range(0,160):
        output = (M1[i] * 0.8 + M2[i] * 0.8 + M1[j] * 0.8 + M2[j] * 0.8)/6.4 
        y_p.append(output)


e_model = 0
for i in range(0,160*160):
    temp = ( dataset[i][3] - y_p[i] ) * ( dataset[i][3] - y_pr[i] )
    e_model += temp


print('error of modeling with x1 and x2 and R1 : ' + str(e_model))


y_p = np.array(y_p,float)
y_p = np.reshape(y_p , (160,160) )
x_plot , y_plot = np.meshgrid(x, y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x_plot, y_plot, y_p, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()
















