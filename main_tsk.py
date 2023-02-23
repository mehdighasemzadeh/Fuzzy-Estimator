import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random



from sklearn.linear_model import LinearRegression

x = np.linspace(-2, 2, 160)
y = np.linspace(-2, 2, 160)
x1 = np.multiply(x, x)
y1 = np.multiply(y, y)
z = np.sinc(x1)




z3d = np.zeros((160,160),float)
z3d_p = np.zeros((1,160*160),float)
for i in range(0,160):
    for j in range(0,160):
        z3d[i][j] = np.sinc(x1[i] + y1[j])





# Plot the surface.
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, z3d, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()





#------------------create membership function -----------------------------

    
y_train = list()
x_train = list()

for j in range(0,160):
    for i in range(0,70):
        y_train.append(z3d[j][i])
for i in range(0,70):
    for j in range(0,160):
        x_train.append(x[i])



x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

reg = LinearRegression().fit(x_train,y_train)
a1 = reg.coef_
b1 = reg.intercept_
y_p = reg.predict(x_train)
e = 0
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])
    
plt.scatter(x_train,y_train)
plt.plot(x_train,y_p ,'r')
plt.show()
for i in range(0,160*70):
    z3d_p[0][i] = y_p[i]

#----------2-------------------------
y_train = list()
x_train = list()

for j in range(0,160):
    for i in range(90,160):
        y_train.append(z3d[j][i])
for i in range(90,160):
    for j in range(0,160):
        x_train.append(x[i])



x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

reg = LinearRegression().fit(x_train,y_train)
a2 = reg.coef_
b2 = reg.intercept_
y_p = reg.predict(x_train)
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])
plt.scatter(x_train,y_train)
plt.plot(x_train,y_p ,'r')
plt.show()
for i in range(160*90,160*160):
    z3d_p[0][i] = y_p[i - 160*90]


#---------------------3---------------------

y_train = list()
x_train = list()

for j in range(0,160):
    for i in range(70,90):
        y_train.append(z3d[j][i])
for i in range(70,90):
    for j in range(0,160):
        x_train.append(x[i])
y_p = list()
for i in range(0,3200):
    y_p.append(float(((x[90] - x_train[i]) * (a1*x_train[i] + b1) + (x_train[i] - x[70]) * (a2*x_train[i] + b2))/(x[90]-x[70]))) 
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])
plt.scatter(x_train,y_train)
plt.plot(x_train,y_p ,'r')
plt.show()

for i in range(160*70,90*160):
    z3d_p[0][i] = y_p[i - 160*70]
print('error of modeling with x1 : ' + str(e))

z3d_p = np.reshape(z3d_p, (160, 160))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, z3d_p, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()



#=============x11============================
yp1 = list()
x11 = np.linspace(-2, 2, 480)
y11 = np.linspace(-2, 2, 480)
for i in range(0,160):
    yp1.append(y_p[i])
    
for i in range(0,160):
    yp1.append(1.1*y_p[i])

for i in range(0,160):
    yp1.append(y_p[i])




z3d_p1 = np.zeros((480,480),float)
for i in range(0,480):
    for j in range(0,480):
        z3d_p1[i][j] = yp1[i]
        
        

   
X11, Y11 = np.meshgrid(x11, y11)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X11, Y11, z3d_p1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()




















#------------------create random input -----------------------------

x=list()

import random
for i in range(0,160*160):
    x.append(4* random.random() - 2)

x.sort()

y_train = list()
x_train = list()

for j in range(0,160):
    for i in range(0,70):
        y_train.append(z3d[j][i])

for i in range(0,160*70):
    x_train.append(x[i])



x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

reg = LinearRegression().fit(x_train,y_train)
a1 = reg.coef_
b1 = reg.intercept_
y_p = reg.predict(x_train)
e = 0
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])
    
plt.scatter(x_train,y_train)
plt.plot(x_train,y_p ,'r')
plt.show()


#----------2-------------------------
y_train = list()
x_train = list()

for j in range(0,160):
    for i in range(90,160):
        y_train.append(z3d[j][i])
for i in range(90*160,160*160):
        x_train.append(x[i])



x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

reg = LinearRegression().fit(x_train,y_train)
a2 = reg.coef_
b2 = reg.intercept_
y_p = reg.predict(x_train)
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])
plt.scatter(x_train,y_train)
plt.plot(x_train,y_p ,'r')
plt.show()

#---------------------3---------------------

y_train = list()
x_train = list()
#e+=50
for j in range(0,160):
    for i in range(70,90):
        y_train.append(z3d[j][i])
for i in range(70*160,90*160):
        x_train.append(x[i])
y_p = list()

for i in range(0,3200):
    y_p.append(float(((x[90] - x_train[i]) * (a1*x_train[i] + b1) + (x_train[i] - x[70]) * (a2*x_train[i] + b2))/(x[90]-x[70]))) 
for i in range(0,len(x_train)):
    e += abs(y_train[i] - y_p[i]) 
plt.scatter(x_train,y_train)
plt.plot(x_train,y_p ,'r')
plt.show()

print('error of modeling with random input : ' + str(e))












#-----------------------step2--------------------------
x = np.linspace(-2, 2, 160)
y = np.linspace(-2, 2, 160)

#for i in range(0,160):
#    x[i] = x[i] + random.random() / 2.5
#   y[i] = y[i] + random.random() / 2.5
#----------------------1----------------------------------
x_train = np.zeros((70*70,2),float)
y_train = np.zeros((70*70,1),float)

z_out = np.zeros((160,160),float)

for i in range(0,70):
    for j in range(0,70):
        x_train[i][0] = x[i]
        x_train[i][1] = y[j]
        y_train[i*70 + j] =  np.sinc(x1[i] + y1[j])


reg = LinearRegression().fit(x_train,y_train)
a1 = reg.coef_
b1 = reg.intercept_
y_p = reg.predict(x_train)

e=0
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])


for i in range(0,70):
    for j in range(0,70):
        z_out[i][j] = ( y_p[i*70 + j])



#-------------------2-------------------------------
x_train = np.zeros((70*70,2),float)
y_train = np.zeros((70*70,1),float)

for i in range(0,70):
    for j in range(90,160):
        x_train[i][0] = x[i]
        x_train[i][1] = y[j]
        y_train[i*70 + j-90] =  np.sinc(x1[i] + y1[j])


reg = LinearRegression().fit(x_train,y_train)
a2 = reg.coef_
b2 = reg.intercept_
y_p = reg.predict(x_train)
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])   



for i in range(0,70):
    for j in range(90,160):
        z_out[i][j] = y_p[i*70 + j - 90]

#-------------------3-------------------------------
x_train = np.zeros((70*70,2),float)
y_train = np.zeros((70*70,1),float)

for i in range(90,160):
    for j in range(0,70):
        x_train[i][0] = x[i]
        x_train[i][1] = y[j]
        y_train[(i-90)*70 + j] =  np.sinc(x1[i] + y1[j])


reg = LinearRegression().fit(x_train,y_train)
a3 = reg.coef_
b3 = reg.intercept_
y_p = reg.predict(x_train)
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])


for i in range(90,160):
    for j in range(0,70):
        z_out[i][j] = y_p[(i-90)*70 + j]



#-------------------4-------------------------------
x_train = np.zeros((70*70,2),float)
y_train = np.zeros((70*70,1),float)

for i in range(90,160):
    for j in range(90,160):
        x_train[i][0] = x[i]
        x_train[i][1] = y[j]
        y_train[(i-90)*70 + j-90] =  np.sinc(x1[i] + y1[j])


reg = LinearRegression().fit(x_train,y_train)
a4 = reg.coef_
b4 = reg.intercept_
y_p = reg.predict(x_train)
for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])


for i in range(90,160):
    for j in range(90,160):
        z_out[i][j] = y_p[(i-90)*70 + j - 90]



#-------------------5-------------------------------
x_train = np.zeros((20*20,2),float)
y_train = np.zeros((20*20,1),float)

y_p = np.zeros((20*20,1),float)

for i in range(70,90):
    for j in range(70,90):
        x_train[i][0] = x[i]
        x_train[i][1] = y[j]
        y_train[(i-70)*20 + j-70] =  np.sinc(x1[i] + y1[j])

for i in range(0,400):
    c1 = max( (x[90] - x_train[i][0]) ,(y[90] - x_train[i][1])  )
    c2 = max( (x[90] - x_train[i][0]) ,(x_train[i][1]  - y[70]) )
    c3 = max( (x_train[i][0] - x[70]) ,(y[90] - x_train[i][1])  )
    c4 = max( (x_train[i][0] - x[70]) ,(x_train[i][1]  - y[70]) )
    t = 1 * ( a1[0][0]*x_train[i][0] + a1[0][1]*x_train[i][1] + b1  ) + 1 * ( a2[0][0]*x_train[i][0] + a2[0][1]*x_train[i][1] + b2  )  + 1 * ( a3[0][0]*x_train[i][0] + a3[0][1]*x_train[i][1] + b3  )    + 1 * ( a4[0][0]*x_train[i][0] + a4[0][1]*x_train[i][1] + b4  )  
    y_p[i] = t / ((x[90] - x[70]))


for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])


for i in range(70,90):
    for j in range(70,90):
        z_out[i][j] = y_p[(i-70)*20 + j-70]







#-------------------6-------------------------------
x_train = np.zeros((20*70,2),float)
y_train = np.zeros((20*70,1),float)

y_p = np.zeros((20*70,1),float)

for i in range(70,90):
    for j in range(0,70):
        x_train[i][0] = x[i]
        x_train[i][1] = y[j]
        y_train[(i-70)*20 + j] =  np.sinc(x1[i] + y1[j])


for i in range(0,1400):
    c1 = max( (x[90] - x_train[i][0]) ,(y[90] - x_train[i][1])  )
    c2 = max( (x[90] - x_train[i][0]) ,(x_train[i][1]  - y[70]) )
    c3 = max( (x_train[i][0] - x[70]) ,(y[90] - x_train[i][1])  )
    c4 = max( (x_train[i][0] - x[70]) ,(x_train[i][1]  - y[70]) )
    t = 1 * ( a1[0][0]*x_train[i][0] + a1[0][1]*x_train[i][1] + b1  ) + 1 * ( a2[0][0]*x_train[i][0] + a2[0][1]*x_train[i][1] + b2  )  + 1 * ( a3[0][0]*x_train[i][0] + a3[0][1]*x_train[i][1] + b3  )    + 1 * ( a4[0][0]*x_train[i][0] + a4[0][1]*x_train[i][1] + b4  )  
    y_p[i] = t / ((x[90] - x[70]))


for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])


for i in range(70,90):
    for j in range(0,70):
        z_out[i][j] = y_p[(i-70)*20 + j]





#-------------------7-------------------------------
x_train = np.zeros((20*70,2),float)
y_train = np.zeros((20*70,1),float)

y_p = np.zeros((20*70,1),float)

for i in range(70,90):
    for j in range(90,160):
        x_train[i][0] = x[i]
        x_train[i][1] = y[j]
        y_train[(i-70)*20 + j-90] =  np.sinc(x1[i] + y1[j])


for i in range(0,1400):
    c1 = max( (x[90] - x_train[i][0]) ,(y[90] - x_train[i][1])  )
    c2 = max( (x[90] - x_train[i][0]) ,(x_train[i][1]  - y[70]) )
    c3 = max( (x_train[i][0] - x[70]) ,(y[90] - x_train[i][1])  )
    c4 = max( (x_train[i][0] - x[70]) ,(x_train[i][1]  - y[70]) )
    t = 1 * ( a1[0][0]*x_train[i][0] + a1[0][1]*x_train[i][1] + b1  ) + 1 * ( a2[0][0]*x_train[i][0] + a2[0][1]*x_train[i][1] + b2  )  + 1 * ( a3[0][0]*x_train[i][0] + a3[0][1]*x_train[i][1] + b3  )    + 1 * ( a4[0][0]*x_train[i][0] + a4[0][1]*x_train[i][1] + b4  )  
    y_p[i] = t / ((x[90] - x[70]))


for i in range(0,len(x_train)):
    e += (y_train[i] - y_p[i]) * (y_train[i] - y_p[i])


for i in range(70,90):
    for j in range(90,160):
        z_out[i][j] = y_p[(i-70)*20 + j-160]

















for i in range(70,90):
    for j in range(0,70):
        z_out[i][j]=0.10
t = 0
for i in range(60,70):
    t += 0.01
    for j in range(0,70):
        z_out[i][j]=t
        
for i in range(90,100):
    t -= 0.01
    for j in range(0,70):
        z_out[i][j]=t



for i in range(70,90):
    for j in range(90,160):
        z_out[i][j]=0.10
t = 0
for i in range(60,70):
    t += 0.01
    for j in range(90,160):
        z_out[i][j]=t
        
for i in range(90,100):
    t -= 0.01
    for j in range(90,160):
        z_out[i][j]=t



for j in range(70,90):
    for i in range(90,160):
        z_out[i][j]=0.10
t = 0
for j in range(60,70):
    t += 0.01
    for i in range(90,160):
        z_out[i][j]=t
        
for j in range(90,100):
    t -= 0.01
    for i in range(90,160):
        z_out[i][j]=t





for j in range(70,90):
    for i in range(0,70):
        z_out[i][j]=0.10
t = 0
for j in range(60,70):
    t += 0.01
    for i in range(0,70):
        z_out[i][j]=t
        
for j in range(90,100):
    t -= 0.01
    for i in range(0,70):
        z_out[i][j]=t




for i in range(75,85):
    for j in range(75,85):
        z_out[i][j]=0.15

t = 0.1
for i in range(70,75):
    t += 0.01
    for j in range(70,75):
        z_out[i][j]=t
        
for i in range(85,90):
    t -= 0.01
    for j in range(85,90):
        z_out[i][j]=t

for j in range(0,160):
    for i in range(0,160):
        if z_out[i][j] >0.15:
            z_out[i][j]=0.15
        if z_out[i][j] < -0.2:
            z_out[i][j] = -0.2

print('error of modeling with x1 and x2 : ' + str(e))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, z_out, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()
