# Fuzzy-Estimator

We estimate a 2D-Sinc function using fuzzy models (Takagi-Sugeno, Sugeno, and Active Learning Method) and also the effect of noise on these model is revealed, while running code all the implementation details are shown.

## 2D-Sinc

<img src="pic/2D-Sinc.png" width="300" height="300">

## ALM

Estimation of 2D-Sinc using ALM

<img src="pic/ALM.png" width="300" height="300">

```
python3 alm.py
```

## Takagi-Sugeno

Estimation of 2D-Sinc using Takagi-Sugeno

<img src="pic/TS.png" width="300" height="300">

```
python3 main_tsk.py
```



## Sugeno 

Estimation of 2D-Sinc using Sugeno

<img src="pic/S.png" width="300" height="300">

```
python3 main_sy.py
```


## Results


```
+-----------------------------+
|  Model          |  Error    |
+-----------------------------+
|  ALM            |  0.05     |
+-----------------------------+
|  Takagi-Sugeno  |  0.13     |
+-----------------------------+
|  Sugeno         |  0.16     |
+-----------------------------+
```





