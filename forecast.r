total <- read.table("/Users/AswinAk/Documents/CSC591/Projects/Capstone/Deep/timeseries_toronto_8years.txt")
pos <- read.table("/Users/AswinAk/Documents/CSC591/Projects/Capstone/Deep/timeseries_toronto_8years_pos.txt")
neg <- read.table("/Users/AswinAk/Documents/CSC591/Projects/Capstone/Deep/timeseries_toronto_8years_neg.txt")

pSeries <- ts(pos,frequency = 12, start = c(2009,1))
nSeries <- ts(neg,frequency = 12, start = c(2009,1))
tSeries <- ts(total,frequency = 12, start = c(2009,1))


#Ploting the three timeseries together
plot.ts(tSeries)
lines(pSeries,col= "green")
lines(nSeries,col= "red")

#Decomposing positive review count
decomposedPos <-decompose(pSeries)
plot(decomposedPos)

#Decomposing total review count
decomposedTotal <-decompose(tSeries)
plot(decomposedTotal)

#Decomposing negative review count
decomposedNeg <-decompose(nSeries)
plot(decomposedNeg)


#From the decomposition plots we can see that we have an additive model with increasing trend and seasonality
#So we can use Holt Winters Exponential smoothing to make short term forcasts
tSeriesForcast<- HoltWinters(tSeries)
tSeriesForcastFuture<- forecast.HoltWinters(tSeriesForcast,h = 12)
plot.forecast(tSeriesForcastFuture)
tSeriesForcastFuture


#          Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
# Feb 2017       3227.977 2844.033 3611.920 2640.786 3815.167
# Mar 2017       3374.939 2872.129 3877.748 2605.958 4143.920
# Apr 2017       3306.117 2707.599 3904.634 2390.764 4221.470
# May 2017       3486.023 2805.121 4166.926 2444.673 4527.374
# Jun 2017       3295.535 2541.192 4049.878 2141.867 4449.203
# Jul 2017       3865.213 3043.971 4686.455 2609.232 5121.194
# Aug 2017       3608.087 2724.999 4491.174 2257.521 4958.652
# Sep 2017       3216.347 2275.470 4157.223 1777.400 4655.293
# Oct 2017       3246.940 2251.624 4242.256 1724.736 4769.144
# Nov 2017       3453.569 2406.641 4500.498 1852.431 5054.708
# Dec 2017       3302.200 2206.087 4398.313 1625.840 4978.560
# Jan 2018       3650.855 2507.671 4794.039 1902.506 5399.204

pSeriesForcast<- HoltWinters(pSeries)
pSeriesForcastFuture<- forecast.HoltWinters(pSeriesForcast,h = 12)
plot.forecast(pSeriesForcastFuture)
pSeriesForcastFuture

#          Point Forecast    Lo 80    Hi 80     Lo 95    Hi 95
# Feb 2017       2018.203 1770.898 2265.507 1639.9832 2396.422
# Mar 2017       2109.916 1787.192 2432.641 1616.3515 2603.481
# Apr 2017       2071.620 1688.028 2455.212 1484.9666 2658.274
# May 2017       2162.073 1726.029 2598.118 1495.2008 2828.946
# Jun 2017       2032.692 1549.860 2515.524 1294.2648 2771.120
# Jul 2017       2342.271 1816.801 2867.740 1538.6344 3145.907
# Aug 2017       2244.314 1679.416 2809.212 1380.3766 3108.251
# Sep 2017       2028.912 1427.163 2630.661 1108.6164 2949.207
# Oct 2017       2047.148 1410.679 2683.618 1073.7522 3020.544
# Nov 2017       2146.968 1477.576 2816.360 1123.2219 3170.714
# Dec 2017       2050.134 1349.365 2750.903  978.4004 3121.867
# Jan 2018       2305.750 1574.950 3036.550 1188.0881 3423.413

nSeriesForcast<- HoltWinters(nSeries)
nSeriesForcastFuture<- forecast.HoltWinters(nSeriesForcast,h = 12)
plot.forecast(nSeriesForcastFuture)
nSeriesForcastFuture

#          Point Forecast    Lo 80     Hi 80    Lo 95     Hi 95
# Feb 2017       682.9963 589.9535  776.0390 540.6997  825.2928
# Mar 2017       721.7484 605.0585  838.4384 543.2866  900.2103
# Apr 2017       704.0368 567.7425  840.3311 495.5926  912.4809
# May 2017       775.1451 621.7315  928.5587 540.5192 1009.7709
# Jun 2017       724.8187 556.0132  893.6243 466.6529  982.9846
# Jul 2017       893.6617 710.7548 1076.5686 613.9298 1173.3936
# Aug 2017       769.3880 573.3918  965.3842 469.6377 1069.1383
# Sep 2017       686.2884 478.0239  894.5529 367.7753 1004.8015
# Oct 2017       668.1109 448.2617  887.9602 331.8805 1004.3414
# Nov 2017       678.3505 447.4971  909.2039 325.2907 1031.4103
# Dec 2017       682.7675 441.4111  924.1238 313.6448 1051.8902
# Jan 2018       710.1241 458.7032  961.5450 325.6090 1094.6392


