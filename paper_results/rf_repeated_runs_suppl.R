### run rf on multiple train/test splits
library(dplyr)
library(ranger)
library(caret)

# set working dir to script path, works in RStudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

### read in data
predictors<-read.csv("./data/tsms_biotic_0.3pCO2_predictors.csv")
dim(predictors)
# [1] 174 105

npdr_scores<-read.csv("./npdr_dist_output/lasso_npdr_urfp_features.csv")
npdr_feat<-as.character(npdr_scores$features)
npdr_ind<-which(colnames(predictors) %in% c(npdr_feat,"biotic"))

npdr_predictors<-predictors[,npdr_ind]
dim(npdr_predictors)
# [1] 174  10

# split into training and testing 
inTrain<-createDataPartition(
  # outcome data
  y=npdr_predictors$biotic,
  # percent data in training set
  p=0.8,
  list=F)

# use train ind values from before
train.dat<-npdr_predictors[inTrain,]
table(train.dat$biotic)
# abiotic  biotic 
# 89      51

test.dat<-npdr_predictors[-inTrain,]
table(test.dat$biotic)
# abiotic  biotic 
# 22      12 

# tuned parameters from before
#   mtry  splitrule min.node.size
# 4    2 extratrees             8
# max_trees = 6000 

# create the model 
rfNpdrFinal.fit <- ranger(biotic ~ ., train.dat, keep.inbag = TRUE,
                          num.trees=6000, # match full variable num trees
                          mtry=2, 
                          importance="permutation", splitrule = "extratrees",
                          min.node.size=8,
                          class.weights = as.numeric(c(1/table(train.dat$biotic))),
                          scale.permutation.importance = T,
                          local.importance = T, num.threads=4)
sorted.imp<-sort(rfNpdrFinal.fit$variable.importance,decreasing=T)
sorted.imp
### RUN 0
#  diff2_acf1 fluctanal_prop_r1  avg_rR45CO244CO2  avg_rd45CO244CO2 
# 0.07963442        0.07558188        0.05111672        0.02488842 
# avg_d45CO244CO2   avg_R45CO244CO2     time_kl_shift  walker_propcross 
# 0.02477735        0.02416445        0.02240897        0.01643164 
# sd_d18O13C 
# 0.01101773 

### RUN 1
# fluctanal_prop_r1        diff2_acf1  avg_rR45CO244CO2   avg_d45CO244CO2 
#        0.08768574        0.06562483        0.04355258        0.02663238 
# avg_R45CO244CO2  avg_rd45CO244CO2     time_kl_shift  walker_propcross 
#     0.02558472        0.02520114        0.01984771        0.01335456 
# sd_d18O13C 
# 0.01218919 

### RUN 2
# diff2_acf1 fluctanal_prop_r1  avg_rR45CO244CO2   avg_R45CO244CO2 
# 0.08841422        0.06792757        0.04857144        0.02245157 
# avg_d45CO244CO2  avg_rd45CO244CO2     time_kl_shift        sd_d18O13C 
# 0.02195227        0.02185946        0.01827227        0.01724791 
# walker_propcross 
# 0.01551547 

### RUN 3
#  diff2_acf1 fluctanal_prop_r1  avg_rR45CO244CO2  avg_rd45CO244CO2 
# 0.08195359        0.06946031        0.04337504        0.02347504 
# avg_R45CO244CO2   avg_d45CO244CO2     time_kl_shift  walker_propcross 
# 0.02346776        0.02305151        0.01906490        0.01636882 
# sd_d18O13C 
# 0.01157385

### RUN 4
#  diff2_acf1 fluctanal_prop_r1  avg_rR45CO244CO2     time_kl_shift 
# 0.080975248       0.067008782       0.038893309       0.024932553 
# avg_R45CO244CO2   avg_d45CO244CO2  avg_rd45CO244CO2  walker_propcross 
# 0.020769790       0.020263474       0.020124822       0.013753962 
# sd_d18O13C 
# 0.006919722 

### RUN 5
# fluctanal_prop_r1        diff2_acf1  avg_rR45CO244CO2   avg_d45CO244CO2 
# 0.07765822        0.07438857        0.04561438        0.02209701 
# avg_rd45CO244CO2   avg_R45CO244CO2     time_kl_shift  walker_propcross 
# 0.02083046        0.02053317        0.01489981        0.01387781 
# sd_d18O13C 
# 0.01302653 


rfNpdrFinal.fit$confusion.matrix
### RUN 0
#  predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        6     45

### RUN 1
#   predicted
# true      abiotic biotic
# abiotic      82      7
# biotic        7     44

### RUN 2
# predicted
# true      abiotic biotic
# abiotic      85      4
# biotic        5     46


### RUN 3
# predicted
# true      abiotic biotic
# abiotic      78     11
# biotic        4     47


### RUN 4
# predicted
# true      abiotic biotic
# abiotic      78     11
# biotic        6     45

### RUN 5
# predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        5     46


rfNpdrFinal.fit$prediction.error

### RUN 0
# [1] 0.1

### RUN 1 
# [1] 0.1

### RUN 2
# [1] 0.06428571

### RUN 3
# [1] 0.1071429

### RUN 4
# [1] 0.1214286

### RUN 5
# [1] 0.09285714

1-rfNpdrFinal.fit$prediction.error
### RUN 0
# [1] 0.9

### RUN 1
# [1] 0.9

### RUN 2
# [1] 0.9357143

### RUN 3
# [1] 0.8928571

### RUN 4
# [1] 0.8785714

### RUN 5
# [1] 0.9071429

# test the model
predNpdrFinal.test<-predict(rfNpdrFinal.fit,data=test.dat)


confusionMatrix(predNpdrFinal.test$predictions,test.dat$biotic)
### RUN 0 
#  Reference
# Prediction abiotic biotic
# abiotic      21      0
# biotic        1     12
# 
# Accuracy : 0.9706          
# 95% CI : (0.8467, 0.9993)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 7.297e-06       
# 
# Kappa : 0.9368          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9545          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.9231          
#              Prevalence : 0.6471          
#          Detection Rate : 0.6176          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9773          
#                                           
#        'Positive' Class : abiotic  

### RUN 1
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      21      1
# biotic        1     11
# 
# Accuracy : 0.9412          
# 95% CI : (0.8032, 0.9928)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 6.961e-05       
# 
# Kappa : 0.8712          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9545          
#             Specificity : 0.9167          
#          Pos Pred Value : 0.9545          
#          Neg Pred Value : 0.9167          
#              Prevalence : 0.6471          
#          Detection Rate : 0.6176          
#    Detection Prevalence : 0.6471          
#       Balanced Accuracy : 0.9356          
#                                           
#        'Positive' Class : abiotic  

### RUN 2
### Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      20      1
# biotic        2     11
# 
# Accuracy : 0.9118          
# 95% CI : (0.7632, 0.9814)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 0.0004321       
# 
# Kappa : 0.8104          
# 
# Mcnemar's Test P-Value : 1.0000000       
#                                           
#             Sensitivity : 0.9091          
#             Specificity : 0.9167          
#          Pos Pred Value : 0.9524          
#          Neg Pred Value : 0.8462          
#              Prevalence : 0.6471          
#          Detection Rate : 0.5882          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9129          
#                                           
#        'Positive' Class : abiotic  


### RUN 3
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      22      0
# biotic        0     12
# 
# Accuracy : 1          
# 95% CI : (0.8972, 1)
# No Information Rate : 0.6471     
# P-Value [Acc > NIR] : 3.733e-07  
# 
# Kappa : 1          
# 
# Mcnemar's Test P-Value : NA         
#                                      
#             Sensitivity : 1.0000     
#             Specificity : 1.0000     
#          Pos Pred Value : 1.0000     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.6471     
#          Detection Rate : 0.6471     
#    Detection Prevalence : 0.6471     
#       Balanced Accuracy : 1.0000     
#                                      
#        'Positive' Class : abiotic  

### RUN 4
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      22      0
# biotic        0     12
# 
# Accuracy : 1          
# 95% CI : (0.8972, 1)
# No Information Rate : 0.6471     
# P-Value [Acc > NIR] : 3.733e-07  
# 
# Kappa : 1          
# 
# Mcnemar's Test P-Value : NA         
#                                      
#             Sensitivity : 1.0000     
#             Specificity : 1.0000     
#          Pos Pred Value : 1.0000     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.6471     
#          Detection Rate : 0.6471     
#    Detection Prevalence : 0.6471     
#       Balanced Accuracy : 1.0000     
#                                      
#        'Positive' Class : abiotic   

### RUN 5
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      21      0
# biotic        1     12
# 
# Accuracy : 0.9706          
# 95% CI : (0.8467, 0.9993)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 7.297e-06       
# 
# Kappa : 0.9368          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9545          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.9231          
#              Prevalence : 0.6471          
#          Detection Rate : 0.6176          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9773          
#                                           
#        'Positive' Class : abiotic  


###########
# function used for feature selection knn
knn=knnSURF.balanced(train.dat$biotic, 
                     sd.frac = .5)
# [1] 30

#########
## full variable space RF runs 
# use parameters from CV tuning
#      mtry  splitrule min.node.size
# 1019   15 extratrees             9
#   maxtrees Accuracy
# 4     6000 0.857827

# split into training and testing 
inRFTrain<-createDataPartition(
  # outcome data
  y=predictors$biotic,
  # percent data in training set
  p=0.8,
  list=F)

rf_train.dat<-predictors[inRFTrain,]
rf_test.dat<-predictors[-inRFTrain,]
dim(rf_train.dat)
# [1] 140 105
dim(rf_test.dat)
# [1]  34 105
table(rf_train.dat$biotic)
# abiotic  biotic 
# 89      51 
table(rf_test.dat$biotic)
# abiotic  biotic 
# 22      12 

### run a final model
rfFullFinal.fit <- ranger(biotic ~ ., rf_train.dat, keep.inbag = TRUE,
                          num.trees=6000, mtry=15, 
                          importance="permutation", splitrule = "extratrees",
                          min.node.size=9,
                          class.weights = as.numeric(c(1/table(rf_train.dat$biotic))),
                          scale.permutation.importance = T,
                          local.importance = T, num.threads=4)
sorted.imp<-sort(rfFullFinal.fit$variable.importance,decreasing=T)
rf.feat<-sorted.imp[1:9] # top nine to compare to LASSO-NPDR-URFP features
rf.feat
### RUN 0
#   max_kl_shift  fluctanal_prop_r1 localsimple_taures   avg_rR45CO244CO2 
# 0.030928329        0.028698109        0.018544878        0.013452959 
# time_level_shift       diff2x_pacf5       diff1x_pacf5     time_var_shift 
# 0.011030454        0.010595307        0.010337182        0.010015880 
# diff2_acf10 
# 0.009073347 

### RUN 1
# fluctanal_prop_r1       max_kl_shift   avg_rR45CO244CO2 localsimple_taures 
# 0.031376952        0.028317225        0.017146562        0.016545861 
# time_level_shift       diff2x_pacf5     time_var_shift       diff1x_pacf5 
# 0.011137980        0.010766913        0.010726860        0.009481586 
# diff2_acf10 
# 0.008740035 

### RUN 2
# fluctanal_prop_r1       max_kl_shift   avg_rR45CO244CO2 localsimple_taures 
# 0.035808033        0.025713965        0.012536498        0.012425532 
# time_level_shift     time_var_shift       diff2x_pacf5       diff1x_pacf5 
# 0.010904421        0.010674175        0.009844713        0.009554373 
# avg_rd45CO244CO2 
# 0.009211158 

### RUN 3
# fluctanal_prop_r1       max_kl_shift localsimple_taures     time_var_shift 
# 0.034518214        0.029661113        0.018759177        0.013247775 
# time_level_shift   avg_rR45CO244CO2       diff1x_pacf5       diff2x_pacf5 
# 0.013049692        0.011206917        0.011200375        0.010411659 
# diff2_acf10 
# 0.009706108 

### RUN 4
#  fluctanal_prop_r1       max_kl_shift   avg_rR45CO244CO2 localsimple_taures 
# 0.041613308        0.028054393        0.014641395        0.014310051 
# time_level_shift     time_var_shift       diff2x_pacf5       diff1x_pacf5 
# 0.013004268        0.012509368        0.009969881        0.009188517 
# avg_rd45CO244CO2 
# 0.008908846 

### RUN 5
#  max_kl_shift  fluctanal_prop_r1 localsimple_taures   avg_rR45CO244CO2 
# 0.032852611        0.027058274        0.020443711        0.015657136 
# diff2x_pacf5       diff1x_pacf5        diff2_acf10         diff2_acf1 
# 0.013253896        0.012458358        0.010883314        0.010832910 
# time_var_shift 
# 0.009349538 

rfFullFinal.fit$confusion.matrix
### RUN 0
#   predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        9     42

### RUN 1
# predicted
# true      abiotic biotic
# abiotic      79     10
# biotic        5     46

### RUN 2
# predicted
# true      abiotic biotic
# abiotic      80      9
# biotic        7     44

### RUN 3
#   predicted
# true      abiotic biotic
# abiotic      83      6
# biotic        5     46

### RUN 4
# predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        5     46

### RUN 5
#  predicted
# true      abiotic biotic
# abiotic      83      6
# biotic        8     43

rfFullFinal.fit$prediction.error
### RUN 0
# [1] 0.1214286

### RUN 1
# [1] 0.1071429

### RUN 2
# [1] 0.1142857

### RUN 3
# [1] 0.07857143

### RUN 4
# [1] 0.09285714

### RUN 5
# [1] 0.1


1-rfFullFinal.fit$prediction.error
### RUN 0
# [1] 0.8785714

### RUN 1
# [1] 0.8928571

### RUN 2
# [1] 0.8857143

### RUN 3
# [1] 0.9214286

### RUN 4 
# [1] 0.9071429

### RUN 5
# [1] 0.9


# test the model
predFullFinal.test<-predict(rfFullFinal.fit,data=rf_test.dat)

confusionMatrix(predFullFinal.test$predictions,rf_test.dat$biotic)
### RUN 0
#   Reference
# Prediction abiotic biotic
# abiotic      21      0
# biotic        1     12
# 
# Accuracy : 0.9706          
# 95% CI : (0.8467, 0.9993)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 7.297e-06       
# 
# Kappa : 0.9368          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9545          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.9231          
#              Prevalence : 0.6471          
#          Detection Rate : 0.6176          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9773          
#                                           
#        'Positive' Class : abiotic    

### RUN 1
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      21      1
# biotic        1     11
# 
# Accuracy : 0.9412          
# 95% CI : (0.8032, 0.9928)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 6.961e-05       
# 
# Kappa : 0.8712          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9545          
#             Specificity : 0.9167          
#          Pos Pred Value : 0.9545          
#          Neg Pred Value : 0.9167          
#              Prevalence : 0.6471          
#          Detection Rate : 0.6176          
#    Detection Prevalence : 0.6471          
#       Balanced Accuracy : 0.9356          
#                                           
#        'Positive' Class : abiotic   

### RUN 2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      21      0
# biotic        1     12
# 
# Accuracy : 0.9706          
# 95% CI : (0.8467, 0.9993)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 7.297e-06       
# 
# Kappa : 0.9368          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9545          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.9231          
#              Prevalence : 0.6471          
#          Detection Rate : 0.6176          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9773          
#                                           
#        'Positive' Class : abiotic  

### RUN 3
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      20      2
# biotic        2     10
# 
# Accuracy : 0.8824         
# 95% CI : (0.7255, 0.967)
# No Information Rate : 0.6471         
# P-Value [Acc > NIR] : 0.001965       
# 
# Kappa : 0.7424         
# 
# Mcnemar's Test P-Value : 1.000000       
#                                          
#             Sensitivity : 0.9091         
#             Specificity : 0.8333         
#          Pos Pred Value : 0.9091         
#          Neg Pred Value : 0.8333         
#              Prevalence : 0.6471         
#          Detection Rate : 0.5882         
#    Detection Prevalence : 0.6471         
#       Balanced Accuracy : 0.8712         
#                                          
#        'Positive' Class : abiotic  

### RUN 4
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      19      0
# biotic        3     12
# 
# Accuracy : 0.9118          
# 95% CI : (0.7632, 0.9814)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 0.0004321       
# 
# Kappa : 0.8172          
# 
# Mcnemar's Test P-Value : 0.2482131       
#                                           
#             Sensitivity : 0.8636          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.8000          
#              Prevalence : 0.6471          
#          Detection Rate : 0.5588          
#    Detection Prevalence : 0.5588          
#       Balanced Accuracy : 0.9318          
#                                           
#        'Positive' Class : abiotic  

### RUN 5
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      17      0
# biotic        5     12
# 
# Accuracy : 0.8529          
# 95% CI : (0.6894, 0.9505)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 0.00698         
# 
# Kappa : 0.7059          
# 
# Mcnemar's Test P-Value : 0.07364         
#                                           
#             Sensitivity : 0.7727          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.7059          
#              Prevalence : 0.6471          
#          Detection Rate : 0.5000          
#    Detection Prevalence : 0.5000          
#       Balanced Accuracy : 0.8864          
#                                           
#        'Positive' Class : abiotic 



#############
#### RF runs with top RF features
rf_feat<-c("max_kl_shift","fluctanal_prop_r1","localsimple_taures",
           "avg_rR45CO244CO2", "time_level_shift", "diff2x_pacf5",      
            "diff1x_pacf5","time_var_shift","diff2_acf10")

rf.ind<-which(colnames(predictors) %in% c(rf_feat,"biotic"))
rfSel.dat<-predictors[,rf.ind]
dim(rfSel.dat)


# split into training and testing 
inRF_selTrain<-createDataPartition(
  # outcome data
  y=rfSel.dat$biotic,
  # percent data in training set
  p=0.8,
  list=F)

rfSel_train.dat<-rfSel.dat[inRF_selTrain,]
rfSel_test.dat<-rfSel.dat[-inRF_selTrain,]
dim(rfSel_train.dat)
# [1] 140 10
dim(rfSel_test.dat)
# [1]  34 10
table(rfSel_train.dat$biotic)
# abiotic  biotic 
# 89      51 
table(rfSel_test.dat$biotic)
# abiotic  biotic 
# 22      12 


# use lasso-npdr-urfp tuned params (same dims)
rfSelFinal.fit <- ranger(biotic ~ ., rfSel_train.dat, keep.inbag = TRUE,
                         num.trees=6000, mtry=2, 
                         importance="permutation", splitrule = "extratrees",
                         min.node.size=8,
                         class.weights = as.numeric(c(1/table(rfSel_train.dat$biotic))),
                         scale.permutation.importance = T,
                         local.importance = T, num.threads=4)
sorted.imp<-sort(rfSelFinal.fit$variable.importance,decreasing=T)
sorted.imp
### RUN 0 
#  max_kl_shift localsimple_taures  fluctanal_prop_r1        diff2_acf10 
# 0.10168672         0.05156252         0.05038208         0.04988826 
# diff2x_pacf5       diff1x_pacf5   time_level_shift     time_var_shift 
# 0.04824201         0.04791643         0.04709537         0.04494646 
# avg_rR45CO244CO2 
# 0.03917553 

### RUN 1
#   max_kl_shift  fluctanal_prop_r1 localsimple_taures   time_level_shift 
# 0.09215671         0.04690920         0.04610125         0.04605680 
# time_var_shift        diff2_acf10       diff2x_pacf5       diff1x_pacf5 
# 0.04508638         0.04450507         0.03997332         0.03828979 
# avg_rR45CO244CO2 
# 0.03653700 

### RUN 2
#   max_kl_shift  fluctanal_prop_r1        diff2_acf10     time_var_shift 
# 0.09489614         0.06457149         0.04790802         0.04718991 
# localsimple_taures   time_level_shift       diff1x_pacf5       diff2x_pacf5 
# 0.04716558         0.04688239         0.04511954         0.04465077 
# avg_rR45CO244CO2 
# 0.03782590 

### RUN 3
# max_kl_shift   time_level_shift  fluctanal_prop_r1     time_var_shift 
# 0.09594186         0.05908262         0.05582339         0.05380642 
# diff2_acf10       diff1x_pacf5       diff2x_pacf5 localsimple_taures 
# 0.05379012         0.05125986         0.05056185         0.04849527 
# avg_rR45CO244CO2 
# 0.03149770 

### RUN 4
# max_kl_shift   time_level_shift localsimple_taures  fluctanal_prop_r1 
# 0.08673147         0.05587984         0.05043559         0.05027966 
# time_var_shift        diff2_acf10       diff1x_pacf5       diff2x_pacf5 
# 0.04919587         0.04416372         0.04015538         0.03838809 
# avg_rR45CO244CO2 
# 0.03538967 

### RUN 5
#    max_kl_shift     time_var_shift  fluctanal_prop_r1        diff2_acf10 
# 0.09936491         0.05309214         0.05274449         0.05122048 
# time_level_shift       diff2x_pacf5       diff1x_pacf5 localsimple_taures 
# 0.04988523         0.04850749         0.04846534         0.04545977 
# avg_rR45CO244CO2 
# 0.03950269 

rfSelFinal.fit$confusion.matrix
### RUN 0
#  predicted
# true      abiotic biotic
# abiotic      77     12
# biotic        7     44

### RUN 1
# predicted
# true      abiotic biotic
# abiotic      73     16
# biotic        5     46

### RUN 2
# predicted
# true      abiotic biotic
# abiotic      75     14
# biotic        3     48

### RUN 3
# predicted
# true      abiotic biotic
# abiotic      77     12
# biotic        5     46


### RUN 4
#  predicted
# true      abiotic biotic
# abiotic      77     12
# biotic        6     45

### RUN 5
#  predicted
# true      abiotic biotic
# abiotic      76     13
# biotic        4     47

rfSelFinal.fit$prediction.error
### RUN 0
# [1] 0.1357143

### RUN 1
# [1] 0.15

### RUN 2
# [1] 0.1214286

### RUN 3
# [1] 0.1214286

### RUN 4
# [1] 0.1285714

### RUN 5
# [1] 0.1214286

1-rfSelFinal.fit$prediction.error
### RUN 0
# [1] 0.8642857

### RUN 1
# [1] 0.85

### RUN 2
# [1] 0.8785714

### RUN 3
# [1] 0.8785714

### RUN 4
# [1] 0.8714286

### RUN 5
# [1] 0.8785714

# test the model
predRFFinal.test<-predict(rfSelFinal.fit,data=rfSel_test.dat)

confusionMatrix(predRFFinal.test$predictions,rfSel_test.dat$biotic)
### RUN 0
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      20      1
# biotic        2     11
# 
# Accuracy : 0.9118          
# 95% CI : (0.7632, 0.9814)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 0.0004321       
# 
# Kappa : 0.8104          
# 
# Mcnemar's Test P-Value : 1.0000000       
#                                           
#             Sensitivity : 0.9091          
#             Specificity : 0.9167          
#          Pos Pred Value : 0.9524          
#          Neg Pred Value : 0.8462          
#              Prevalence : 0.6471          
#          Detection Rate : 0.5882          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9129          
#                                           
#        'Positive' Class : abiotic


### RUN 1
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      21      0
# biotic        1     12
# 
# Accuracy : 0.9706          
# 95% CI : (0.8467, 0.9993)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 7.297e-06       
# 
# Kappa : 0.9368          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9545          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.9231          
#              Prevalence : 0.6471          
#          Detection Rate : 0.6176          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9773          
#                                           
#        'Positive' Class : abiotic    

### RUN 2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      18      1
# biotic        4     11
# 
# Accuracy : 0.8529          
# 95% CI : (0.6894, 0.9505)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 0.00698         
# 
# Kappa : 0.6953          
# 
# Mcnemar's Test P-Value : 0.37109         
#                                           
#             Sensitivity : 0.8182          
#             Specificity : 0.9167          
#          Pos Pred Value : 0.9474          
#          Neg Pred Value : 0.7333          
#              Prevalence : 0.6471          
#          Detection Rate : 0.5294          
#    Detection Prevalence : 0.5588          
#       Balanced Accuracy : 0.8674          
#                                           
#        'Positive' Class : abiotic  


### RUN 3
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      19      1
# biotic        3     11
# 
# Accuracy : 0.8824         
# 95% CI : (0.7255, 0.967)
# No Information Rate : 0.6471         
# P-Value [Acc > NIR] : 0.001965       
# 
# Kappa : 0.7518         
# 
# Mcnemar's Test P-Value : 0.617075       
#                                          
#             Sensitivity : 0.8636         
#             Specificity : 0.9167         
#          Pos Pred Value : 0.9500         
#          Neg Pred Value : 0.7857         
#              Prevalence : 0.6471         
#          Detection Rate : 0.5588         
#    Detection Prevalence : 0.5882         
#       Balanced Accuracy : 0.8902         
#                                          
#        'Positive' Class : abiotic     


### RUN 4
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      20      1
# biotic        2     11
# 
# Accuracy : 0.9118          
# 95% CI : (0.7632, 0.9814)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 0.0004321       
# 
# Kappa : 0.8104          
# 
# Mcnemar's Test P-Value : 1.0000000       
#                                           
#             Sensitivity : 0.9091          
#             Specificity : 0.9167          
#          Pos Pred Value : 0.9524          
#          Neg Pred Value : 0.8462          
#              Prevalence : 0.6471          
#          Detection Rate : 0.5882          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9129          
#                                           
#        'Positive' Class : abiotic   

### RUN 5
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      20      1
# biotic        2     11
# 
# Accuracy : 0.9118          
# 95% CI : (0.7632, 0.9814)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 0.0004321       
# 
# Kappa : 0.8104          
# 
# Mcnemar's Test P-Value : 1.0000000       
#                                           
#             Sensitivity : 0.9091          
#             Specificity : 0.9167          
#          Pos Pred Value : 0.9524          
#          Neg Pred Value : 0.8462          
#              Prevalence : 0.6471          
#          Detection Rate : 0.5882          
#    Detection Prevalence : 0.6176          
#       Balanced Accuracy : 0.9129          
#                                           
#        'Positive' Class : abiotic  