#### biotic-abiotic feature selection with NPDR and classification using random forest
# load libraries
library(ranger)
library(reshape2)
library(dplyr)
library(npdr)
library(caret)
library(doParallel)
library(foreach)
library(glm2)
library(igraph)
library(forcats)

# set working dir to script path, works in RStudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

##################
# Part I: read in training and testing data
##################
# read in previously processed training and testing data
# training data
train.dat<-read.csv("./data/tsms_0.3p_bio_train.csv")
train.dat<-train.dat%>%select(!Analysis)
head(colnames(train.dat))
# [1] "x_acf1"      "x_acf10"     "diff1_acf1"  "diff1_acf10" "diff2_acf1" 
# [6] "diff2_acf10"
tail(colnames(train.dat))
# [1] "sd_pkArea"         "avg_d18O13C"       "sd_d18O13C"       
# [4] "avg_calib_d18O16O" "sd_calib_d18O16O"  "biotic" 
dim(train.dat)
# [1] 140 105

table(train.dat$biotic)
# abiotic  biotic 
#     89      51 

# testing data
test.dat<-read.csv("./data/tsms_0.3p_bio_test.csv")
test.dat<-test.dat%>%select(!Analysis)
dim(test.dat)
# [1]  34 105

table(test.dat$biotic)
# abiotic  biotic 
#      22      12 



##################
# Part II: unsupervised random forest proximity (URFP) distance
#################
#### unsupervised RF
# function to extract proximity from RF model
extract_proximity_oob = function(fit, olddata) {
  pred = predict(fit, olddata, type = "terminalNodes")$predictions
  prox = matrix(NA, nrow(pred), nrow(pred))
  ntree = ncol(pred)
  n = nrow(prox)
  
  if (is.null(fit$inbag.counts)) {
    stop("call ranger with keep.inbag = TRUE")
  }
  
  # Get inbag counts
  inbag = simplify2array(fit$inbag.counts)
  
  for (i in 1:n) {
    for (j in 1:n) {
      # Use only trees where both obs are OOB
      tree_idx = inbag[i, ] == 0 & inbag[j, ] == 0
      prox[i, j] = sum(pred[i, tree_idx] == pred[j, tree_idx]) / sum(tree_idx)
    }
  }
  
  prox
}

# create synthetic data for unsupervised RF
bio.synth <- as.data.frame(lapply(train.dat[,-ncol(train.dat)], 
                                  function(x) {sample(x, length(x), replace = TRUE)}))
bioSynth.dat<-rbind(data.frame(y="real",train.dat[,-ncol(train.dat)]),
                    data.frame(y="synth",bio.synth))
head(bioSynth.dat)[,1:2]
dim(bioSynth.dat)
bioSynth.dat$y<-as.factor(bioSynth.dat$y)
tail(colnames(bioSynth.dat))
rf.fit <- ranger(y ~ ., bioSynth.dat, keep.inbag = TRUE,
                 num.trees=5000, mtry=2, 
                 importance="permutation",
                 scale.permutation.importance = T,
                 local.importance = T, num.threads=4)

prox <- extract_proximity_oob(rf.fit, bioSynth.dat)[1:nrow(train.dat), 
                                                    1:nrow(train.dat)]
urfp.dist<-sqrt(1-prox)
head(urfp.dist)[,1:3]
#           [,1]      [,2]      [,3]
# [1,] 0.0000000 0.9515389 0.5895430
# [2,] 0.9515389 0.0000000 0.9614669
# [3,] 0.5895430 0.9614669 0.0000000
# [4,] 0.9252592 0.9661407 0.9298542
# [5,] 0.9262765 0.8113470 0.9212773
# [6,] 0.9639590 0.6650833 0.9660394

# write distance matrix to file (could be used in other analyses)
write.table(urfp.dist,"./npdr_dist_output/urfp_dist_bioAbio.csv",row.names=F,col.names=F,quote=F,sep=",")


##################
# Part III: LASSO-penalized NPDR feature selection with URFP distance
##################
# feature selection
start<-Sys.time()
bio_npdrURFP <- npdr::npdr("biotic", train.dat,
                           regression.type="binomial",
                           attr.diff.type="numeric-abs",
                           nbd.method="relieff",
                           nbd.metric = "precomputed",
                           external.dist=urfp.dist,
                           knn=knnSURF.balanced(train.dat$biotic, 
                                                sd.frac = .5),
                           use.glmnet = T, glmnet.alpha = 1, 
                           glmnet.lower = 0, 
                           glmnet.lam=0.00001,#"lambda.min",# use val below lambda.min
                           neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                           padj.method="bonferroni", verbose=T)
end<-Sys.time()
npdr.time<-end-start
npdr.time
# Time difference of 5.660002 secs
npdr_scores<-bio_npdrURFP %>% tibble::rownames_to_column(var = "features") %>%
  filter(scores!=0, features!="intercept")
npdr_scores
# paper result: could differ due to tuning parameter lambda 
#            features       scores
# 1   avg_R45CO244CO2 3.369468e+03
# 2  avg_rR45CO244CO2 3.891871e+02
# 3        sd_d18O13C 5.002506e+01
# 4        diff2_acf1 1.470835e+01
# 5  walker_propcross 1.101718e+01
# 6 fluctanal_prop_r1 3.235418e+00
# 7  avg_rd45CO244CO2 1.036536e-04
# 8     time_kl_shift 8.031565e-05
# 9   avg_d45CO244CO2 3.508453e-08

# save features to file 
write.table(npdr_scores,"./npdr_dist_output/lasso_npdr_urfp_features.csv",sep=",")


##################
# Part IV: parameter tuning (caret) and supervised RF in the full variable space
#############
### supervised RF parameter tuning
# use caret library to tune RF model parameters

## supervied RF for full variable space
# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(train.dat)[2]-1),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(5,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, savePredictions = T,
                        search="random",
                        allowParallel=T)
start<-Sys.time()
rf1<-train(biotic~., data=train.dat, 
           method="ranger",
           importance="permutation",
           metric="Accuracy", 
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(train.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 1.266156 hours

rf1$bestTune
#      mtry  splitrule min.node.size
# 1019   15 extratrees             9

# best accuracy
rf1$finalModel$prediction.error
# [1] 0.1071429
1-rf1$finalModel$prediction.error
# [1] 0.8928571

# saved tuned parameters
bestFull_mtry<-rf1$bestTune$mtry
bestFull_minNodeSize<-rf1$bestTune$min.node.size
bestFull_splitRule<-rf1$bestTune$splitrule


# tune the number of trees using best params from previous run
fileName<-"./tuning_output/maxtrees_out.csv" # for parallel output

tuneGrid2<-expand.grid(.mtry=bestFull_mtry,
                       .splitrule=bestFull_splitRule,
                       .min.node.size=bestFull_minNodeSize)

trControl2<-trainControl(method = "cv",number=5,
                         verboseIter = T,savePredictions = T,
                         search="random",
                         allowParallel=F # parallelize across number of trees
)


start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=train.dat,
            method="ranger",
            importance="permutation",
            metric="Accuracy", 
            trControl=trControl2, 
            tuneGrid=tuneGrid2,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(train.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 2.320716 mins

# read in file from manual tree tuning 
tree.results<-read.csv("./tuning_output/maxtrees_out.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
maxFull_acc<-tree.results[which.max(tree.results$Accuracy),]
maxFull_acc 
#   maxtrees Accuracy
# 4     6000 0.857827
bestFull_maxTree<-maxFull_acc$maxtrees

### run a final model
rfFullFinal.fit <- ranger(biotic ~ ., train.dat, keep.inbag = TRUE,
                      num.trees=bestFull_maxTree, mtry=bestFull_mtry, 
                      importance="permutation", splitrule = bestFull_splitRule,
                      min.node.size=bestFull_minNodeSize,
                      class.weights = as.numeric(c(1/table(train.dat$biotic))),
                      scale.permutation.importance = T,
                      local.importance = T, num.threads=4)
sorted.imp<-sort(rfFullFinal.fit$variable.importance,decreasing=T)
rf.feat<-sorted.imp[1:9] # save top nine for further analysis
rf.feat
#   max_kl_shift  fluctanal_prop_r1 localsimple_taures   avg_rR45CO244CO2 
# 0.030928329        0.028698109        0.018544878        0.013452959 
# time_level_shift       diff2x_pacf5       diff1x_pacf5     time_var_shift 
# 0.011030454        0.010595307        0.010337182        0.010015880 
# diff2_acf10 
# 0.009073347 
# for published results
# rf.feat<-c("max_kl_shift","fluctanal_prop_r1","localsimple_taures",
#            "avg_rR45CO244CO2","time_level_shift","diff2x_pacf5",
#            "diff1x_pacf5","time_var_shift","diff2_acf10")
rfFullFinal.fit$confusion.matrix
#   predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        9     42

rfFullFinal.fit$prediction.error
# [1] 0.1214286
1-rfFullFinal.fit$prediction.error
# [1] 0.8785714

predFullFinal.test<-predict(rfFullFinal.fit,data=test.dat)

confusionMatrix(predFullFinal.test$predictions,test.dat$biotic)
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


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfFullFinal.fit, "./model_output/bioAbio_fullVarRF_bestFit.rds")
# save predictions
fullTrain.dat<-train.dat
fullTrain.dat$pred<-rfFullFinal.fit$predictions
fullTest.dat<-test.dat
fullTest.dat$pred<-predFullFinal.test$predictions
write.table(fullTrain.dat,"./model_output/bioAbio_fullVarRF_train.csv",quote=F,row.names=F,sep=",")
write.table(fullTest.dat,"./model_output/bioAbio_fullVarRF_test.csv",quote=F,row.names=F,sep=",")


##################
# Part V: parameter tuning and supervised RF in the LASSO-NPDR-URFP variable space
#####################
#### supervised RF using LASSO-NPDR-URFP selected features
selTrain.dat<-train.dat
selTest.dat<-test.dat

npdr.feat<-as.character(npdr_scores$features)
npdr.ind<-which(colnames(train.dat) %in% c(npdr.feat,"biotic"))

selTrain.dat<-selTrain.dat[,npdr.ind]
selTest.dat<-selTest.dat[,npdr.ind]
dim(selTrain.dat)
# [1] 140  10
dim(selTest.dat)
# [1] 34 10

# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(selTrain.dat)[2]-1),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(5,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, savePredictions = T,
                        search="random",
                        allowParallel=T)
start<-Sys.time()
rf2<-train(biotic~., data=selTrain.dat, 
           method="ranger",
           importance="permutation",
           metric="Accuracy", 
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(selTrain.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 11.53106 mins

rf2$bestTune
# mtry  splitrule min.node.size
# 4    2 extratrees             8

# best accuracy
rf2$finalModel$prediction.error
# [1] 0.09285714
1-rf2$finalModel$prediction.error
# [1] 0.9071429

# saved tuned parameters
bestNpdr_mtry<-rf2$bestTune$mtry
bestNpdr_minNodeSize<-rf2$bestTune$min.node.size
bestNpdr_splitRule<-rf2$bestTune$splitrule


# tune the number of trees using best params from previous run
fileName<-"./tuning_output/maxtrees2_out.csv" # for parallel output

tuneGrid2<-expand.grid(.mtry=bestNpdr_mtry,
                       .splitrule=bestNpdr_splitRule,
                       .min.node.size=bestNpdr_minNodeSize)

trControl2<-trainControl(method = "cv",number=5,
                         verboseIter = T,savePredictions = T,
                         search="random",
                         allowParallel=F # parallelize across number of trees
)


start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=selTrain.dat,
            method="ranger",
            importance="permutation",
            metric="Accuracy", 
            trControl=trControl2, 
            tuneGrid=tuneGrid2,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(selTrain.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 1.43075 mins

tree.results<-read.csv("./tuning_output/maxtrees2_out.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
maxNpdr_acc<-tree.results[which.max(tree.results$Accuracy),]
maxNpdr_acc 
#  maxtrees  Accuracy
#1     3000 0.9078635
bestNpdr_maxTree<-6000 # match full variable since this is fewer


### run a final model
rfNpdrFinal.fit <- ranger(biotic ~ ., selTrain.dat, keep.inbag = TRUE,
                          num.trees=bestNpdr_maxTree, # match full variable num trees
                          mtry=bestNpdr_mtry, 
                          importance="permutation", splitrule = bestNpdr_splitRule,
                          min.node.size=bestNpdr_minNodeSize,
                          class.weights = as.numeric(c(1/table(selTrain.dat$biotic))),
                          scale.permutation.importance = T,
                          local.importance = T, num.threads=4)
sorted.imp<-sort(rfNpdrFinal.fit$variable.importance,decreasing=T)
sorted.imp
#  diff2_acf1 fluctanal_prop_r1  avg_rR45CO244CO2  avg_rd45CO244CO2 
# 0.07963442        0.07558188        0.05111672        0.02488842 
# avg_d45CO244CO2   avg_R45CO244CO2     time_kl_shift  walker_propcross 
# 0.02477735        0.02416445        0.02240897        0.01643164 
# sd_d18O13C 
# 0.01101773 

rfNpdrFinal.fit$confusion.matrix
#  predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        6     45

rfNpdrFinal.fit$prediction.error
# [1] 0.1

1-rfNpdrFinal.fit$prediction.error
# [1] 0.9
# full var
# [1] 0.8785714

predNpdrFinal.test<-predict(rfNpdrFinal.fit,data=selTest.dat)

confusionMatrix(predNpdrFinal.test$predictions,selTest.dat$biotic)
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


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfNpdrFinal.fit, "./model_output/bioAbio_npdrRF_bestFit.rds")
# save predictions
selTrainPred.dat<-selTrain.dat
selTrainPred.dat$pred<-rfNpdrFinal.fit$predictions
selTestPred.dat<-selTest.dat
selTestPred.dat$pred<-predNpdrFinal.test$predictions
write.table(selTrainPred.dat,"./model_output/bioAbio_npdrVarRF_train.csv",quote=F,row.names=F,sep=",")
write.table(selTestPred.dat,"./model_output/bioAbio_npdrVarRF_test.csv",quote=F,row.names=F,sep=",")


####################
# Part VI: LASSO-penalized NPDR feature selection with Manhattan 
#          distance and supervised RF
#############
#### compare previous results with NPDR Manhattan features 
start<-Sys.time()
bio_npdrMan <- npdr::npdr("biotic", train.dat,
                           regression.type="binomial",
                           attr.diff.type="numeric-abs",
                           nbd.method="relieff",
                           nbd.metric = "manhattan",
                           #external.dist=urfp.dist,
                           knn=knnSURF.balanced(train.dat$biotic, 
                                                sd.frac = .5),
                           use.glmnet = T, glmnet.alpha = 1, 
                           glmnet.lower = 0, glmnet.lam=0.00001,#"lambda.min",
                           neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                           padj.method="bonferroni", verbose=T)
end<-Sys.time()
npdr.time<-end-start
npdr.time
# Time difference of 5.964069 secs
npdrMan_scores<-bio_npdrMan %>% tibble::rownames_to_column(var = "features") %>%
  filter(scores!=0, features!="intercept")
npdrMan_scores
#      features       scores
# 1    avg_R45CO244CO2 2.176307e+04
# 2   avg_rR45CO244CO2 1.301101e+02
# 3    motiftwo_entro3 8.350563e+01
# 4         sd_d18O13C 6.810971e+01
# 5   avg_rR46CO244CO2 3.310776e+01
# 6   walker_propcross 2.924818e+01
# 7  fluctanal_prop_r1 1.270800e+01
# 8   avg_rd45CO244CO2 3.616214e-05
# 9      time_kl_shift 3.073866e-05
# 10   avg_d45CO244CO2 3.855818e-08
# save features to file 
write.table(npdrMan_scores,"./npdr_dist_output/lasso_npdr_manhattan_features.csv",sep=",")

# training and testing data using selected features 
manTrain.dat<-train.dat
manTest.dat<-test.dat
npdrMan.feat<-npdrMan_scores$features
npdrMan.ind<-which(colnames(train.dat) %in% c(npdrMan.feat,"biotic"))
manTrain.dat<-manTrain.dat[,npdrMan.ind]
manTest.dat<-manTest.dat[,npdrMan.ind]

# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(manTrain.dat)[2]-1),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(5,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, savePredictions = T,
                        search="random",
                        allowParallel=T)
start<-Sys.time()
rf3<-train(biotic~., data=manTrain.dat, 
           method="ranger",
           importance="permutation",
           metric="Accuracy", 
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(manTrain.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 11.86659 mins

rf3$bestTune
#    mtry splitrule min.node.size
# 61    2 hellinger            13

# best accuracy
rf3$finalModel$prediction.error
#  [1] 0.1428571
1-rf3$finalModel$prediction.error
# [1] 0.8571429

# saved tuned parameters
bestMan_mtry<-rf3$bestTune$mtry
bestMan_minNodeSize<-rf3$bestTune$min.node.size
bestMan_splitRule<-rf3$bestTune$splitrule


# tune the number of trees using best params from previous run
fileName<-"./tuning_output/maxtrees3_out.csv" # for parallel output

tuneGrid2<-expand.grid(.mtry=bestMan_mtry,
                       .splitrule=bestMan_splitRule,
                       .min.node.size=bestMan_minNodeSize)

trControl2<-trainControl(method = "cv",number=5,
                         verboseIter = T,savePredictions = T,
                         search="random",
                         allowParallel=F # parallelize across number of trees
)


start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=manTrain.dat,
            method="ranger",
            importance="permutation",
            metric="Accuracy", 
            trControl=trControl2, 
            tuneGrid=tuneGrid2,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(manTrain.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 1.361518 mins

tree.results<-read.csv("./tuning_output/maxtrees3_out.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
maxNpdr_acc<-tree.results[which.max(tree.results$Accuracy),]
maxNpdr_acc 
# maxtrees Accuracy
#4     6000 0.878763
bestMan_maxTree<-maxNpdr_acc$maxtrees


### run a final model
rfManFinal.fit <- ranger(biotic ~ ., manTrain.dat, keep.inbag = TRUE,
                          num.trees=bestMan_maxTree, 
                          mtry=bestMan_mtry, 
                          importance="permutation", 
                          splitrule = bestMan_splitRule,
                          min.node.size=bestMan_minNodeSize,
                          class.weights = as.numeric(c(1/table(manTrain.dat$biotic))),
                          scale.permutation.importance = T,
                          local.importance = T, num.threads=4)
sorted.imp<-sort(rfManFinal.fit$variable.importance,decreasing=T)
sorted.imp
#  fluctanal_prop_r1  avg_rR45CO244CO2  walker_propcross  avg_rR46CO244CO2 
# 0.11210375        0.05861851        0.03657668        0.03569406 
# motiftwo_entro3   avg_R45CO244CO2   avg_d45CO244CO2  avg_rd45CO244CO2 
# 0.03494062        0.02017797        0.01989203        0.01978173 
# time_kl_shift        sd_d18O13C 
# 0.01693304        0.01119451 

rfManFinal.fit$confusion.matrix
#  predicted
# true      abiotic biotic
# abiotic      82      7
# biotic       11     40

rfManFinal.fit$prediction.error
# [1] 0.1285714
1-rfManFinal.fit$prediction.error
# [1] 0.8714286

predManFinal.test<-predict(rfManFinal.fit,data=manTest.dat)

confusionMatrix(predManFinal.test$predictions,manTest.dat$biotic)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction abiotic biotic
# abiotic      20      3
# biotic        2      9
# 
# Accuracy : 0.8529          
# 95% CI : (0.6894, 0.9505)
# No Information Rate : 0.6471          
# P-Value [Acc > NIR] : 0.00698         
# 
# Kappa : 0.6718          
# 
# Mcnemar's Test P-Value : 1.00000         
#                                           
#             Sensitivity : 0.9091          
#             Specificity : 0.7500          
#          Pos Pred Value : 0.8696          
#          Neg Pred Value : 0.8182          
#              Prevalence : 0.6471          
#          Detection Rate : 0.5882          
#    Detection Prevalence : 0.6765          
#       Balanced Accuracy : 0.8295          
#                                           
#        'Positive' Class : abiotic


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfManFinal.fit, "./model_output/bioAbio_npdrMan_bestFit.rds")

# save predictions
manTrainPred.dat<-manTrain.dat
manTrainPred.dat$pred<-rfManFinal.fit$predictions
manTestPred.dat<-manTest.dat
manTestPred.dat$pred<-predManFinal.test$predictions
write.table(manTrainPred.dat,"./model_output/bioAbio_npdrManVarRF_train.csv",quote=F,row.names=F,sep=",")
write.table(manTestPred.dat,"./model_output/bioAbio_npdrManVarRF_test.csv",quote=F,row.names=F,sep=",")



##################
# Part VII: probability forest for LASSO-NPDR-URFP features
##################
## make a probability forest for case wise importance analysis
# using LASSO-NPDR-URFP features and make a probability forest
selTrain.dat$biotic<-as.factor(selTrain.dat$biotic)
rfNpdrProb.fit <- ranger(biotic ~ ., selTrain.dat, keep.inbag = TRUE,
                          num.trees=bestNpdr_maxTree, # match full variable num trees
                          mtry=bestNpdr_mtry, 
                          importance="permutation", splitrule = bestNpdr_splitRule,
                          min.node.size=bestNpdr_minNodeSize,
                          class.weights = as.numeric(c(1/table(selTrain.dat$biotic))),
                          scale.permutation.importance = T, probability = T,
                          local.importance = T, num.threads=4)
rfNpdrProb_case<-rfNpdrProb.fit$variable.importance.local
bioAbioRF_prob<-rfNpdrProb.fit$predictions

prob.test = predict(rfNpdrProb.fit, data = selTest.dat)$predictions
head(prob.test)
prob.df<-rbind.data.frame(bioAbioRF_prob,prob.test)
head(prob.df)
#     abiotic    biotic
# 1 0.2332616 0.7667384
# 2 0.3501293 0.6498707
# 3 0.2576820 0.7423180
# 4 0.4259084 0.5740916
# 5 0.4287439 0.5712561
# 6 0.3047135 0.6952865

# get misclassified samples from previous analysis
train.miss<-which(rfNpdrFinal.fit$predictions != selTrain.dat$biotic)
test.miss<-which(predNpdrFinal.test$predictions != selTest.dat$biotic)

# read in data with analysis numbers
trainAnalysis.dat<-read.csv("./data/tsms_0.3p_bio_train.csv")
testAnalysis.dat<-read.csv("./data/tsms_0.3p_bio_test.csv")

an1miss<-trainAnalysis.dat$Analysis[train.miss]
an2miss<-testAnalysis.dat$Analysis[test.miss]
an.miss<-c(an1miss,an2miss)
# an.miss<-c(5757,5770,5784,5878,5880,5885,5886,5892,5894,5897,5911,5917,5953,
#            5960, 5933)

# make predicted and actual class vectors
predAll.vec<-c(rfNpdrFinal.fit$predictions,predNpdrFinal.test$predictions)
bioAct.vec<-c(selTrain.dat$biotic,selTest.dat$biotic)
# make dataset vectors (to tell whether analysis was in training or testing data)
trainTest.vec<-rep("train",140)
trainTest.vec<-c(trainTest.vec,rep("test",34))
# replace categorical integers with labels (from factors)
pred.vec<-rep("abiotic",length(predAll.vec))
pred.vec[which(predAll.vec==2)]<-"biotic"
bio.vec<-rep("abiotic",length(bioAct.vec))
bio.vec[which(bioAct.vec==2)]<-"biotic"
# make analysis vector
an.vec<-c(trainAnalysis.dat$Analysis,testAnalysis.dat$Analysis)
pred.df<-cbind.data.frame(bio.vec,pred.vec,an.vec,trainTest.vec)
colnames(pred.df)<-c("actual","pred","Analysis","trainTest")
head(pred.df)
#    actual   pred Analysis trainTest
# 1 biotic biotic     2960     train
# 2 biotic biotic     2961     train
# 3 biotic biotic     2962     train
# 4 biotic biotic     2963     train
# 5 biotic biotic     2965     train
# 6 biotic biotic     2966     train

# add analysis vec to prob df
prob.df$Analysis<-an.vec
head(prob.df)
#     abiotic    biotic Analysis
# 1 0.2351666 0.7648334     2960
# 2 0.3436013 0.6563987     2961
# 3 0.2650792 0.7349208     2962
# 4 0.4197209 0.5802791     2963
# 5 0.4271395 0.5728605     2965
# 6 0.3088334 0.6911666     2966

# get misclassifications
missClass.df<-pred.df[which(pred.df$actual != pred.df$pred),]
missClass.df
#      actual    pred Analysis trainTest
# 70   biotic abiotic     5757     train
# 77  abiotic  biotic     5770     train
# 83   biotic abiotic     5784     train
# 84   biotic abiotic     5878     train
# 86   biotic abiotic     5880     train
# 90  abiotic  biotic     5885     train
# 91  abiotic  biotic     5886     train
# 93  abiotic  biotic     5892     train
# 95  abiotic  biotic     5894     train
# 98  abiotic  biotic     5897     train
# 106 abiotic  biotic     5911     train
# 111 abiotic  biotic     5917     train
# 134  biotic abiotic     5953     train
# 140  biotic abiotic     5960     train
# 168 abiotic  biotic     5933      test

missprob<-prob.df[which(pred.df$actual != pred.df$pred),1:2]
# add to missclass df
missClassProb.df<-cbind.data.frame(missClass.df,missprob)
missClassProb.df
# ** = prob doesn't match classification: because these are on the edge
# (probabilities are close) and two forests were constructed (one for class, one for probabilities), 
# so we expect some small differences like this for samples that are difficult to predict
# Fortran RF output calculates probabilities and predicted classes for the same forest
# so these discrepancies don't occur.
# We would have to find a way to access the number of votes for the classes for each sample from the model
# to replicate the original implementation better.
# Using Fortran for the analysis is more streamlined but less accessible overall to modern programmers
# We accept the discrepancies and understand why they occur.
# Future work may involve working with the Fortran code through customized R wrapper functions to
# make the analysis more accessible
#      actual    pred Analysis trainTest   abiotic    biotic
# 70   biotic abiotic     5757     train 0.6883255 0.3116745
# 77  abiotic  biotic     5770     train 0.5415164 0.4584836**
# 83   biotic abiotic     5784     train 0.6420207 0.3579793
# 84   biotic abiotic     5878     train 0.8390971 0.1609029
# 86   biotic abiotic     5880     train 0.7245267 0.2754733
# 90  abiotic  biotic     5885     train 0.2412484 0.7587516
# 91  abiotic  biotic     5886     train 0.5792096 0.4207904**
# 93  abiotic  biotic     5892     train 0.2755204 0.7244796
# 95  abiotic  biotic     5894     train 0.5674980 0.4325020**
# 98  abiotic  biotic     5897     train 0.2182992 0.7817008
# 106 abiotic  biotic     5911     train 0.5652089 0.4347911**
# 111 abiotic  biotic     5917     train 0.5175190 0.4824810**
# 134  biotic abiotic     5953     train 0.6485748 0.3514252
# 140  biotic abiotic     5960     train 0.6271311 0.3728689
# 168 abiotic  biotic     5933      test 0.4024459 0.5975541


### insepct two correct classifications (biotic/abiotic) and two incorrect 
# try Analyses 2962 (correct biotic), 5880 (incorrect biotic), 
# 5918 (correct abiotic), 5933 (incorrect abiotic)
inspAn<-c(2962,5880,5918,5933) # edit for runs with other misclassifications

# make one df for probabilities and predicted classes
classProb.df<-cbind.data.frame(pred.df,prob.df[,1:2])
head(classProb.df)
#   actual   pred Analysis trainTest   abiotic    biotic
# 1 biotic biotic     2960     train 0.2351666 0.7648334
# 2 biotic biotic     2961     train 0.3436013 0.6563987
# 3 biotic biotic     2962     train 0.2650792 0.7349208
# 4 biotic biotic     2963     train 0.4197209 0.5802791
# 5 biotic biotic     2965     train 0.4271395 0.5728605
# 6 biotic biotic     2966     train 0.3088334 0.6911666

# look at Analyses to inspect
classProb.df[which(classProb.df$Analysis %in% inspAn),]
# * = missclassified
#      actual    pred Analysis trainTest   abiotic    biotic
# 3    biotic  biotic     2962     train 0.2650792 0.7349208
# 86   biotic abiotic     5880     train 0.7245267 0.2754733*
# 112 abiotic abiotic     5918     train 0.6800508 0.3199492
# 168 abiotic  biotic     5933      test 0.4024459 0.5975541*

## save classProb.df to file
write.table(classProb.df,"./samplewise_output/bioAbio_npdrURFP_RF_classProb.csv",quote=F,row.names=F,sep=",")
#classProb.df<-read.csv("./samplewise_output/bioAbio_npdrURFP_RF_classProb.csv")



##################
# Part VIII: case-wise variable importance for all samples (training + testing)
####################
## now get case-wise variable importance: need to use all data as training data
allTrain.dat<-rbind.data.frame(selTrain.dat,selTest.dat)

# make classification forest and generate casewise importance data for training and 
# testing together
# expect same tuning parameters to provide good fit
# mtry  splitrule min.node.size
# 4    2 extratrees             8
# used 6000 for maxtrees
rfNpdrCase.fit <- ranger(biotic ~ ., allTrain.dat, keep.inbag = TRUE,
                         num.trees=bestNpdr_maxTree, # match full variable num trees
                         mtry=bestNpdr_mtry, 
                         importance="permutation", splitrule = bestNpdr_splitRule,
                         min.node.size=bestNpdr_minNodeSize,
                         class.weights = as.numeric(c(1/table(allTrain.dat$biotic))),
                         scale.permutation.importance = T, #probability = T,
                         local.importance = T, num.threads=4)
rfNpdrCase<-rfNpdrCase.fit$variable.importance.local*100 #make magnitudes easier to read
rfNpdrCase<-as.data.frame(rfNpdrCase)
head(rfNpdrCase)
#diff2_acf1 time_kl_shift fluctanal_prop_r1 walker_propcross avg_rR45CO244CO2
# 1   4.000000     1.8000000          4.000000        1.1500000        6.0333333
# 2   6.250000     1.0666667          5.833333       -2.4333333        4.8000000
# 3   5.550000     2.1333333          4.383333        2.3833333        8.2666667
# 4   1.033333     0.3833333          2.933333        1.3000000        0.6166667
# 5   3.883333     1.1166667          3.666667        0.3166667        1.3666667
# 6   4.283333     2.2666667          4.033333        0.9333333        3.6666667
# avg_R45CO244CO2 avg_rd45CO244CO2 avg_d45CO244CO2 sd_d18O13C
# 1       0.9833333        1.3666667       1.5000000  0.8666667
# 2       2.2666667        1.6500000       1.3333333  1.1500000
# 3       0.7833333        1.1000000       0.8333333  1.1166667
# 4       1.0000000        0.3000000       0.6333333  3.0666667
# 5       0.1500000        0.1333333       0.3500000 -0.4666667
# 6       1.2666667        1.8666667       1.4833333 -0.3000000

# put in order of LASSO-NPDR-URFP ranking
colorder<-as.character(npdr_scores$features)
colorder
# [1] "avg_R45CO244CO2"   "avg_rR45CO244CO2"  "sd_d18O13C"       
# [4] "diff2_acf1"        "walker_propcross"  "fluctanal_prop_r1"
# [7] "avg_rd45CO244CO2"  "time_kl_shift"     "avg_d45CO244CO2"  
# use reverse order bc will flip plot 
newCase.imp<-cbind.data.frame(rfNpdrCase$avg_d45CO244CO2,
                              rfNpdrCase$time_kl_shift,
                              rfNpdrCase$avg_rd45CO244CO2,
                              rfNpdrCase$fluctanal_prop_r1,
                              rfNpdrCase$walker_propcross,
                              rfNpdrCase$diff2_acf1,
                              rfNpdrCase$sd_d18O13C,
                              rfNpdrCase$avg_rR45CO244CO2,
                              rfNpdrCase$avg_R45CO244CO2)
colnames(newCase.imp)<-rev(colorder)
case.imp<-newCase.imp

# add predicted and actual class + Analysis data and probabilities
# redo colnames for graphing
case.imp$biotic<-classProb.df$actual
case.imp$pred<-classProb.df$pred
case.imp$Analysis<-classProb.df$Analysis
case.imp$abio_prob<-classProb.df$abiotic
case.imp$bio_prob<-classProb.df$biotic

# get data for analyses of interest
plotCaseImp<-case.imp[which(case.imp$Analysis %in% inspAn),]
plotCaseImp
#     avg_d45CO244CO2 time_kl_shift avg_rd45CO244CO2 fluctanal_prop_r1 walker_propcross diff2_acf1 sd_d18O13C
# 3         0.9833333    2.11666667       1.15000000          4.250000         2.033333   5.116667  0.8666667
# 86        0.1000000   -0.08333333       0.16666667          2.750000        -0.750000   1.050000  1.8833333
# 112       0.1333333    0.18333333      -0.15000000         -2.550000         0.750000   4.516667 -6.6000000
# 168       0.2666667   -0.46666667       0.01666667         -4.333333        -1.066667  -1.533333 -0.1000000
#     avg_rR45CO244CO2 avg_R45CO244CO2  biotic    pred Analysis abio_prob  bio_prob
# 3           9.166667       0.7500000  biotic  biotic     2962 0.2650792 0.7349208
# 86         -3.083333       0.2000000  biotic abiotic     5880 0.7245267 0.2754733
# 112         2.016667       0.0000000 abiotic abiotic     5918 0.6800508 0.3199492
# 168        -1.966667       0.2333333 abiotic  biotic     5933 0.4024459 0.5975541


## save to file
write.table(plotCaseImp,"./samplewise_output/caseWiseImp_inspectAnalyses.csv",row.names=F,quote=F,sep=",")


## created melted df for plotting for each row in plotCaseImp
melted1 <- melt(plotCaseImp[1,],
               id=c("biotic","pred","Analysis","bio_prob","abio_prob")) 
head(melted1)
#   biotic   pred Analysis  bio_prob abio_prob          variable     value
# 1 biotic biotic     2962 0.7349208 0.2650792   avg_d45CO244CO2 0.9833333
# 2 biotic biotic     2962 0.7349208 0.2650792     time_kl_shift 2.1166667
# 3 biotic biotic     2962 0.7349208 0.2650792  avg_rd45CO244CO2 1.1500000
# 4 biotic biotic     2962 0.7349208 0.2650792 fluctanal_prop_r1 4.2500000
# 5 biotic biotic     2962 0.7349208 0.2650792  walker_propcross 2.0333333
# 6 biotic biotic     2962 0.7349208 0.2650792        diff2_acf1 5.1166667

melted2 <- melt(plotCaseImp[2,],
                id=c("biotic","pred","Analysis","bio_prob","abio_prob"))
melted3 <- melt(plotCaseImp[3,],
                id=c("biotic","pred","Analysis","bio_prob","abio_prob"))
melted4 <- melt(plotCaseImp[4,],
                id=c("biotic","pred","Analysis","bio_prob","abio_prob"))

# get sign of permutation importance for plot and add to melted dfs
sign1.vec<-rep(0,dim(melted1)[1])
sign1.vec[which(melted1$value<=0)]<-"increase_accuracy"
sign1.vec[which(melted1$value>0)]<-"decrease_accuracy"
melted1$variable_permutation<-sign1.vec
colnames(melted1)[which(colnames(melted1)=="value")]<-"local_importance"

sign2.vec<-rep(0,dim(melted2)[1])
sign2.vec[which(melted2$value<=0)]<-"increase_accuracy"
sign2.vec[which(melted2$value>0)]<-"decrease_accuracy"
melted2$variable_permutation<-sign2.vec
colnames(melted2)[which(colnames(melted2)=="value")]<-"local_importance"

sign3.vec<-rep(0,dim(melted3)[1])
sign3.vec[which(melted3$value<=0)]<-"increase_accuracy"
sign3.vec[which(melted3$value>0)]<-"decrease_accuracy"
melted3$variable_permutation<-sign3.vec
colnames(melted3)[which(colnames(melted3)=="value")]<-"local_importance"

sign4.vec<-rep(0,dim(melted4)[1])
sign4.vec[which(melted4$value<=0)]<-"increase_accuracy"
sign4.vec[which(melted4$value>0)]<-"decrease_accuracy"
melted4$variable_permutation<-sign4.vec
colnames(melted4)[which(colnames(melted4)=="value")]<-"local_importance"

## now make plots
## plot
case.p1<-ggplot(melted1, 
              aes(x=variable,y=local_importance,fill=variable_permutation)) + 
  geom_bar(position="stack",stat="identity") +
  coord_flip() + 
  scale_fill_manual(values=c("lightblue","salmon")) +
  geom_hline(yintercept=0)  + theme_bw() +
  labs(title="Case-wise variable importance for LASSO-NPDR-URFP selected features",
       subtitle=paste("Analysis: ",melted1$Analysis[1],
                      "\n","Class: ","\n", "\t"," actual = ",melted1$biotic[1],
                      "\t \t predicted = ",melted1$pred[1],
                      "\nProbability (percent votes): ", "\n","\t abiotic = ",
                      round(melted1$abio_prob[1],4),
                      "\t \t biotic = ", round(melted1$bio_prob[1],4),
                      sep=""),
       caption=paste("local importance for variable m = \n(percent correct votes OOB) - (percent correct votes permuted m)"))
case.p1 


case.p2<-ggplot(melted2, 
                aes(x=variable,y=local_importance,fill=variable_permutation)) + 
  geom_bar(position="stack",stat="identity") +
  coord_flip() + 
  scale_fill_manual(values=c("lightblue","salmon")) +
  geom_hline(yintercept=0)  + theme_bw() +
  labs(title="Case-wise variable importance for LASSO-NPDR-URFP selected features",
       subtitle=paste("Analysis: ",melted2$Analysis[1],
                      "\n","Class: ","\n", "\t"," actual = ",melted2$biotic[1],
                      "\t \t predicted = ",melted2$pred[1],
                      "\nProbability (percent votes): ", "\n","\t abiotic = ",
                      round(melted2$abio_prob[1],4),
                      "\t \t biotic = ", round(melted2$bio_prob[1],4),
                      sep=""),
       caption=paste("local importance for variable m = \n(percent correct votes OOB) - (percent correct votes permuted m)"))
case.p2 

case.p3<-ggplot(melted3, 
                aes(x=variable,y=local_importance,fill=variable_permutation)) + 
  geom_bar(position="stack",stat="identity") +
  coord_flip() + 
  scale_fill_manual(values=c("lightblue","salmon")) +
  geom_hline(yintercept=0)  + theme_bw() +
  labs(title="Case-wise variable importance for LASSO-NPDR-URFP selected features",
       subtitle=paste("Analysis: ",melted3$Analysis[1],
                      "\n","Class: ","\n", "\t"," actual = ",melted3$biotic[1],
                      "\t \t predicted = ",melted3$pred[1],
                      "\nProbability (percent votes): ", "\n","\t abiotic = ",
                      round(melted3$abio_prob[1],4),
                      "\t \t biotic = ", round(melted3$bio_prob[1],4),
                      sep=""),
       caption=paste("local importance for variable m = \n(percent correct votes OOB) - (percent correct votes permuted m)"))
case.p3

case.p4<-ggplot(melted4, 
                aes(x=variable,y=local_importance,fill=variable_permutation)) + 
  geom_bar(position="stack",stat="identity") +
  coord_flip() + 
  scale_fill_manual(values=c("lightblue","salmon")) +
  geom_hline(yintercept=0)  + theme_bw() +
  labs(title="Case-wise variable importance for LASSO-NPDR-URFP selected features",
       subtitle=paste("Analysis: ",melted4$Analysis[1],
                      "\n","Class: ","\n", "\t"," actual = ",melted4$biotic[1],
                      "\t \t predicted = ",melted4$pred[1],
                      "\nProbability (percent votes): ", "\n","\t abiotic = ",
                      round(melted4$abio_prob[1],4),
                      "\t \t biotic = ", round(melted4$bio_prob[1],4),
                      sep=""),
       caption=paste("local importance for variable m = \n(percent correct votes OOB) - (percent correct votes permuted m)"))
case.p4


##################
# Part IX: create biotic and abiotic prototype plots
##############
### prototypes for biotic and abiotic classes based on NPDR and RF features

# make manual biotic and abiotic prototypes using the median and quartiles for the data
# scale data first
scale.dat<-scale(allTrain.dat%>%select(!biotic),scale=T,center=T)
# split data into biotic and abiotic
bio.dat<-scale.dat[which(allTrain.dat$biotic=="biotic"),]
abio.dat<-scale.dat[which(allTrain.dat$biotic=="abiotic"),]

# calculate quantiles
quant.dat<-data.frame(matrix(rep(NA,(ncol(allTrain.dat)-1)*5),ncol=5))
colnames(quant.dat)<-c("0%","25%","50%","75%","100%")
for(i in seq(1,ncol(allTrain.dat)-1)){
  curr.quant<-quantile(allTrain.dat[,i])
  quant.dat[i,]<-curr.quant
}
quant.dat<-cbind.data.frame(colnames(allTrain.dat)[1:ncol(allTrain.dat)-1],quant.dat)
colnames(quant.dat)<-c("variable",colnames(quant.dat)[2:ncol(quant.dat)])
(quant.dat)
#           variable            0%           25%           50%           75%          100%
# 1        diff2_acf1  6.878918e-01   0.727839030   0.741252181   0.786732326    0.81758775
# 2     time_kl_shift  1.330000e+02 133.000000000 133.000000000 323.000000000 4038.00000000
# 3 fluctanal_prop_r1  3.469388e-01   0.408163265   0.408163265   0.428571429    0.48979592
# 4  walker_propcross  1.396323e-02   0.018152199   0.020013963   0.021875727    0.02932961
# 5  avg_rR45CO244CO2  1.176681e+00   1.178881882   1.179501131   1.180248635    1.18226546
# 6   avg_R45CO244CO2  1.182064e-02   0.011830110   0.011837673   0.011849046    0.01186736
# 7  avg_rd45CO244CO2  2.294231e+01  23.761759990  24.416167054  25.400419785   26.98510885
# 8   avg_d45CO244CO2 -1.370154e+01 -12.911438286 -12.280473396 -11.331478571   -9.80355625
# 9        sd_d18O13C  9.174806e-04   0.002507211   0.003168706   0.004059154    0.01135171

# biotic data
bio_quant.dat<-data.frame(matrix(rep(NA,(ncol(bio.dat)*5)),ncol=5))
dim(bio_quant.dat)
# [1] 9 5 # one value for each feature
# abiotic data
abio_quant.dat<-data.frame(matrix(rep(NA,(ncol(abio.dat))*5),ncol=5))
colnames(bio_quant.dat)<-c("0%","25%","50%","75%","100%")
colnames(abio_quant.dat)<-c("0%","25%","50%","75%","100%")

# add data 
# bio
for(i in seq(1,ncol(bio.dat))){
  curr.quant<-quantile(bio.dat[,i])
  bio_quant.dat[i,]<-curr.quant
}
bio_quant.dat<-cbind.data.frame(colnames(bio.dat)[1:ncol(bio.dat)],bio_quant.dat)
colnames(bio_quant.dat)<-c("variable",colnames(bio_quant.dat)[2:ncol(bio_quant.dat)])
(bio_quant.dat)
#            variable         0%         25%         50%         75%     100%
# 1        diff2_acf1 -1.7081738 -0.62629091 -0.49769436 -0.01195112 1.866081
# 2     time_kl_shift -0.3683294 -0.36832938 -0.36832938 -0.36832938 3.333871
# 3 fluctanal_prop_r1 -1.8707197  0.05534674  0.69736887  0.69736887 2.623435
# 4  walker_propcross -0.8971350 -0.47260870 -0.04808235  0.51795277 2.782093
# 5  avg_rR45CO244CO2 -2.4504311 -0.63531333 -0.19186996  0.39275497 1.232039
# 6   avg_R45CO244CO2 -1.6258500 -0.93576807 -0.29794963  0.85139100 2.365826
# 7  avg_rd45CO244CO2 -1.6258550 -0.93577032 -0.29796942  0.85139665 2.365823
# 8   avg_d45CO244CO2 -1.6258550 -0.93577032 -0.29796942  0.85139665 2.365823
# 9        sd_d18O13C -1.3336417 -0.52090445 -0.21426710  0.31210232 4.559171

# abio
for(i in seq(1,ncol(abio.dat))){
  curr.quant<-quantile(abio.dat[,i])
  abio_quant.dat[i,]<-curr.quant
}
abio_quant.dat<-cbind.data.frame(colnames(abio.dat)[1:ncol(abio.dat)],abio_quant.dat)
colnames(abio_quant.dat)<-c("variable",colnames(abio_quant.dat)[2:ncol(abio_quant.dat)])
(abio_quant.dat)
#            variable         0%        25%         50%        75%     100%
# 1        diff2_acf1 -1.9554008 -0.7506146 -0.06253756  1.3445948 1.988056
# 2     time_kl_shift -0.3683294 -0.3683294 -0.36832938 -0.1881967 3.333871
# 3 fluctanal_prop_r1 -1.8707197 -1.8707197  0.05534674  0.6973689 2.623435
# 4  walker_propcross -1.8876965 -0.8971350 -0.04808235  0.4479560 2.784168
# 5  avg_rR45CO244CO2 -2.4235792 -0.4222365  0.15077504  0.7943089 2.460109
# 6   avg_R45CO244CO2 -1.5293833 -0.7820109 -0.09087046  0.7896005 1.921259
# 7  avg_rd45CO244CO2 -1.5293911 -0.7820022 -0.09085782  0.7896034 1.921259
# 8   avg_d45CO244CO2 -1.5293911 -0.7820022 -0.09085782  0.7896034 1.921259
# 9        sd_d18O13C -1.5565288 -0.7137964 -0.26212595  0.2529197 1.948842

# combine - now have diff cols for abio and bio
both.quant<-cbind.data.frame(abio_quant.dat,bio_quant.dat[,2:ncol(bio_quant.dat)])
colnames(both.quant)<-c("variable","abio_0", "abio_25", "abio_50","abio_75","abio_100",
                        "bio_0","bio_25","bio_50","bio_75","bio_100")
proto.proc<-both.quant # reassign for prototype plots

### reorder by variable importance 
npdrColOrder<-npdr.feat
npdrColOrder
# [1] "avg_R45CO244CO2"   "avg_rR45CO244CO2"  "sd_d18O13C"       
# [4] "diff2_acf1"        "walker_propcross"  "fluctanal_prop_r1"
# [7] "avg_rd45CO244CO2"  "time_kl_shift"     "avg_d45CO244CO2"    
protoNpdr.proc<-rbind.data.frame(
  proto.proc[which(proto.proc$variable==npdrColOrder[1]),],
  proto.proc[which(proto.proc$variable==npdrColOrder[2]),],
  proto.proc[which(proto.proc$variable==npdrColOrder[3]),],
  proto.proc[which(proto.proc$variable==npdrColOrder[4]),], 
  proto.proc[which(proto.proc$variable==npdrColOrder[5]),], 
  proto.proc[which(proto.proc$variable==npdrColOrder[6]),],  
  proto.proc[which(proto.proc$variable==npdrColOrder[7]),], 
  proto.proc[which(proto.proc$variable==npdrColOrder[8]),], 
  proto.proc[which(proto.proc$variable==npdrColOrder[9]),])
head(protoNpdr.proc)

# write NPDR prototype data to file
write.table(protoNpdr.proc,"./samplewise_output/bioAbio_NpdrURFPVars_prototypes.csv",row.names=F,quote=F,sep=",")


## plot prototype data
melt_proto<-melt(protoNpdr.proc %>% select(!c("abio_0", "abio_25","abio_75","abio_100",
                                          "bio_0","bio_25","bio_75","bio_100")),
                 id=c("variable"))
colnames(melt_proto)<-c("variable","group","value")
head(melt_proto)
#            variable   group       value
# 1   avg_R45CO244CO2 abio_50 -0.09087046
# 2  avg_rR45CO244CO2 abio_50  0.15077504
# 3        sd_d18O13C abio_50 -0.26212595
# 4        diff2_acf1 abio_50 -0.06253756
# 5  walker_propcross abio_50 -0.04808235
# 6 fluctanal_prop_r1 abio_50  0.05534674

# tell ggplot to order the x axis by importance
protoNpdr.plot <- ggplot(data=melt_proto, 
  mapping=aes(x=fct_inorder(variable),
  y=value,group=group)) +
  geom_line(aes(color=group),linewidth=0.5) + 
  geom_point(aes(color=group),size=1) +
  theme_bw() + theme(#panel.border = element_blank(), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=10)) +
  scale_color_manual(values=c("black", "limegreen")) #+
#scale_x_discrete(breaks=melt_proto$variable,labels=as.character(proto.proc$variable))
protoNpdr.plot+ggtitle("Random Forest prototypes: biotic and abiotic classes in 0.3% CO2 IRMS data",
                   subtitle = "LASSO-NPDR-URFP variables")+labs(x="variable",y="prototype")


### compare with RF top 9 variables from permutation importance
rf_feat<-names(rf.feat)
rf_feat
# [1] "max_kl_shift"       "fluctanal_prop_r1"  "localsimple_taures"
# [4] "avg_rR45CO244CO2"   "time_level_shift"   "diff2x_pacf5"      
# [7] "diff1x_pacf5"       "time_var_shift"     "diff2_acf10"
# subset data based on these features
rfFullTrain.dat<-rbind.data.frame(train.dat,test.dat)
# combine testing and training data
rfAll.dat<-rfFullTrain.dat[,which(colnames(rfFullTrain.dat) %in% c(rf_feat,"biotic"))]

# scale data
scalerf.dat<-scale(rfAll.dat%>%select(!biotic),scale=T,center=T)

# split data into biotic and abiotic
bioRF.dat<-scalerf.dat[which(rfAll.dat$biotic=="biotic"),]
abioRF.dat<-scalerf.dat[which(rfAll.dat$biotic=="abiotic"),]

# initialize df for RF feature quantiles
quantRF.dat<-data.frame(matrix(rep(NA,(ncol(rfAll.dat)-1)*5),ncol=5))
colnames(quantRF.dat)<-c("0%","25%","50%","75%","100%")
for(i in seq(1,ncol(rfAll.dat)-1)){
  curr.quant<-quantile(rfAll.dat[,i])
  quantRF.dat[i,]<-curr.quant
}
quantRF.dat<-cbind.data.frame(colnames(rfAll.dat)[1:ncol(rfAll.dat)-1],quantRF.dat)
colnames(quantRF.dat)<-c("variable",colnames(quantRF.dat)[2:ncol(quantRF.dat)])
head(quantRF.dat)
#           variable           0%          25%          50%          75%         100%
# 1      diff2_acf10    0.6990858    0.7494848    0.7776238    0.9263998     1.277038
# 2     max_kl_shift 3463.1495820 4092.0606082 7062.5551553 8100.5721290 22400.141671
# 3 time_level_shift  625.0000000 1297.0000000 1300.0000000 1308.0000000  1309.000000
# 4   time_var_shift  622.0000000 1291.2500000 1295.0000000 1303.0000000  1304.000000
# 5     diff1x_pacf5    2.3405670    2.4248401    2.4553789    2.5608495     2.757065
# 6     diff2x_pacf5    1.6068517    1.6787749    1.7020914    1.8031440     2.056630
# split into biotic and abiotic
bio_quantRF.dat<-data.frame(matrix(rep(NA,(ncol(bioRF.dat))*5),ncol=5))
colnames(bio_quantRF.dat)<-c("0%","25%","50%","75%","100%")

abio_quantRF.dat<-data.frame(matrix(rep(NA,(ncol(abioRF.dat))*5),ncol=5))
colnames(abio_quantRF.dat)<-c("0%","25%","50%","75%","100%")

# add data
# bio
for(i in seq(1,ncol(bioRF.dat))){
  curr.quant<-quantile(bioRF.dat[,i])
  bio_quantRF.dat[i,]<-curr.quant
}
bio_quantRF.dat<-cbind.data.frame(colnames(bioRF.dat)[1:ncol(bioRF.dat)],bio_quantRF.dat)
colnames(bio_quantRF.dat)<-c("variable",colnames(bio_quantRF.dat)[2:ncol(bio_quantRF.dat)])
(bio_quantRF.dat)
#          variable         0%        25%        50%        75%       100%
# 1        diff2_acf10 -0.9478020 -0.6221475 -0.5546822 -0.3256271 1.30450917
# 2       max_kl_shift -0.9901058 -0.8565191 -0.3483122 -0.1562708 0.09527431
# 3   time_level_shift  0.2327237  0.2327237  0.3024706  0.3082829 0.30828289
# 4     time_var_shift  0.2330130  0.2330130  0.3028868  0.3028868 0.30870962
# 5       diff1x_pacf5 -1.3490934 -0.6173012 -0.5033816 -0.1126484 1.51185033
# 6       diff2x_pacf5 -1.1432862 -0.5911004 -0.4959549 -0.1902715 1.37199693
# 7  fluctanal_prop_r1 -1.8707197  0.05534674  0.6973689  0.6973689 2.62343526
# 8 localsimple_taures -0.2647357 -0.26473568  0.6211106  0.6211106 0.62111063
# 9   avg_rR45CO244CO2 -2.4504311 -0.63531333 -0.1918700  0.3927550 1.23203950

# abio rf quantile data
for(i in seq(1,ncol(abioRF.dat))){
  curr.quant<-quantile(abioRF.dat[,i])
  abio_quantRF.dat[i,]<-curr.quant
}
abio_quantRF.dat<-cbind.data.frame(colnames(abioRF.dat)[1:ncol(abioRF.dat)],abio_quantRF.dat)
colnames(abio_quantRF.dat)<-c("variable",colnames(abio_quantRF.dat)[2:ncol(abio_quantRF.dat)])
(abio_quantRF.dat)
#             variable         0%        25%          50%       75%      100%
# 1        diff2_acf10 -1.0035527 -0.6955787 -0.348187109 1.1433051 2.7420784
# 2       max_kl_shift -0.9621528 -0.2415885 -0.001399098 0.2267103 3.5687597
# 3   time_level_shift -3.6672937  0.2385359  0.255972670 0.3024706 0.3082829
# 4     time_var_shift -3.6624506  0.2388358  0.256304280 0.3028868 0.3087096
# 5       diff1x_pacf5 -1.5696369 -0.8499462 -0.205208962 1.3186390 2.6631653
# 6       diff2x_pacf5 -1.3273397 -0.8000470 -0.267115770 1.2428663 2.8948703
# 7  fluctanal_prop_r1 -1.8707197 -1.8707197  0.055346735 0.6973689 2.6234353
# 8 localsimple_taures -3.8081209 -0.2647357  0.621110625 0.6211106 0.6211106
# 9   avg_rR45CO244CO2 -2.4235792 -0.4222365  0.150775041 0.7943089 2.4601093

bothRF.quant<-cbind.data.frame(abio_quantRF.dat,bio_quantRF.dat[,2:ncol(bio_quantRF.dat)])
colnames(bothRF.quant)<-c("variable","abio_0", "abio_25", "abio_50","abio_75","abio_100",
                        "bio_0","bio_25","bio_50","bio_75","bio_100")
protoRF.proc<-bothRF.quant
head(protoRF.proc)
#           variable     abio_0    abio_25      abio_50   abio_75  abio_100      bio_0     bio_25
# 1      diff2_acf10 -1.0035527 -0.6955787 -0.348187109 1.1433051 2.7420784 -0.9478020 -0.6221475
# 2     max_kl_shift -0.9621528 -0.2415885 -0.001399098 0.2267103 3.5687597 -0.9901058 -0.8565191
# 3 time_level_shift -3.6672937  0.2385359  0.255972670 0.3024706 0.3082829  0.2327237  0.2327237
# 4   time_var_shift -3.6624506  0.2388358  0.256304280 0.3028868 0.3087096  0.2330130  0.2330130
# 5     diff1x_pacf5 -1.5696369 -0.8499462 -0.205208962 1.3186390 2.6631653 -1.3490934 -0.6173012
# 6     diff2x_pacf5 -1.3273397 -0.8000470 -0.267115770 1.2428663 2.8948703 -1.1432862 -0.5911004
#       bio_50     bio_75    bio_100
# 1 -0.5546822 -0.3256271 1.30450917
# 2 -0.3483122 -0.1562708 0.09527431
# 3  0.3024706  0.3082829 0.30828289
# 4  0.3028868  0.3028868 0.30870962
# 5 -0.5033816 -0.1126484 1.51185033
# 6 -0.4959549 -0.1902715 1.37199693

### reorder by variable importance 
rfColOrder<-rf_feat
rfColOrder
# [1] "max_kl_shift"       "fluctanal_prop_r1"  "localsimple_taures" "avg_rR45CO244CO2"  
# [5] "time_level_shift"   "diff2x_pacf5"       "diff1x_pacf5"       "time_var_shift"    
# [9] "diff2_acf10"       
protoRF.proc2<-rbind.data.frame(
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[1]),],
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[2]),],
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[3]),],
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[4]),], 
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[5]),], 
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[6]),],  
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[7]),], 
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[8]),], 
  protoRF.proc[which(protoRF.proc$variable==rfColOrder[9]),])
head(protoRF.proc2)


# write RF prototype data to file
write.table(protoRF.proc2,"./samplewise_output/bioAbio_rfTopVars_prototypes.csv",row.names=F,quote=F,sep=",")

# melt prototype data
melt_proto2<-melt(protoRF.proc2 %>% select(!c("abio_0", "abio_25","abio_75","abio_100",
                                            "bio_0","bio_25","bio_75","bio_100")),
                  id=c("variable"))
colnames(melt_proto2)<-c("variable","group","value")

# tell ggplot to order the x axis by importance
protoRF.plot <- ggplot(data=melt_proto2, 
  mapping=aes(x=fct_inorder(variable),y=value,group=group)) +
  geom_line(aes(color=group),linewidth=0.5) + 
  geom_point(aes(color=group),size=1) +
  theme_bw() + theme(#panel.border = element_blank(), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=10)) +
  scale_color_manual(values=c("black", "limegreen")) #+
#scale_x_discrete(breaks=melt_proto$variable,labels=as.character(proto.proc$variable))
protoRF.plot+ggtitle("Random Forest prototypes: biotic and abiotic classes in 0.3% CO2 IRMS data",
                    subtitle = "Random Forest permutation importance top variables")+labs(x="variable",y="prototype")




##################
# Part X: supervised RF for top nine RF importance variables
##############
## RF model in RF top variable feature space 
# use RF top vars in a new RF model and assess performance 

# create training and testing data: same samples, different features
rfTrain.dat<-train.dat[,which(colnames(train.dat) %in% c(rf_feat,"biotic"))]
rfTest.dat<-test.dat[,which(colnames(test.dat) %in% c(rf_feat,"biotic"))]
head(rfTrain.dat)
#   diff2_acf10 max_kl_shift time_level_shift time_var_shift diff1x_pacf5 diff2x_pacf5
# 1   0.7445355     4070.361             1296           1291     2.421829     1.675950
# 2   0.7696068     4226.687             1296           1291     2.445484     1.697435
# 3   0.7465851     4058.530             1296           1291     2.423962     1.678719
# 4   0.7165986     3852.688             1296           1291     2.380570     1.643994
# 5   0.7773469     4153.396             1296           1291     2.454390     1.706982
# 6   0.7752833     4088.991             1296           1291     2.455226     1.708571
#   fluctanal_prop_r1 localsimple_taures avg_rR45CO244CO2 biotic
# 1         0.4081633                 10         1.176956 biotic
# 2         0.4081633                 10         1.178937 biotic
# 3         0.4081633                 10         1.176910 biotic
# 4         0.4897959                 10         1.179907 biotic
# 5         0.4081633                 10         1.179628 biotic
# 6         0.4081633                 10         1.178887 biotic

rfSelFinal.fit <- ranger(biotic ~ ., rfTrain.dat, keep.inbag = TRUE,
                          num.trees=6000, mtry=2, 
                          importance="permutation", splitrule = "extratrees",
                          min.node.size=8,
                          class.weights = as.numeric(c(1/table(rfTrain.dat$biotic))),
                          scale.permutation.importance = T,
                          local.importance = T, num.threads=4)
sorted.imp<-sort(rfSelFinal.fit$variable.importance,decreasing=T)
sorted.imp
#  max_kl_shift localsimple_taures  fluctanal_prop_r1        diff2_acf10 
# 0.10168672         0.05156252         0.05038208         0.04988826 
# diff2x_pacf5       diff1x_pacf5   time_level_shift     time_var_shift 
# 0.04824201         0.04791643         0.04709537         0.04494646 
# avg_rR45CO244CO2 
# 0.03917553 

rfSelFinal.fit$confusion.matrix
#  predicted
# true      abiotic biotic
# abiotic      77     12
# biotic        7     44

rfSelFinal.fit$prediction.error
# [1] 0.1357143
1-rfSelFinal.fit$prediction.error
# [1] 0.8642857
predRFFinal.test<-predict(rfSelFinal.fit,data=rfTest.dat)

confusionMatrix(predRFFinal.test$predictions,rfTest.dat$biotic)
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

#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfSelFinal.fit, "./model_output/bioAbio_selVarRF_bestFit.rds")
# save predictions
rfPredTrain.dat<-rfTrain.dat
rfPredTrain.dat$pred<-rfSelFinal.fit$predictions
rfPredTest.dat<-rfTest.dat
rfPredTest.dat$pred<-predRFFinal.test$predictions
write.table(rfPredTrain.dat,"./model_output/bioAbio_selVarRF_train.csv",quote=F,row.names=F,sep=",")
write.table(rfPredTest.dat,"./model_output/bioAbio_selVarRF_test.csv",quote=F,row.names=F,sep=",")


##################
# Part XI: correlation heatmaps for LASSO-NPDR-URFP features and top RF features
##############
### Correlation analysis for LASSO-NPDR-URFP selected features and RF top features
# read in full data
predictors<-read.csv("./data/tsms_biotic_0.3pCO2_predictors.csv")
dats <- predictors %>% mutate_at("biotic",as.factor)
npdr.ind<-which(colnames(dats) %in% npdr.feat)
# subset dats for npdr features
datsNpdr<-dats[,c(npdr.ind,ncol(dats))]

# rf features: could scale node size by variable importance if an interaction
# network is made 
rf.ind<-which(colnames(dats) %in% rf_feat)
datsRF<-dats[,c(rf.ind,ncol(dats))]

# create correlation matrices for heatmaps 
corNpdr.dat<-cor(datsNpdr[,-ncol(datsNpdr)])
head(corNpdr.dat)[,1:3]
#                    diff2_acf1 time_kl_shift fluctanal_prop_r1
# diff2_acf1         1.00000000  -0.277261230       -0.02538663
# time_kl_shift     -0.27726123   1.000000000       -0.21942526
# fluctanal_prop_r1 -0.02538663  -0.219425258        1.00000000
# walker_propcross  -0.50185362   0.005449528        0.18755501
# avg_rR45CO244CO2   0.11018658  -0.262151989        0.26283651
# avg_R45CO244CO2   -0.33371317   0.134978069       -0.11989562

# rf correlation matrix
corRF.dat<-cor(datsRF[,-ncol(datsRF)])


# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
# reorder the correlation matrix
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}


# Reorder the correlation matrix
cormat_npdr <- reorder_cormat(corNpdr.dat)
# get upper tri
upper_tri_npdr <- get_upper_tri(cormat_npdr)

# Reorder the correlation matrix rf
cormat_rf <- reorder_cormat(corRF.dat)
# get upper tri
upper_tri_rf <- get_upper_tri(cormat_rf)

# Melt the correlation matrices for plotting 
melted_cormat_npdr <- melt(upper_tri_npdr, na.rm = TRUE)
head(melted_cormat_npdr)
#                 Var1              Var2       value
# 1         diff2_acf1        diff2_acf1  1.00000000
# 10        diff2_acf1     time_kl_shift -0.27726123
# 11     time_kl_shift     time_kl_shift  1.00000000
# 19        diff2_acf1 fluctanal_prop_r1 -0.02538663
# 20     time_kl_shift fluctanal_prop_r1 -0.21942526
# 21 fluctanal_prop_r1 fluctanal_prop_r1  1.00000000

melted_cormat_rf <- melt(upper_tri_rf, na.rm=TRUE)
head(melted_cormat_rf)
#                Var1             Var2      value
# 1       diff2_acf10      diff2_acf10  1.0000000
# 10      diff2_acf10     max_kl_shift  0.7794767
# 11     max_kl_shift     max_kl_shift  1.0000000
# 19      diff2_acf10 time_level_shift -0.6868586
# 20     max_kl_shift time_level_shift -0.5561541
# 21 time_level_shift time_level_shift  1.0000000


# Create a ggheatmap
# NPDR features
ggheatmap_npdr <- ggplot(melted_cormat_npdr, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 8, hjust = 1),
        axis.text.y=element_text(size=8))+
  coord_fixed()
# Print the heatmap
print(ggheatmap_npdr)

## rf features
# Create a ggheatmap
ggheatmap_rf <- ggplot(melted_cormat_rf, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 8, hjust = 1),
        axis.text.y=element_text(size=8))+
  coord_fixed()
# Print the heatmap
print(ggheatmap_rf)



##################
# Part XII: reGAIN interaction plot for LASSO-NPDR-URFP features
##############
#### reGAIN
# Interaction network for LASSO-NPDR-URFP selected features
# future work: Fortran output for RF-detected interactions
# best-performing model = most parsimonious while preserving high accuracy
# LASSO-NPDR-URFP is the best-performing model
msts.regain <- npdr::regain(datsNpdr,
                            indVarNames=colnames(datsNpdr)[-ncol(datsNpdr)],
                            depVarName="biotic",
                            reg.fam="binomial",
                            nCovs=0,
                            excludeMainEffects=F)
# some warning messages for regression, low variance likely

#### preliminary look at regain results
betas <- as.matrix(msts.regain$stdbetas)
regain.nodiag <- betas
diag(regain.nodiag) <- 0

### Cumulative interactions of each variable IGNORING SIGN
# degree of regain matrix without main effects.
# some interaction regression might fail
regain.nodiag.deg <- rowSums(abs(regain.nodiag))

# Ranking here is cumulative interactions a variable has
regain.nodiag.deg[order(regain.nodiag.deg, decreasing = TRUE)]/length(regain.nodiag.deg)

### Main effects
main.effs.abs <- abs(diag(betas))
# Main effects of each variable
diag(betas)[order(main.effs.abs, decreasing = TRUE)]
# fluctanal_prop_r1        diff2_acf1  avg_rR45CO244CO2        sd_d18O13C 
# 4.0779108        -3.1756680        -2.8308687         2.4025313 
# walker_propcross     time_kl_shift   avg_d45CO244CO2  avg_rd45CO244CO2 
# 2.1952830         0.7247328        -0.5003745        -0.5003745 
# avg_R45CO244CO2 
# -0.5003734 

### filter regain matrix for plotting
# Threshold regain-nodiag matrix for plot
# Interactions can be negative, so do abs
# if you only want to see positive interactions, remove abs
hist(abs(regain.nodiag),breaks=20)
regain.nodiag.adj <- as.matrix(abs(regain.nodiag)>1.3)+0  # remove edges below threshold
regain.nodiag.weight <- regain.nodiag.adj*regain.nodiag

## the following will remove unconnected nodes in the network
# you might want to keep them in case they have a main effect of interest
# keep non-zero degree
rsums<-rowSums(regain.nodiag.adj)
na_mask <- !is.na(rsums)
regain.nodiag.adj <- regain.nodiag.adj[na_mask,na_mask]
regain.nodiag.weight <- regain.nodiag.weight[na_mask,na_mask]
beta_diag <- diag(betas)[na_mask]
rsums2 <- rowSums(regain.nodiag.adj)
degree_mask <- rsums2>0 
regain.nodiag.adj <- regain.nodiag.adj[degree_mask,degree_mask]
regain.nodiag.weight <- regain.nodiag.weight[degree_mask,degree_mask]
beta_diag <- beta_diag[degree_mask]

### adjacency matrix
A.adj <- graph_from_adjacency_matrix(regain.nodiag.adj, mode="undirected")
A.weight <- graph_from_adjacency_matrix(regain.nodiag.weight, mode="undirected", weight=T)
my.layout <- layout_with_fr(A.adj, niter = 10000)

# make nodes with negative main effects green (positive class = abiotic)
vertices.colors <- rep("limegreen", rep(length(V(A.adj))))
vertices.colors[beta_diag>0] <- "gray" 

# make negative interactions green as  well
E(A.adj)$color <- 'limegreen'
E(A.adj)$color[E(A.weight)$weight > 0] <- 'gray'
E(A.adj)$lty <- 1
E(A.adj)$lty[E(A.weight)$weight > 0] <- 2  # line type  

# epistasis-rank centrality
er.ranks <- npdr::EpistasisRank(betas,Gamma_vec = .85, magnitude.sort=T)
er.ranks
#                gene         ER
# 4  walker_propcross  391.77552
# 3 fluctanal_prop_r1  334.79199
# 5  avg_rR45CO244CO2  298.88624
# 2     time_kl_shift -263.18470
# 7  avg_rd45CO244CO2 -210.22951
# 8   avg_d45CO244CO2 -210.22898
# 6   avg_R45CO244CO2 -208.29631
# 1        diff2_acf1  -97.92346
# 9        sd_d18O13C  -34.59078

plot.igraph(A.adj,layout=my.layout, 
            vertex.color=vertices.colors,
            edge.width=abs(E(A.weight)$weight)*1.5, # make edges easier to see
            vertex.label.dist=1.5, 
            vertex.label.color="black",
            #vertex.size = regain.filter.degree # could size by degree
            vertex.size=abs(beta_diag)*5  # size by main effect
            #vertex.size=abs(er.ranks[order(as.numeric(row.names(er.ranks))), 2])*75 # size by ER rank
)

# lets you manually edit the layout and save as a high quality image
tkplot(A.adj,layout=my.layout, 
       vertex.color=vertices.colors,     
       edge.width=abs(E(A.weight)$weight)*1, 
       vertex.label.dist=1.5, vertex.label.color="black",
       #vertex.size = regain.filter.degree # size by degree
       vertex.size=abs(beta_diag)*5  # size by main effect
       #vertex.size=abs(er.ranks[order(as.numeric(row.names(er.ranks))), 2])*75 # size by ER
)
tk_close() # will generate warning



