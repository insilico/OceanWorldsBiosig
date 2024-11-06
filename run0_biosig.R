#### code for feature selection, machine learning classification, and interaction network visualization
#     for main manucscript results in Earth and Space Sciences publication: 
#     "Interpretable Machine Learning Biosignature Detection from Ocean Worlds 
#     Analogue CO2 Isotopologue Data"
# code prepared by Lily Clough, 2022-2023
# last edited: 11/05/2024

library(ranger)
library(caret)
library(dplyr)
library(npdr)
library(doParallel)
library(reshape2)

#### read in distance matrix + training and testing data for 0.3p (0.3% by volume) CO2
# setwd - the following command is specfic to Rstudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# previously computed distance matrix - see RF_replicates_biosig.R for procedure
urfp_0.3p.dist <- read.csv("./run0_data/urfp_dist_0.3p_bioAbio.csv",header=F)
# previously split train/test data - see RF_replicates_biosig.R for procedure
train_0.3p.dat <- read.csv("./run0_data/tsms_0.3pCO2_train_new.csv")
test_0.3p.dat <- read.csv("./run0_data/tsms_0.3pCO2_test_new.csv")


#####################################################################
# 1. RF biosignature classificaiton: full variable  space (all features)
#####################################################################
rm.ind <- which(colnames(train_0.3p.dat) %in% c("Analysis","dataset"))

# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(train_0.3p.dat)[2]-3),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(5,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, savePredictions = T,
                        search="random",
                        allowParallel=T)
start<-Sys.time()
rf1<-train(biotic~., data=train_0.3p.dat[,-rm.ind], 
           method="ranger",
           importance="permutation",
           metric="Accuracy", 
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(train_0.3p.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 50.98078 mins

rf1$bestTune
#  mtry splitrule min.node.size
# 1069   15 hellinger             7   

# best accuracy
rf1$finalModel$prediction.error
# [1] 0.1142857
1-rf1$finalModel$prediction.error
# [1] 0.8857143

# saved tuned parameters
bestFull_mtry<-rf1$bestTune$mtry
bestFull_minNodeSize<-rf1$bestTune$min.node.size
bestFull_splitRule<-rf1$bestTune$splitrule

# tune the number of trees using best params from previous run
fileName<-"maxtrees_out.csv" # for parallel output

# new tuning grid
tuneGrid<-expand.grid(.mtry=bestFull_mtry,
                      .splitrule=bestFull_splitRule,
                      .min.node.size=bestFull_minNodeSize)

trControl<-trainControl(method = "cv",number=5,
                        verboseIter = T,savePredictions = T,
                        search="random",
                        allowParallel=F # parallelize across number of trees
)


start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=train_0.3p.dat[,-rm.ind],
            method="ranger",
            importance="permutation", #"none"?
            metric="Accuracy", 
            trControl=trControl, 
            tuneGrid=tuneGrid,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(train_0.3p.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 1.659232 mins

tree.results<-read.csv("maxtrees_out.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
maxFull_acc<-tree.results[which.max(tree.results$Accuracy),]
maxFull_acc 
# maxtrees  Accuracy
#    3000 0.8200328   
bestFull_maxTree<-5000#maxFull_acc$maxtrees -- use at least 5000 trees



### run a final model
train_0.3p.dat$biotic <- as.factor(train_0.3p.dat$biotic)
rfFullFinal_0.3p.fit <- ranger(biotic ~ ., train_0.3p.dat[,-rm.ind], 
                               keep.inbag = TRUE,
                               num.trees=bestFull_maxTree, 
                               mtry=bestFull_mtry, 
                               importance="permutation", 
                               splitrule = bestFull_splitRule,
                               min.node.size=bestFull_minNodeSize,
                               class.weights = as.numeric(c(1/table(train_0.3p.dat$biotic))),
                               scale.permutation.importance = T,
                               local.importance = T, num.threads=4)
sorted.imp<-sort(rfFullFinal_0.3p.fit$variable.importance,decreasing=T)
rf_full_0.3p.feat<-sorted.imp[1:10] # save top ten for further analysis
names(rf_full_0.3p.feat)
# [1] "max_kl_shift"      "time_level_shift"  "time_var_shift"    "max_level_shift"  
# [5] "arch_acf"          "max_var_shift"     "e_acf1"            "avg_rR45CO244CO2" 
# [9] "diff1x_pacf5"      "fluctanal_prop_r1"

rfFullFinal_0.3p.fit$confusion.matrix
#         predicted
# true      abiotic biotic
# abiotic      82      7
# biotic        9     42

table(train_0.3p.dat$biotic)
# abiotic  biotic 
#      89      51 
rfFullFinal_0.3p.fit$prediction.error
# [1] 0.1142857
1-rfFullFinal_0.3p.fit$prediction.error
# [1] 0.8857143

predFullFinal_0.3p.test<-predict(rfFullFinal_0.3p.fit,data=test_0.3p.dat[,-rm.ind])
test_0.3p.dat$biotic <- as.factor(test_0.3p.dat$biotic)
confusionMatrix(predFullFinal_0.3p.test$predictions,test_0.3p.dat$biotic)
# Accuracy : 0.9118
table(test_0.3p.dat$biotic, predFullFinal_0.3p.test$predictions)
#             predicted
# true        abiotic biotic
# abiotic      21      1
# biotic        2     10 


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfFullFinal_0.3p.fit, "./run0_data/bioAbio_fullVarRF_bestFit_0.3p.rds")

# save predictions
fullTrain.dat<-train_0.3p.dat
fullTrain.dat$pred<-rfFullFinal_0.3p.fit$predictions
fullTest.dat<-test_0.3p.dat
fullTest.dat$pred<-predFullFinal_0.3p.test$predictions
write.table(fullTrain.dat,"./run0_data/bioAbio_fullVarRF_train_0.3p.csv",quote=F,row.names=F,sep=",")
write.table(fullTest.dat,"./run0_data/bioAbio_fullVarRF_test_0.3p.csv",quote=F,row.names=F,sep=",")

##################################################
# 2. NPDR-LURF feature selection for biosignatures
###################################################
# bio_npdr_lurf
start<-Sys.time()
bio_npdr_lurf <- npdr::npdr("biotic", train_0.3p.dat[,-rm.ind],
                            regression.type="binomial",
                            attr.diff.type="numeric-abs",
                            nbd.method="relieff",
                            nbd.metric = "precomputed",
                            external.dist=urfp_0.3p.dist,
                            knn=knnSURF.balanced(train_0.3p.dat$biotic, 
                                                 sd.frac = .5),
                            use.glmnet = T, glmnet.alpha = 1, 
                            glmnet.lower = 0, glmnet.lam="lambda.min",
                            neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                            padj.method="bonferroni", verbose=T)
end<-Sys.time()
npdr.time<-end-start
npdr.time
# Time difference of 3.49109 secs
bio_npdr_lurf_scores<-bio_npdr_lurf %>% tibble::rownames_to_column(var = "features") %>%
  filter(scores!=0, features!="intercept")
bio_npdr_lurf_scores
# lam.min:  0.0002158621* <--- lambda used
# lambda.1se:  0.01179098
#            features       scores
# 1   avg_R45CO244CO2 5.445894e+02
# 2  avg_rR45CO244CO2 3.871818e+02
# 3        sd_d18O13C 1.263945e+02
# 4        diff2_acf1 6.658139e+00
# 5 fluctanal_prop_r1 4.329864e+00
# 6     time_kl_shift 1.418614e-04

# save results
write.table(bio_npdr_lurf_scores,"./run0_data/bio_npdr_lurf_scores_0.3p.csv",sep=",")



#############################################################
# 3. RF biosignature classification: NPDR-LURF selected features
#############################################################

bio_lurf_npdr_ind <- which(colnames(train_0.3p.dat) %in% c("Analysis",bio_npdr_lurf_scores$features,"biotic"))
bio_lurf_0.3p_train.dat <- train_0.3p.dat[,bio_lurf_npdr_ind]
head(bio_lurf_0.3p_train.dat)
bio_lurf_0.3p_test.dat <- test_0.3p.dat[,bio_lurf_npdr_ind]

####### tune RF
rm.ind <- which(colnames(bio_lurf_0.3p_train.dat) =="Analysis")

# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(bio_lurf_0.3p_train.dat)[2]-2),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(1,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, savePredictions = T,
                        search="random",
                        allowParallel=T)
start<-Sys.time()
rf3<-train(biotic~., data=bio_lurf_0.3p_train.dat[,-rm.ind], 
           method="ranger",
           importance="none",
           metric="Accuracy", 
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(bio_lurf_0.3p_train.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 4.146891 mins

rf3$bestTune
#  mtry splitrule min.node.size
#78    2 hellinger            18

# best accuracy
rf3$finalModel$prediction.error
# [1] 0.1285714
1-rf3$finalModel$prediction.error
# [1] 0.8714286

# saved tuned parameters
bestLurf_mtry<-rf3$bestTune$mtry
bestLurf_minNodeSize<-rf3$bestTune$min.node.size
bestLurf_splitRule<-rf3$bestTune$splitrule


# tune the number of trees using best params from previous run
fileName<-"maxtrees_out3.csv" # for parallel output

tuneGrid<-expand.grid(.mtry=bestLurf_mtry,
                      .splitrule=bestLurf_splitRule,
                      .min.node.size=bestLurf_minNodeSize)

trControl<-trainControl(method = "cv",number=5,
                        verboseIter = T,savePredictions = T,
                        search="random",
                        allowParallel=F # parallelize across number of trees
)


start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=bio_lurf_0.3p_train.dat[,-rm.ind],
            method="ranger",
            importance="none", #"none"?
            metric="Accuracy", 
            trControl=trControl, 
            tuneGrid=tuneGrid,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(bio_lurf_0.3p_train.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 33.57202 secs

tree.results<-read.csv("maxtrees_out3.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
maxLurf_acc<-tree.results[which.max(tree.results$Accuracy),]
maxLurf_acc 
#   maxtrees  Accuracy
# 1     3000 0.8848659 
bestLurf_maxTree<-5000#maxFull_acc$maxtrees -- use at least 5000 trees



### run a final model
bio_lurf_0.3p_train.dat$biotic <- as.factor(bio_lurf_0.3p_train.dat$biotic)
rfLurfFinal_0.3p.fit <- ranger(biotic ~ ., bio_lurf_0.3p_train.dat[,-rm.ind], 
                               keep.inbag = TRUE,
                               num.trees=bestLurf_maxTree, 
                               mtry=bestLurf_mtry, 
                               importance="permutation", 
                               splitrule = bestLurf_splitRule,
                               min.node.size=bestLurf_minNodeSize,
                               class.weights = as.numeric(c(1/table(bio_lurf_0.3p_train.dat$biotic))),
                               scale.permutation.importance = T,
                               local.importance = T, num.threads=4)
sorted.imp<-sort(rfLurfFinal_0.3p.fit$variable.importance,decreasing=T)
rf_lurf_0.3p.feat<-sorted.imp
names(rf_lurf_0.3p.feat)
# [1] "diff2_acf1"        "fluctanal_prop_r1" "avg_rR45CO244CO2"  "avg_R45CO244CO2"  
# [5] "sd_d18O13C"        "time_kl_shift" 

rfLurfFinal_0.3p.fit$confusion.matrix
#         predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        9     42

table(bio_lurf_0.3p_train.dat$biotic)
# abiotic  biotic 
#      89      51 
rfLurfFinal_0.3p.fit$prediction.error
# [1] 0.1214286
1-rfLurfFinal_0.3p.fit$prediction.error
# [1] 0.8785714

predLurfFinal_0.3p.test<-predict(rfLurfFinal_0.3p.fit,data=bio_lurf_0.3p_test.dat[,-rm.ind])
bio_lurf_0.3p_test.dat$biotic <- as.factor(bio_lurf_0.3p_test.dat$biotic)
confusionMatrix(predLurfFinal_0.3p.test$predictions,bio_lurf_0.3p_test.dat$biotic)
# Accuracy : 0.8824 
table(bio_lurf_0.3p_test.dat$biotic, predLurfFinal_0.3p.test$predictions)
#         predicted 
# true    abiotic biotic
# abiotic      20      2
# biotic        2     10     


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfLurfFinal_0.3p.fit, "./run0_data/bioAbio_LurfVarRF_bestFit_0.3p.rds")

# save predictions
lurfTrain.dat<-bio_lurf_0.3p_train.dat
lurfTrain.dat$pred<-rfLurfFinal_0.3p.fit$predictions
lurfTest.dat<-bio_lurf_0.3p_test.dat
lurfTest.dat$pred<-predLurfFinal_0.3p.test$predictions
write.table(lurfTrain.dat,"./run0_data/bioAbio_LurfVarRF_train_0.3p.csv",quote=F,row.names=F,sep=",")
write.table(lurfTest.dat,"./run0_data/bioAbio_LurfVarRF_test_0.3p.csv",quote=F,row.names=F,sep=",")




#################################################################
# 4. Probability Forest for local (case or sample-wise) importance 
################################################################
bio_0.3p_train.dat<-read.csv("./run0_data/bioAbio_LurfVarRF_train_0.3p.csv")
bio_0.3p_test.dat<-read.csv("./run0_data/bioAbio_LurfVarRF_test_0.3p.csv")
colnames(bio_0.3p_test.dat)

bio_0.3p_train2.dat <- rbind.data.frame(bio_0.3p_train.dat,bio_0.3p_test.dat)

# full dataset (train+test) so we can analyze sample-wise importance for test data
rm.ind <- which(colnames(bio_0.3p_train2.dat) %in% c("Analysis","pred"))
bio_0.3p_train2.dat$biotic <- as.factor(bio_0.3p_train2.dat$biotic)

bio_0.3p_prob2.rf <- ranger(biotic~., bio_0.3p_train2.dat[,-rm.ind], 
                            keep.inbag = TRUE,
                            num.trees = 5000,
                            mtry = 2, 
                            importance = "permutation", 
                            splitrule = "hellinger", #3
                            min.node.size = 18,
                            class.weights = as.numeric(c(1/table(bio_0.3p_train2.dat$biotic))),
                            scale.permutation.importance = T, 
                            probability = T,
                            local.importance = T, 
                            num.threads = 4)
#bio_0.3p_case<-bio_0.3p_prob.rf$variable.importance.local
#bio_0.3p_prob<-bio_0.3p_prob.rf$predictions

pred <- bio_0.3p_prob2.rf$predictions
bio_0.3p_prob2.rf$prediction.error
# [1] 0.08899029

dat <- bio_0.3p_train2.dat
dat$pred <- pred


# write model to file
saveRDS(bio_0.3p_prob2.rf,"./run0_data/bio_0.3p_prob_rf_npdrLurf_full.rds")
write.table(dat,"./run0_data/bio_0.3p_prob_train_full.csv",row.names=F,quote=F,sep=",")


################################
# 5.RF case-wise importance plots
###################################
# put in order of NPDR-LURF ranking
# read 0.3% CO2 LURF scores for biotic-abiotic
bio_0.3p_npdr_scores <- read.csv("./run0_data/bio_npdr_lurf_scores_0.3p.csv")
bio_0.3p.dat <- read.csv("./run0_data/bio_0.3p_prob_train_full.csv")
colorder<-as.character(bio_0.3p_npdr_scores$features)
colorder
# [1] "avg_R45CO244CO2"   "avg_rR45CO244CO2"  "sd_d18O13C"        "diff2_acf1"       
# [5] "fluctanal_prop_r1" "time_kl_shift" 

# read in npdr-lurf data
bio_0.3p_lurf_train.dat <- read.csv("./run0_data/bioAbio_LurfVarRF_train_0.3p.csv")
bio_0.3p_lurf_test.dat <- read.csv("./run0_data/bioAbio_LurfVarRF_test_0.3p.csv")
bio_0.3p_lurf.dat <- rbind.data.frame(bio_0.3p_lurf_train.dat,
                                      bio_0.3p_lurf_test.dat)

# read in model for case wise importance 
bio_0.3p_lurf.rf <- readRDS("./run0_data/bio_0.3p_prob_rf_npdrLurf_full.rds")
bio_0.3p_loc.imp <- bio_0.3p_lurf.rf$variable.importance.local
head(bio_0.3p_loc.imp)
#   diff2_acf1 time_kl_shift fluctanal_prop_r1 avg_rR45CO244CO2 avg_R45CO244CO2
# 1 0.11680560   0.018461576        0.07364023       0.04335956     0.007201241
# 2 0.02725249   0.008602699        0.03961322       0.01423325     0.014270347
# 3 0.12101689   0.012950210        0.06723124       0.04165740     0.004725950
# 4 0.07424740   0.020319749        0.06356392       0.07953472     0.002881893
# 5 0.02430220   0.006678954        0.03924250       0.01880324     0.019816222
# 6 0.08930793   0.010109625        0.05780251       0.02962774    -0.006705845
#    sd_d18O13C
# 1 0.011507205
# 2 0.075597023
# 3 0.004942055
# 4 0.009106791
# 5 0.066545994
# 6 0.004671038


# put bio_0.3p_loc.imp in order of npdr-lurf scores
loc_imp.df <- as.data.frame(matrix(rep(NA,dim(bio_0.3p_loc.imp)[1]*
                                         (dim(bio_0.3p_loc.imp)[2])), 
                                   ncol=length(colorder)))
colnames(loc_imp.df) <- colorder
head(loc_imp.df)

for(i in seq(1,length(colorder))){
  curr_col <- colorder[i]
  col_ind <- which(colnames(bio_0.3p_loc.imp)==curr_col)
  col_ordered <- bio_0.3p_loc.imp[,col_ind]
  loc_imp.df[,i] <- col_ordered
}
# add analysis numbers
loc_imp.df$Analysis <- bio_0.3p.dat$Analysis
# add biotic and predictions
sum(bio_0.3p_lurf.dat$Analysis==loc_imp.df$Analysis)
# [1] 174
loc_imp.df$pred <- bio_0.3p_lurf.dat$pred
loc_imp.df$biotic <- bio_0.3p_lurf.dat$biotic
loc_imp.df$prob_abiotic <- bio_0.3p.dat$pred.abiotic
loc_imp.df$prob_biotic <- bio_0.3p.dat$pred.biotic

head(loc_imp.df)
#   avg_R45CO244CO2 avg_rR45CO244CO2  sd_d18O13C diff2_acf1 fluctanal_prop_r1
# 1     0.007201241       0.04335956 0.011507205 0.11680560        0.07364023
# 2     0.014270347       0.01423325 0.075597023 0.02725249        0.03961322
# 3     0.004725950       0.04165740 0.004942055 0.12101689        0.06723124
# 4     0.002881893       0.07953472 0.009106791 0.07424740        0.06356392
# 5     0.019816222       0.01880324 0.066545994 0.02430220        0.03924250
# 6    -0.006705845       0.02962774 0.004671038 0.08930793        0.05780251
#   time_kl_shift Analysis   pred biotic prob_abiotic prob_biotic
# 1   0.018461576     2961 biotic biotic    0.1088387   0.8911613
# 2   0.008602699     2962 biotic biotic    0.2656664   0.7343336
# 3   0.012950210     2963 biotic biotic    0.1426624   0.8573376
# 4   0.020319749     2965 biotic biotic    0.1251529   0.8748471
# 5   0.006678954     2966 biotic biotic    0.2668535   0.7331465
# 6   0.010109625     2968 biotic biotic    0.1967870   0.8032130

## save to file
write.table(loc_imp.df,"./run0_data/bio_0.3p_lurf_caseImpProb.csv",row.names=F,quote=F,sep=",")

# use reverse order bc will flip plot 
newCase.imp<-cbind.data.frame(loc_imp.df$time_kl_shift,
                              loc_imp.df$fluctanal_prop_r1,
                              loc_imp.df$diff2_acf1,
                              loc_imp.df$sd_d18O13C,
                              loc_imp.df$avg_rR45CO244CO2,
                              loc_imp.df$avg_R45CO244CO2,
                              loc_imp.df$Analysis,
                              loc_imp.df$biotic,
                              loc_imp.df$pred,
                              loc_imp.df$prob_abiotic,
                              loc_imp.df$prob_biotic)
colnames(newCase.imp)<-c(rev(colorder),"Analysis","biotic","pred","prob_abiotic","prob_biotic")
case.imp<-newCase.imp
head(case.imp)
#   time_kl_shift fluctanal_prop_r1 diff2_acf1  sd_d18O13C avg_rR45CO244CO2
# 1   0.018461576        0.07364023 0.11680560 0.011507205       0.04335956
# 2   0.008602699        0.03961322 0.02725249 0.075597023       0.01423325
# 3   0.012950210        0.06723124 0.12101689 0.004942055       0.04165740
# 4   0.020319749        0.06356392 0.07424740 0.009106791       0.07953472
# 5   0.006678954        0.03924250 0.02430220 0.066545994       0.01880324
# 6   0.010109625        0.05780251 0.08930793 0.004671038       0.02962774
#   avg_R45CO244CO2 Analysis biotic   pred prob_abiotic prob_biotic
# 1     0.007201241     2961 biotic biotic    0.1088387   0.8911613
# 2     0.014270347     2962 biotic biotic    0.2656664   0.7343336
# 3     0.004725950     2963 biotic biotic    0.1426624   0.8573376
# 4     0.002881893     2965 biotic biotic    0.1251529   0.8748471
# 5     0.019816222     2966 biotic biotic    0.2668535   0.7331465
# 6    -0.006705845     2968 biotic biotic    0.1967870   0.8032130


## get data for analyses of interest
# TP, TN, FP, FN
TP.an <- "2962"
TN.an <- "3234"
FP.an <- "5924"
FN.an <- "5894"
inspAn <- c(TP.an,TN.an,FP.an,FN.an)

plotCaseImp<-case.imp[which(case.imp$Analysis %in% inspAn),]
plotCaseImp
#    time_kl_shift fluctanal_prop_r1   diff2_acf1   sd_d18O13C avg_rR45CO244CO2
# 2     0.008602699        0.03961322  0.027252494  0.075597023       0.01423325
# 29   -0.002206524        0.20050618 -0.008107872  0.005605113      -0.03111833
# 94    0.002103596        0.01393334 -0.125109572  0.032739434       0.01278253
# 113  -0.020653148       -0.05786283 -0.099748124 -0.012061768      -0.02749320
# avg_R45CO244CO2 Analysis  biotic    pred prob_abiotic prob_biotic
# 2        0.01427035     2962  biotic  biotic   0.26566636   0.7343336
# 29      -0.00794497     3234 abiotic abiotic   0.81647968   0.1835203
# 94       0.04419012     5894  biotic abiotic   0.57055512   0.4294449
# 113     -0.01227706     5924 abiotic  biotic   0.06194689   0.9380531

library(reshape2)
## created melted df for plotting for each row in plotCaseImp
melted1 <- melt(plotCaseImp[1,],
                id=c("Analysis","biotic","pred","prob_abiotic","prob_biotic")) 
head(melted1)
#   Analysis biotic   pred prob_abiotic prob_biotic          variable       value
# 1     2962 biotic biotic    0.2656664   0.7343336     time_kl_shift 0.008602699
# 2     2962 biotic biotic    0.2656664   0.7343336 fluctanal_prop_r1 0.039613216
# 3     2962 biotic biotic    0.2656664   0.7343336        diff2_acf1 0.027252494
# 4     2962 biotic biotic    0.2656664   0.7343336        sd_d18O13C 0.075597023
# 5     2962 biotic biotic    0.2656664   0.7343336  avg_rR45CO244CO2 0.014233250
# 6     2962 biotic biotic    0.2656664   0.7343336   avg_R45CO244CO2 0.014270347 

melted2 <- melt(plotCaseImp[2,],
                id=c("Analysis","biotic","pred","prob_abiotic","prob_biotic"))
melted3 <- melt(plotCaseImp[3,],
                id=c("Analysis","biotic","pred","prob_abiotic","prob_biotic"))
melted4 <- melt(plotCaseImp[4,],
                id=c("Analysis","biotic","pred","prob_abiotic","prob_biotic"))

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
library(ggplot2)
## plot
case.p1<-ggplot(melted1, 
                aes(x=variable,y=local_importance,fill=variable_permutation)) + 
  geom_bar(position="stack",stat="identity") +
  coord_flip() + 
  scale_fill_manual(values=c("lightblue","salmon")) +
  geom_hline(yintercept=0)  + theme_bw() +
  labs(title="Case-wise variable importance for NPDR-LURF selected features",
       subtitle=paste("Analysis: ",melted1$Analysis[1],
                      "\n","Class: ","\n", "\t"," actual = ",melted1$biotic[1],
                      "\t \t predicted = ",melted1$pred[1],
                      "\nProbability (percent votes): ", "\n","\t abiotic = ",
                      round(melted1$prob_abiotic[1],4),
                      "\t \t biotic = ", round(melted1$prob_biotic[1],4),
                      sep=""),
       caption=paste("local importance for variable m = \n(percent correct votes OOB) - (percent correct votes permuted m)"))
case.p1 # legend is messed up because only one type of sign


case.p2<-ggplot(melted2, 
                aes(x=variable,y=local_importance,fill=variable_permutation)) + 
  geom_bar(position="stack",stat="identity") +
  coord_flip() + 
  scale_fill_manual(values=c("lightblue","salmon")) +
  geom_hline(yintercept=0)  + theme_bw() +
  labs(title="Case-wise variable importance for NPDR-LURF selected features",
       subtitle=paste("Analysis: ",melted2$Analysis[1],
                      "\n","Class: ","\n", "\t"," actual = ",melted2$biotic[1],
                      "\t \t predicted = ",melted2$pred[1],
                      "\nProbability (percent votes): ", "\n","\t abiotic = ",
                      round(melted2$prob_abiotic[1],4),
                      "\t \t biotic = ", round(melted2$prob_biotic[1],4),
                      sep=""),
       caption=paste("local importance for variable m = \n(percent correct votes OOB) - (percent correct votes permuted m)"))
case.p2 

case.p3<-ggplot(melted3, 
                aes(x=variable,y=local_importance,fill=variable_permutation)) + 
  geom_bar(position="stack",stat="identity") +
  coord_flip() + 
  scale_fill_manual(values=c("lightblue","salmon")) +
  geom_hline(yintercept=0)  + theme_bw() +
  labs(title="Case-wise variable importance for NPDR-LURF selected features",
       subtitle=paste("Analysis: ",melted3$Analysis[1],
                      "\n","Class: ","\n", "\t"," actual = ",melted3$biotic[1],
                      "\t \t predicted = ",melted3$pred[1],
                      "\nProbability (percent votes): ", "\n","\t abiotic = ",
                      round(melted3$prob_abiotic[1],4),
                      "\t \t biotic = ", round(melted3$prob_biotic[1],4),
                      sep=""),
       caption=paste("local importance for variable m = \n(percent correct votes OOB) - (percent correct votes permuted m)"))
case.p3

case.p4<-ggplot(melted4, 
                aes(x=variable,y=local_importance,fill=variable_permutation)) + 
  geom_bar(position="stack",stat="identity") +
  coord_flip() + 
  scale_fill_manual(values=c("salmon","lightblue")) + # ** "lightblue",
  geom_hline(yintercept=0)  + theme_bw() +
  labs(title="Case-wise variable importance for NPDR-LURF selected features",
       subtitle=paste("Analysis: ",melted4$Analysis[1],
                      "\n","Class: ","\n", "\t"," actual = ",melted4$biotic[1],
                      "\t \t predicted = ",melted4$pred[1],
                      "\nProbability (percent votes): ", "\n","\t abiotic = ",
                      round(melted4$prob_abiotic[1],4),
                      "\t \t biotic = ", round(melted4$prob_biotic[1],4),
                      sep=""),
       caption=paste("local importance for variable m = \n(percent correct votes OOB) - (percent correct votes permuted m)"))
case.p4 # legend messed up because only one sign - edit plot command 


##############################################################
# 6. RAIN: Regression based Association-Interaction Network
###############################################################
library(igraph) #for plotting networks

# read in full data for regression
bio_0.3p.dat <- read.csv("./run0_data/bio_0.3p_prob_train_full.csv")

rm.ind <- which(colnames(bio_0.3p.dat) %in% c("Analysis","pred.abiotic","pred.biotic"))
bio.ind <- which(colnames(bio_0.3p.dat)=="biotic")
bio_0.3p.dat$biotic <- as.factor(bio_0.3p.dat$biotic)
bio_0.3p_regain <- npdr::regain(bio_0.3p_train2.dat,
                                indVarNames=colnames(bio_0.3p.dat)[-c(rm.ind,bio.ind)],
                                depVarName="biotic",
                                reg.fam="binomial",
                                nCovs=0,
                                excludeMainEffects=F)

#### preliminary look at regain results
betas <- as.matrix(bio_0.3p_regain$stdbetas)
regain.nodiag <- betas
diag(regain.nodiag) <- 0

### Cumulative interactions of each variable IGNORING SIGN
# degree of regain matrix without main effects.
# some interaction regression might fail
regain.nodiag.deg <- rowSums(abs(regain.nodiag))

# Ranking here is cumulative interactions a variable has
regain.nodiag.deg[order(regain.nodiag.deg, decreasing = TRUE)]/length(regain.nodiag.deg)
#   diff2_acf1 fluctanal_prop_r1        sd_d18O13C  avg_rR45CO244CO2 
# 2.140080          1.369768          1.354005          1.162732 
# time_kl_shift   avg_R45CO244CO2 
# 1.069926          1.032040

### Main effects
main.effs.abs <- abs(diag(betas))
# Main effects of each variable
diag(betas)[order(main.effs.abs, decreasing = TRUE)]
#fluctanal_prop_r1        diff2_acf1  avg_rR45CO244CO2        sd_d18O13C 
# 4.0779108        -3.1756680        -2.8308687         2.4025313 
# time_kl_shift   avg_R45CO244CO2 
# 0.7247328        -0.5003734

### filter regain matrix for plotting
# Threshold regain-nodiag matrix for plot
# Interactions can be negative, so do abs
# if you only want to see positive interactions, remove abs
hist(abs(regain.nodiag),breaks=20)
regain.nodiag.adj <- as.matrix(abs(regain.nodiag)>1.2)+0  # remove edges below threshold
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

# make nodes with negative main effects ?
colnames(bio_0.3p.dat)
t.test(formula=sd_d18O13C~biotic,data=bio_0.3p.dat)
# mean in group abiotic  mean in group biotic 
#           0.003328698           0.004003809 
# biotic has larger mean
glm(biotic~sd_d18O13C,data=bio_0.3p.dat, family=binomial)
# Coefficients:
# (Inntercept)   sd_d18O13C  
# -1.404      231.103      # positive coefficient
# abiotic is the reference group
# positive beta means increased chance of being biotic
vertices.colors <- rep("limegreen", rep(length(V(A.adj))))
vertices.colors[beta_diag<0] <- "gray" 

# make negative interactions green as  well
E(A.adj)$color <- 'limegreen'
E(A.adj)$color[E(A.weight)$weight < 0] <- 'gray'
E(A.adj)$lty <- 1
E(A.adj)$lty[E(A.weight)$weight < 0] <- 2  # line type  

# epistasis-rank centrality
er.ranks <- npdr::EpistasisRank(betas,Gamma_vec = .85, magnitude.sort=T)
er.ranks
#  gene          ER
# 6        sd_d18O13C  0.54083111
# 5   avg_R45CO244CO2  0.43958932
# 2     time_kl_shift  0.24583652
# 4  avg_rR45CO244CO2 -0.17361776
# 1        diff2_acf1 -0.07122914
# 3 fluctanal_prop_r1  0.01858995             

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

