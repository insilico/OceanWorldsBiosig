#### biotic-abiotic feature selection with NPDR and classification using random forest
# load libraries
library(ranger)
library(reshape2)
library(dplyr)
library(npdr)
#library(caret)
#library(doParallel)
#library(foreach)
library(forcats)
library(glm2)
library(igraph)

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


##################
# Part II: unsupervised random forest proximity (URFP) distance
#################
# read in previously created distance matrix (see "npdr_rf_biosig.R" Part II for code)
urfp.dist<-read.csv("./npdr_dist_output/urfp_dist_bioAbio.csv")


##################


##################
# Part III: LASSO-penalized NPDR feature selection with URFP distance 
# (see "npdr_rf_biosig.R" Part III for code)
#############
npdr_scores<-read.csv("./npdr_dist_output/lasso_npdr_urfp_features.csv")
npdr_scores
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


##################


##################
# Part IV: parameter tuning (caret) and supervised RF in the full variable space
# (see "npdr_rf_biosig.R" Part IV for full code)
#############
# tuned parameters for the full variable space
bestFull_mtry <- 15
bestFull_minNodeSize <- 9
bestFull_splitRule <- "extratrees"
bestFull_maxTree <- 6000

# read in model, data + predictions
rfFullFinal.fit <- readRDS("./model_output/bioAbio_fullVarRF_bestFit.rds")
fullTrain.dat <- read.csv("./model_output/bioAbio_fullVarRF_train.csv")
fullTest.dat <- read.csv("./model_output/bioAbio_fullVarRF_test.csv")

sorted.imp<-sort(rfFullFinal.fit$variable.importance,decreasing=T)
rf.feat<-sorted.imp[1:9] # save top nine for further analysis
rf.feat
# max_kl_shift  fluctanal_prop_r1 localsimple_taures   avg_rR45CO244CO2 
# 0.03090578         0.02726200         0.01860284         0.01340862 
# diff1x_pacf5   time_level_shift     time_var_shift       diff2x_pacf5 
# 0.01151226         0.01078819         0.01049708         0.01028811 
# diff2_acf10 
# 0.01021133 

rfFullFinal.fit$confusion.matrix
#         predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        9     42

predFullFinal.test<-predict(rfFullFinal.fit,data=test.dat)
confusionMatrix(as.factor(fullTest.dat$pred),as.factor(test.dat$biotic))
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


##################


##################
# Part V: parameter tuning and supervised RF in the LASSO-NPDR-URFP variable space
# (see "npdr_rf_biosig.R" Part V for full code)
#################
npdr.feat<-as.character(npdr_scores$features)
npdr.ind<-which(colnames(train.dat) %in% c(npdr.feat,"biotic"))
# make train and test data
selTrain.dat<-train.dat
selTest.dat<-test.dat
selTrain.dat<-selTrain.dat[,npdr.ind]
selTest.dat<-selTest.dat[,npdr.ind]
dim(selTrain.dat)
# [1] 140  10
dim(selTest.dat)
# [1] 34 10

# read in model, data + predictions
rfNpdrFinal.fit <- readRDS("./model_output/bioAbio_npdrRF_bestFit.rds")
selTrainPred.dat <- read.csv("./model_output/bioAbio_npdrVarRF_train.csv")
selTestPred.dat <- read.csv("./model_output/bioAbio_npdrVarRF_test.csv")

# tuned parameters for the full variable space
bestNpdr_mtry <- 4
bestNpdr_minNodeSize <- 8
bestNpdr_splitRule <- "extratrees"
bestNpdr_maxTree <- 6000 # match full variable since number of tuned trees is fewer


sorted.imp<-sort(rfNpdrFinal.fit$variable.importance,decreasing=T)
sorted.imp
# diff2_acf1 fluctanal_prop_r1  avg_rR45CO244CO2  avg_rd45CO244CO2   avg_d45CO244CO2 
# 0.07963442        0.07558188        0.05111672        0.02488842        0.02477735 
# avg_R45CO244CO2     time_kl_shift  walker_propcross        sd_d18O13C 
# 0.02416445        0.02240897        0.01643164        0.01101773

rfNpdrFinal.fit$confusion.matrix
#         predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        6     45

confusionMatrix(as.factor(selTestPred.dat$pred),as.factor(selTest.dat$biotic))
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


##################


##################
# Part VI: LASSO-penalized NPDR feature selection with Manhattan 
#          distance and supervised RF
# (see "npdr_rf_biosig.R" Part VI for full code)
##################
npdrMan_scores <- read.csv("./npdr_dist_output/lasso_npdr_manhattan_features.csv")
npdrMan_scores
#             features       scores
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

# tuned parameters
bestMan_mtry <- 2
bestMan_minNodeSize <- 13
bestMan_splitRule<- "hellinger"
bestMan_maxTree <- 6000

# training and testing data using selected features
manTrain.dat<-train.dat
manTest.dat<-test.dat
npdrMan.feat<-npdrMan_scores$features
npdrMan.ind<-which(colnames(train.dat) %in% c(npdrMan.feat,"biotic"))

manTrain.dat<-manTrain.dat[,npdrMan.ind]
manTest.dat<-manTest.dat[,npdrMan.ind]

# model, data + predictions
rfManFinal.fit <- readRDS("./model_output/bioAbio_npdrMan_bestFit.rds")
manTrainPred.dat <- read.csv("./model_output/bioAbio_npdrManVarRF_train.csv")
manTestPred.dat <- read.csv("./model_output/bioAbio_npdrManVarRF_test.csv")

# training confusion matrix
rfManFinal.fit$confusion.matrix
#         predicted
# true      abiotic biotic
# abiotic      82      7
# biotic       11     40

# testing confusion matrix
confusionMatrix(as.factor(manTestPred.dat$pred),as.factor(manTest.dat$biotic))
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



##################


##################
# Part VII: probability forest for LASSO-NPDR-URFP features
# (see "npdr_rf_biosig.R" Part VII for code)
##################
classProb.df<-read.csv("./samplewise_output/bioAbio_npdrURFP_RF_classProb.csv")
head(classProb.df)
#   actual   pred Analysis trainTest   abiotic    biotic
# 1 biotic biotic     2960     train 0.2351666 0.7648334
# 2 biotic biotic     2961     train 0.3436013 0.6563987
# 3 biotic biotic     2962     train 0.2650792 0.7349208
# 4 biotic biotic     2963     train 0.4197209 0.5802791
# 5 biotic biotic     2965     train 0.4271395 0.5728605
# 6 biotic biotic     2966     train 0.3088334 0.6911666


##################


##################
# Part VIII: case-wise variable importance for all samples (training + testing)
##################
plotCaseImp <- read.csv("./samplewise_output/caseWiseImp_inspectAnalyses.csv")

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


##################
# Part IX: create biotic and abiotic prototype plots
##################
protoNpdr.proc <- read.csv("./samplewise_output/bioAbio_NpdrURFPVars_prototypes.csv")

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
protoRF.proc2 <- read.csv("./samplewise_output/bioAbio_rfTopVars_prototypes.csv")

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


##################
# Part X: supervised RF for top nine RF importance variables
##################

# train and test data
rf_feat<-names(rf.feat)
rfTrain.dat<-train.dat[,which(colnames(train.dat) %in% c(rf_feat,"biotic"))]
rfTest.dat<-test.dat[,which(colnames(test.dat) %in% c(rf_feat,"biotic"))]

# model, data + predictions
rfSelFinal.fit <- readRDS("./model_output/bioAbio_selVarRF_bestFit.rds")
rfPredTrain.dat <- read.csv("./model_output/bioAbio_selVarRF_train.csv")
rfPredTest.dat <- read.csv("./model_output/bioAbio_selVarRF_test.csv")

sorted.imp<-sort(rfSelFinal.fit$variable.importance,decreasing=T)
sorted.imp
# max_kl_shift localsimple_taures  fluctanal_prop_r1        diff2_acf10 
# 0.10168672         0.05156252         0.05038208         0.04988826 
# diff2x_pacf5       diff1x_pacf5   time_level_shift     time_var_shift 
# 0.04824201         0.04791643         0.04709537         0.04494646 
# avg_rR45CO244CO2 
# 0.03917553 

# training confusion matrix
rfSelFinal.fit$confusion.matrix
#         predicted
# true      abiotic biotic
# abiotic      77     12
# biotic        7     44

# testing confusion matrix
confusionMatrix(as.factor(rfPredTest.dat$pred),as.factor(rfTest.dat$biotic))
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

##################


##################
# Part XI: correlation heatmaps for LASSO-NPDR-URFP features and top RF features
##################
# full dataset
predictors<-read.csv("./data/tsms_biotic_0.3pCO2_predictors.csv")
dats <- predictors %>% mutate_at("biotic",as.factor)
npdr.ind<-which(colnames(dats) %in% npdr.feat)
# subset dats for npdr features
datsNpdr<-dats[,c(npdr.ind,ncol(dats))]

# create correlation matrices for heatmaps 
corNpdr.dat<-cor(datsNpdr[,-ncol(datsNpdr)])
head(corNpdr.dat)[,1:3]

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

# Melt the correlation matrices for plotting 
melted_cormat_npdr <- melt(upper_tri_npdr, na.rm = TRUE)

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


## compare with RF features 
rf.ind<-which(colnames(dats) %in% rf_feat)
datsRF<-dats[,c(rf.ind,ncol(dats))]
# rf correlation matrix
corRF.dat<-cor(datsRF[,-ncol(datsRF)])

# Reorder the correlation matrix
cormat_rf <- reorder_cormat(corRF.dat)
# get upper tri
upper_tri_rf <- get_upper_tri(cormat_rf)
melted_cormat_rf <- melt(upper_tri_rf, na.rm=TRUE)


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


##################
# Part XII: reGAIN interaction plot for LASSO-NPDR-URFP features
##################
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
##################

