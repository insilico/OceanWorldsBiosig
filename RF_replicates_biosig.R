#### code for feature selection and machine learning classification replicates 
#     for supplemental results in Earth and Space Sciences publication: 
#     "Interpretable Machine Learning Biosignature Detection from Ocean Worlds 
#     Analogue CO2 Isotopologue Data"
# code prepared by Lily Clough, 2022-2023
# last edited: 11/05/2024

### load libraries
#library(QCIRMS)
#library(isoreader)
library(ranger)
#library(isotree)
library(caret)
library(dplyr)
library(npdr)
library(doParallel)
library(reshape2)

### set working directory - the following command is specfic to Rstudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


##############################################
# 1. read in predictor data -all TSMS features
##############################################
tsms.dat <- read.csv("./replicates_data/tsms_0.3pCO2_predictors.csv")
dim(tsms.dat)
# [1] 174 107
head(colnames(tsms.dat))
# [1] "Analysis"    "x_acf1"      "x_acf10"     "diff1_acf1"  "diff1_acf10" "diff2_acf1"
tail(colnames(tsms.dat))
# [1] "avg_d18O13C"       "sd_d18O13C"        "avg_calib_d18O16O" "sd_calib_d18O16O" 
# [5] "biotic"            "dataset"



##################################################
# 2. split into train and test: 5 more replicates
###################################################

# split the rest into test/train
#set.seed(123)
inTrain <- createDataPartition(
  y = tsms.dat$biotic, # the outcome data are needed
  p = .8, # The percentage of data in the training set
  list = FALSE
)
# run inTrain then assign to new training set
inTrain5 <- inTrain
inTrain4 <- inTrain
inTrain3 <- inTrain
inTrain2 <- inTrain
inTrain1 <- inTrain


train1.dat <- tsms.dat[inTrain1,] %>% select(!dataset)
train2.dat <- tsms.dat[inTrain2,] %>% select(!dataset)
train3.dat <- tsms.dat[inTrain3,] %>% select(!dataset)
train4.dat <- tsms.dat[inTrain4,] %>% select(!dataset)
train5.dat <- tsms.dat[inTrain5,] %>% select(!dataset)

test1.dat <- tsms.dat[-inTrain1,] %>% select(!dataset)
test2.dat <- tsms.dat[-inTrain2,] %>% select(!dataset)
test3.dat <- tsms.dat[-inTrain3,] %>% select(!dataset)
test4.dat <- tsms.dat[-inTrain4,] %>% select(!dataset)
test5.dat <- tsms.dat[-inTrain5,] %>% select(!dataset)

####################################################################
# 3. Unsupervised Random Forest Proximity (URFP) distance matrices 
###################################################################

## function to extract RF proximity from URF
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

# create synthetic data for unsupervised RF, remove label cols
rm.ind <- which(colnames(train1.dat) %in% c("Analysis","biotic"))

### loop for URF and extracting proximity distance matrices
train.list <- list(train1.dat,train2.dat,train3.dat,train4.dat,train5.dat)
prox.list <- list()
for(i in seq(1,length(train.list))){
  synth <- as.data.frame(lapply(train.list[[i]][,-rm.ind], 
                                function(x) {sample(x, length(x), replace = TRUE)}))
  synth.dat<-rbind(data.frame(y="real",train.list[[i]][,-rm.ind]),
                      data.frame(y="synth",synth))
  synth.dat$y<-as.factor(synth.dat$y)
  # URF
  urf.fit <- ranger(y~., synth.dat, keep.inbag = TRUE,
                    num.trees=5000, mtry=2, 
                    importance="none",
                    local.importance = F, 
                    num.threads=4)
  prox <- extract_proximity_oob(urf.fit, synth.dat)[1:nrow(train.list[[i]]), 
                                                       1:nrow(train.list[[i]])]
  urfp.dist<-sqrt(1-prox)
  fileName <- paste("./replicates_data/dist","_",i,".csv",sep="")
  
  write.table(urfp.dist,fileName,row.names=F,col.names=F,quote=F,sep=",")
  prox.list[[i]] <- urfp.dist
}




######################################################
# 4. read in npdr scores from run 0 (see run0_biosig.R)
#######################################################
lurf_scores <- read.csv("./replicates_data/bio_npdr_lurf_scores_0.3p.csv")
lurf_feat <- lurf_scores$features
lurf_feat
# [1] "avg_R45CO244CO2"   "avg_rR45CO244CO2"  "sd_d18O13C"        "diff2_acf1"       
# [5] "fluctanal_prop_r1" "time_kl_shift" 

urf_scores <- read.csv("./replicates_data/bio_npdr_urf_scores_0.3p.csv")
urf_feat <- urf_scores$x
urf_feat
# [1] "avg_rR45CO244CO2"  "avg_R13C12C"       "avg_d13C12C"       "avg_R45CO244CO2"  
# [5] "avg_rd45CO244CO2"  "avg_d45CO244CO2"   "sd_d18O13C"        "fluctanal_prop_r1"
# [9] "time_kl_shift"     "diff2_acf1" 

# NPDR manhattan from previous analysis (see run0_biosig.R), for correlation comparison later
man_feat <- c("avg_R45CO244CO2","avg_rR45CO244CO2","sd_d18O13C","motiftwo_entro3",
              "avg_rR46CO244CO2", "fluctanal_prop_r1","walker_propcross",
              "time_kl_shift")



########################################
# 5. NPDR-LMan for Run 0 train/test split
########################################
# read in data

lman_train.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_train_0.3p.csv")
head(colnames(lman_train.dat))
# [1] "Analysis"    "x_acf1"      "x_acf10"     "diff1_acf1"  "diff1_acf10" "diff2_acf1" 
tail(colnames(lman_train.dat))
# [1] "sd_d18O13C"        "avg_calib_d18O16O" "sd_calib_d18O16O"  "biotic"           
# [5] "dataset"           "pred"
dim(lman_train.dat)
# [1] 140 108

lman_test.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_test_0.3p.csv")
# remove label columns
lman_train0.dat <- lman_train.dat[,-c(107,108)]
lman_test0.dat <- lman_test.dat[,-c(107,108)]
rm.ind <- which(colnames(lman_train0.dat) %in% c("Analysis"))

bio_npdr_lman0 <- npdr::npdr("biotic", lman_train0.dat[,-rm.ind],
                            regression.type="binomial",
                            attr.diff.type="numeric-abs",
                            nbd.method="relieff",
                            nbd.metric = "manhattan",
                            knn=knnSURF.balanced(lman_train0.dat$biotic, 
                                                 sd.frac = .5),
                            use.glmnet = T, glmnet.alpha = 1, 
                            glmnet.lower = 0, glmnet.lam="lambda.min",
                            neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                            padj.method="bonferroni", verbose=T)
bio_npdr_lman_scores<-bio_npdr_lman0 %>% tibble::rownames_to_column(var = "features") %>%
  filter(scores!=0, features!="intercept")
# lambda.min:  0.0001919906* 
# lambda.1se:  0.0126317
bio_npdr_lman_scores
#           features       scores
# 1   avg_R45CO244CO2 3.148782e+04
# 2  avg_rR45CO244CO2 2.022014e+02
# 3        sd_d18O13C 1.234900e+02
# 4   motiftwo_entro3 4.003661e+01
# 5  avg_rR46CO244CO2 2.192526e+01
# 6 fluctanal_prop_r1 1.158598e+01
# 7  walker_propcross 9.996261e+00
# 8     time_kl_shift 7.080369e-05

write.table(bio_npdr_lman_scores,"./replicates_data/bio_npdr_lman_scores_run0.csv",row.names=F,quote=F,sep=",")

lman_feat <- bio_npdr_lman_scores$features

lman_train0.dat <- lman_train0.dat[,which(colnames(lman_train0.dat) %in% 
                                            c("Analysis",lman_feat,"biotic"))]
lman_test0.dat <- lman_test0.dat[,which(colnames(lman_test0.dat) %in% 
                                          c("Analysis",lman_feat,"biotic"))]

colnames(lman_train0.dat)
# [1] "Analysis"          "time_kl_shift"     "fluctanal_prop_r1" "motiftwo_entro3"  
# [5] "walker_propcross"  "avg_rR45CO244CO2"  "avg_rR46CO244CO2"  "avg_R45CO244CO2"  
# [9] "sd_d18O13C"        "biotic" 


##########################################
# 6. NPDR-Manhattan for Run 0 train/test split
###########################################
man_train.dat <- lman_train.dat[,-c(107,108)]
man_test.dat <- lman_test.dat[,-c(107,108)]
rm.ind <- which(colnames(man_train.dat) %in% c("Analysis"))
head(man_train.dat)

bio_npdr_man <- npdr::npdr("biotic", man_train.dat[,-rm.ind],
                             regression.type="binomial",
                             attr.diff.type="numeric-abs",
                             nbd.method="relieff",
                             nbd.metric = "manhattan",
                             knn=knnSURF.balanced(man_train.dat$biotic, 
                                                  sd.frac = .5),
                             use.glmnet = F, glmnet.alpha = 1, 
                             glmnet.lower = 0, glmnet.lam="lambda.min",
                             neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                             padj.method="bonferroni", verbose=T)
bio_npdr_man_scores<-bio_npdr_man %>% filter(pval.adj<.05) %>% pull(att)
bio_npdr_man_scores
# [1] "avg_R45CO244CO2"      "avg_d45CO244CO2"      "avg_rd45CO244CO2"    
# [4] "avg_d13C12C"          "avg_R13C12C"          "fluctanal_prop_r1"
# [7] "motiftwo_entro3"      "sd_d18O13C"           "avg_rR45CO244CO2"    
# [10] "avg_rR46CO244CO2"     "sd_R13C12C"           "sd_d13C12C"          
# [13] "sd_d45CO244CO2"       "sd_rd45CO244CO2"      "sd_R45CO244CO2"      
# [16] "sd_rR45CO244CO2"      "unitroot_kpss"        "avg_d18O13C"         
# [19] "trend"                "outlierinclude_mdrmd" "diff2_acf1"          
# [22] "linearity"
write.table(bio_npdr_man_scores,"./replicates_data/bio_npdr_man_run0.csv",quote=F,
            row.names=F,sep=",")

########################################################
## 7. make correlation heatmap for NPDR-Manhattan features
########################################################

# create correlation matrices for heatmaps 
man.dat <- man_train.dat[,which(colnames(man_train.dat) %in% 
                          c("Analysis",bio_npdr_man_scores,"biotic"))]
head(man.dat)
rm.ind <- c(which(colnames(man.dat) %in% c("Analysis","biotic")))
manCor.dat<-cor(man.dat[,-rm.ind])
head(manCor.dat)[,1:3]
#   diff2_acf1       trend  linearity
# diff2_acf1         1.00000000 -0.98281822  0.8531657
# trend             -0.98281822  1.00000000 -0.8861865
# linearity          0.85316574 -0.88618646  1.0000000
# unitroot_kpss     -0.81551188  0.86405955 -0.9859960
# fluctanal_prop_r1 -0.09085064  0.09949042  0.1550603
# motiftwo_entro3    0.71498385 -0.79538180  0.9164074

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
cormat <- reorder_cormat(manCor.dat)
# get upper tri
upper_tri <- get_upper_tri(cormat)

# Melt the correlation matrices for plotting 
melted_cormat <- melt(upper_tri, na.rm = TRUE)
head(melted_cormat)
#             Var1          Var2     value
#1            diff2_acf1           diff2_acf1 1.0000000
# 23           diff2_acf1 outlierinclude_mdrmd 0.5889869
# 24 outlierinclude_mdrmd outlierinclude_mdrmd 1.0000000
# 45           diff2_acf1            linearity 0.8531657
# 46 outlierinclude_mdrmd            linearity 0.7726470
# 47            linearity            linearity 1.0000000

# Create a ggheatmap
# NPDR Manhattan features
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
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
print(ggheatmap)

write.table(manCor.dat,"./replicates_data/npdrManhattan_correlation_run0.csv",
            row.names=TRUE,quote=F,sep=",")



#########################################
# 8. Tune RF model for LMan Features: Run 0
########################################
# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(lman_train0.dat)[2]-2),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(1,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, 
                        search="random",
                        allowParallel=T)
rm.ind
#[1] 1

start<-Sys.time()
rfd<-train(biotic~., data=lman_train0.dat[,-rm.ind], 
           method="ranger",
           importance="none",
           metric="Accuracy",
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(lman_train0.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 7.361628 mins

rfd$bestTune
#    mtry splitrule min.node.size
#155    3 hellinger             5

# best accuracy
rfd$finalModel$prediction.error
# [1] 0.1357143
1-rfd$finalModel$prediction.error
# [1] 0.8642857

# saved tuned parameters
lMan_mtry<-rfd$bestTune$mtry
lMan_minNodeSize<-rfd$bestTune$min.node.size
lMan_splitRule<-rfd$bestTune$splitrule


# tune the number of trees using best params from previous run
fileName<-"maxtrees_outd.csv" # for parallel output

tuneGrid<-expand.grid(.mtry=lMan_mtry,
                      .splitrule=lMan_splitRule,
                      .min.node.size=lMan_minNodeSize)

trControl<-trainControl(method = "cv",number=5,
                        verboseIter = T, savePredictions=TRUE,
                        search="random",
                        allowParallel=F # parallelize across number of trees
)

start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=lman_train0.dat[,-rm.ind],
            method="ranger",
            importance="none", #"none"?
            metric="Accuracy", 
            trControl=trControl, 
            tuneGrid=tuneGrid,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(lman_train0.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 42.96095 secs

tree.results<-read.csv("maxtrees_outd.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
max_acc<-tree.results[which.max(tree.results$Accuracy),]
max_acc 
#   maxtrees  Accuracy
#1     3000 0.8713373
lMan_maxTree<-5000#maxFull_acc$maxtrees --> use at least 5000 trees



### run a final model
lman_train0.dat$biotic <- as.factor(lman_train0.dat$biotic)
head(lman_train0.dat)

rfFinal.fit <- ranger(biotic~., lman_train0.dat[,-rm.ind], 
                      keep.inbag = TRUE,
                      num.trees=lMan_maxTree, 
                      mtry=lMan_mtry, 
                      importance="permutation", 
                      splitrule = lMan_splitRule,
                      min.node.size=lMan_minNodeSize,
                      class.weights = as.numeric(c(1/table(lman_train0.dat$biotic))),
                      scale.permutation.importance = T,
                      local.importance = T, num.threads=4)
sorted.imp<-sort(rfFinal.fit$variable.importance,decreasing=TRUE)
rf.feat<-sorted.imp # save top nine for further analysis
names(rf.feat)
# [1] "fluctanal_prop_r1" "avg_rR45CO244CO2"  "motiftwo_entro3"   "walker_propcross" 
# [5] "avg_R45CO244CO2"   "avg_rR46CO244CO2"  "time_kl_shift"     "sd_d18O13C"    
rfFinal.fit$confusion.matrix 
#       predicted
# true      abiotic biotic
# abiotic      80      9
# biotic        9     42   

rfFinal.fit$prediction.error 
#[1] 0.1285714
1-rfFinal.fit$prediction.error
# [1] 0.8714286

predFinal.test<-predict(rfFinal.fit,data=lman_test0.dat[,-rm.ind])
lman_test0.dat$biotic <- as.factor(lman_test0.dat$biotic)
confusionMatrix(predFinal.test$predictions,lman_test0.dat[,-rm.ind]$biotic)
#     Reference
# Prediction abiotic biotic
# abiotic      20      4
# biotic        2      8

# Accuracy : 0.7879 

###############################################
# 9. make correlation heatmap for NPDR-URF features
###################################################

# get NPDR-URF features
dats <- tsms.dat %>% mutate_at("biotic",as.factor) %>% select(!dataset)
npdr.ind<-which(colnames(dats) %in% urf_feat)
# subset dats for npdr features
datsNpdr<-dats[,c(npdr.ind,ncol(dats))]
colnames(datsNpdr) # check

# create correlation matrices for heatmaps 
corNpdr.dat<-cor(datsNpdr[,-ncol(datsNpdr)])
head(corNpdr.dat)[,1:3]
#                    diff2_acf1 time_kl_shift fluctanal_prop_r1
# diff2_acf1         1.00000000    -0.2772612       -0.02538663
# time_kl_shift     -0.27726123     1.0000000       -0.21942526
# fluctanal_prop_r1 -0.02538663    -0.2194253        1.00000000
# avg_rR45CO244CO2   0.11018658    -0.2621520        0.26283651
# avg_R45CO244CO2   -0.33371317     0.1349781       -0.11989562
# avg_rd45CO244CO2  -0.33371256     0.1349782       -0.11989621


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

# Melt the correlation matrices for plotting 
melted_cormat_npdr <- melt(upper_tri_npdr, na.rm = TRUE)
head(melted_cormat_npdr)
#                 Var1              Var2       value
# 1         diff2_acf1        diff2_acf1  1.00000000
# 11        diff2_acf1 fluctanal_prop_r1 -0.02538663
# 12 fluctanal_prop_r1 fluctanal_prop_r1  1.00000000
# 21        diff2_acf1     time_kl_shift -0.27726123
# 22 fluctanal_prop_r1     time_kl_shift -0.21942526
# 23     time_kl_shift     time_kl_shift  1.00000000

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



###############################################
# 10. Feature Selection Replicates: NPDR-LURF
############################################
head(colnames(train.list[[1]]))
# [1] "Analysis"    "x_acf1"      "x_acf10"     "diff1_acf1"  "diff1_acf10" "diff2_acf1" 
tail(colnames(train.list[[1]]))
# [1] "sd_pkArea"         "avg_d18O13C"       "sd_d18O13C"        "avg_calib_d18O16O"
# [5] "sd_calib_d18O16O"  "biotic" 

rm.ind <- which(colnames(train.list[[1]]) %in% c("Analysis"))

## NPDR-LURF loop for replicates
npdr_lurf.list<-list()
for(i in seq(1,length(prox.list))){
  head(train.list[[i]])
  bio_npdr_lurf <- npdr::npdr("biotic", train.list[[i]][,-rm.ind],
                                  regression.type="binomial",
                                  attr.diff.type="numeric-abs",
                                  nbd.method="relieff",
                                  nbd.metric = "precomputed",
                                  external.dist=prox.list[[i]],
                                  knn=knnSURF.balanced(train.list[[i]]$biotic, 
                                                       sd.frac = .5),
                                  use.glmnet = T, glmnet.alpha = 1, 
                                  glmnet.lower = 0, glmnet.lam="lambda.min",
                                  neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                                  padj.method="bonferroni", verbose=T)
  bio_npdr_lurf_scores<-bio_npdr_lurf %>% tibble::rownames_to_column(var = "features") %>%
    filter(scores!=0, features!="intercept")
  bio_npdr_lurf_scores
  npdr_lurf.list[[i]]<-bio_npdr_lurf_scores
  fileName <- paste("./replicates_data/bio_npdr_lurf_scores","_",i,".csv",sep="")
  write.table(bio_npdr_lurf_scores,fileName,row.names=F,quote=F,sep=",")
}

# 1
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  3.0909e-05**
# lambda.1se:  0.06356473 
npdr_lurf.list[[1]]
#            features       scores
# 1  avg_rR45CO244CO2 4.612671e+02*
# 2        sd_d18O13C 5.270236e+01*
# 3  walker_propcross 3.428082e+01
# 4        diff2_acf1 1.568039e+01*
# 5  avg_rR46CO244CO2 1.156149e+01
# 6 fluctanal_prop_r1 4.368116e+00*
# 7     time_kl_shift 8.181101e-05*

# 2
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  8.919148e-05**
# lambda.1se:  0.005346889
npdr_lurf.list[[2]]
#             features       scores
# 1         sd_R13C12C 1.457969e+05
# 2    avg_R45CO244CO2 5.733438e+03
# 3   avg_rR45CO244CO2 2.491336e+02*
# 4         sd_d18O13C 6.389030e+01*
# 5    motiftwo_entro3 9.327808e+00
# 6  fluctanal_prop_r1 5.926871e+00*
# 7         diff2_acf1 4.376624e+00*
# 8         sd_d13C12C 1.044282e-02
# 9      time_kl_shift 7.475175e-05*
# 10  avg_rd45CO244CO2 5.796818e-05
# 11   avg_d45CO244CO2 8.786461e-08


# 3
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  0.0001316829** 
# lambda.1se:  0.004116034
npdr_lurf.list[[3]]
#           features       scores
# 1  avg_rR45CO244CO2 3.065813e+02*
# 2        sd_d18O13C 5.568905e+01*
# 3        diff2_acf1 1.712281e+01*
# 4 fluctanal_prop_r1 3.894489e+00*
# 5     time_kl_shift 8.711181e-05*

# 4
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  0.02226695**
# lambda.1se:  0.05143954
npdr_lurf.list[[4]]
#           features     scores
# 1  avg_R45CO244CO2 6137.98133
# 2 avg_rR45CO244CO2  272.17877
# 3       diff2_acf1   10.26696

# 5
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  0.00160616**
# lambda.1se:  0.004072201 
npdr_lurf.list[[5]]
#            features       scores
# 1  avg_rR45CO244CO2 2.167371e+02*
# 2        sd_d18O13C 1.245021e+02*
# 3        diff2_acf1 9.006172e+00*
# 4 fluctanal_prop_r1 6.687737e+00*
# 5  avg_rR46CO244CO2 6.603784e+00
# 6     time_kl_shift 4.738902e-05*



###############################################
# 11. Feature Selection Replicates: NPDR-URF
##############################################

npdr_urf.list<-list()
for(i in seq(1,length(prox.list))){
  bio_npdr_urf <- npdr::npdr("biotic", train.list[[i]][-rm.ind],
                                 regression.type="binomial",
                                 attr.diff.type="numeric-abs",
                                 nbd.method="relieff",
                                 nbd.metric = "precomputed",
                                 external.dist=prox.list[[i]],
                                 knn=knnSURF.balanced(train.list[[i]]$biotic, 
                                                      sd.frac = .5),
                                 use.glmnet = F, glmnet.alpha = 1, 
                                 glmnet.lower = 0, glmnet.lam="lambda.min",
                                 neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                                 padj.method="bonferroni", verbose=T)
  bio_npdr_urf_scores<-bio_npdr_urf %>% filter(pval.adj<.05) %>% pull(att)
  bio_npdr_urf_scores
  npdr_urf.list[[i]]<-bio_npdr_urf_scores
  fileName <- paste("./replicates_data/bio_npdr_urf_scores","_",i,".csv",sep="")
  write.table(bio_npdr_urf_scores,fileName,row.names=F,quote=F,sep=",")
  
}

# 1
npdr_urf.list[[1]]
# [1] "avg_rR45CO244CO2"  "fluctanal_prop_r1" "diff2_acf1"        "avg_d13C12C"      
# [5] "avg_R13C12C"       "avg_R45CO244CO2"   "avg_d45CO244CO2"   "avg_rd45CO244CO2" 
# [9] "time_kl_shift"     "trend"             "avg_rR46CO244CO2"  "sd_d18O13C"       
# [13] "sd_d13C12C"        "sd_R13C12C" 

# 2
npdr_urf.list[[2]]
# [1] "avg_rR45CO244CO2"  "fluctanal_prop_r1" "avg_d45CO244CO2"   "avg_rd45CO244CO2" 
# [5] "avg_R45CO244CO2"   "avg_R13C12C"       "avg_d13C12C"       "sd_d18O13C"       
# [9] "sd_d13C12C"        "sd_R13C12C"        "sd_R45CO244CO2"    "sd_rd45CO244CO2"  
# [13] "sd_d45CO244CO2"    "sd_rR45CO244CO2"   "motiftwo_entro3"   "time_kl_shift"    
# [17] "diff2_acf1"        "avg_rR46CO244CO2"

# 3
npdr_urf.list[[3]]
# [1] "avg_rR45CO244CO2"  "fluctanal_prop_r1" "diff2_acf1"        "avg_R13C12C"      
# [5] "avg_d13C12C"       "avg_R45CO244CO2"   "avg_d45CO244CO2"   "avg_rd45CO244CO2" 
# [9] "trend"             "sd_d18O13C"        "time_kl_shift" 

# 4
npdr_urf.list[[4]]
# [1] "avg_rR45CO244CO2"  "avg_d13C12C"       "avg_R13C12C"       "avg_R45CO244CO2"  
# [5] "avg_d45CO244CO2"   "avg_rd45CO244CO2"  "diff2_acf1"        "sd_d18O13C"       
# [9] "trend"             "fluctanal_prop_r1" "sd_d13C12C"        "sd_R13C12C"       
# [13] "sd_R45CO244CO2"    "sd_rd45CO244CO2"   "sd_d45CO244CO2"    "sd_rR45CO244CO2"

# 5
npdr_urf.list[[5]]
# [1] "fluctanal_prop_r1" "avg_rR45CO244CO2"  "sd_d18O13C"        "diff2_acf1"       
# [5] "avg_R45CO244CO2"   "avg_d45CO244CO2"   "avg_rd45CO244CO2"  "avg_R13C12C"      
# [9] "avg_d13C12C"       "avg_rR46CO244CO2"  "trend"             "motiftwo_entro3"  
# [13] "time_kl_shift"     "sd_d13C12C"        "sd_R13C12C"        "sd_R45CO244CO2"   
# [17] "sd_rd45CO244CO2"   "sd_d45CO244CO2"    "sd_rR45CO244CO2"  


################################################
# 12. Feature Selection Replicates: NPDR-LMan
###############################################

npdr_lman.list<-list()
for(i in seq(1,length(train.list))){
  #head(train.list[[i]])
  bio_npdr_lman <- npdr::npdr("biotic", train.list[[i]][,-rm.ind],
                              regression.type="binomial",
                              attr.diff.type="numeric-abs",
                              nbd.method="relieff",
                              nbd.metric = "manhattan",
                              knn=knnSURF.balanced(train.list[[i]]$biotic, 
                                                   sd.frac = .5),
                              use.glmnet = T, glmnet.alpha = 1, 
                              glmnet.lower = 0, glmnet.lam="lambda.min",
                              neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                              padj.method="bonferroni", verbose=T)
  bio_npdr_lman_scores<-bio_npdr_lman %>% tibble::rownames_to_column(var = "features") %>%
    filter(scores!=0, features!="intercept")
  npdr_lman.list[[i]]<-bio_npdr_lman_scores
  fileName <- paste("./replicates_data/bio_npdr_lman_scores","_",i,".csv",sep="")
  write.table(bio_npdr_lman_scores,fileName,row.names=F,quote=F,sep=",")
}



# 1
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  0.02072249** 
# lambda.1se:  0.03299606 
npdr_lman.list[[1]]
#  features       scores
# 1        sd_R13C12C 5.957040e+04
# 2   avg_R45CO244CO2 2.666511e+04
# 3  avg_rR45CO244CO2 1.440844e+02
# 4   motiftwo_entro3 5.598474e+01
# 5  avg_rR46CO244CO2 3.988942e+01
# 6 fluctanal_prop_r1 1.074871e+01
# 7  walker_propcross 9.933651e-01

# 2
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  0.04420688** 
# lambda.1se:  0.05843889 
npdr_lman.list[[2]]
# features       scores
# 1   avg_R45CO244CO2 1.946942e+04
# 2       avg_R13C12C 1.270833e+03
# 3   motiftwo_entro3 6.054005e+01
# 4 fluctanal_prop_r1 7.886200e+00
# 5       avg_d13C12C 1.826051e-05

# 3
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  0.001041662**
# lambda.1se:  0.01546836 
npdr_lman.list[[3]]
#  features       scores
# 1       avg_R45CO244CO2 3.078229e+04
# 2            sd_R13C12C 1.130519e+04
# 3       motiftwo_entro3 1.101649e+02
# 4            sd_d18O13C 7.353383e+01
# 5      avg_rR45CO244CO2 7.153002e+01
# 6               sampenc 1.252849e+01
# 7     fluctanal_prop_r1 1.085077e+01
# 8           avg_d18O13C 5.797350e-01
# 9  outlierinclude_mdrmd 2.675939e-01
# 10        time_kl_shift 7.476230e-05
# 11         sampen_first 1.149955e-12

# 4
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  0.02901612**
# lambda.1se:  0.04209743 
npdr_lman.list[[4]]
#    features       scores
# 1   avg_R45CO244CO2 2.679006e+04
# 2   motiftwo_entro3 5.785690e+01
# 3  avg_rR46CO244CO2 4.113811e+01
# 4  avg_rR45CO244CO2 3.278696e+01
# 5 fluctanal_prop_r1 6.415680e+00
# 6        sd_d18O13C 3.275470e+00
# 7  avg_rd45CO244CO2 8.777969e-06
# 8   avg_d45CO244CO2 3.145453e-08

# 5
# Theoretical multiSURF average neighbors: 42.
# Theoretical best average neighbors for class imbalance: 30.
# lambda.min:  0.002523401** 
# lambda.1se:  0.07186735
npdr_lman.list[[5]]
#  features       scores
# 1   avg_R45CO244CO2 1.607325e+04
# 2        sd_d18O13C 1.365545e+02
# 3  avg_rR45CO244CO2 9.742193e+01
# 4   motiftwo_entro3 6.486125e+01
# 5  avg_rR46CO244CO2 6.343493e+01
# 6 fluctanal_prop_r1 1.451252e+01
# 7     time_kl_shift 9.997869e-06


###########################################
# 13. set up hyperparmeters
###########################################

####### previously tuned hyperparamters
#### see run0_biosig.R 

# fullVar hyperparameters
fullVar_mtry <- 15
fullVar_splitrule <- "hellinger"
fullVar_minnode <- 7
fullVar_ntrees <- 5000

# npdr-lurf hyperparameters
lurf_mtry <- 2
lurf_splitrule <- "hellinger"
lurf_minnode <- 18
lurf_ntrees <- 5000


# npdr-urf hyperparameters 
urf_mtry <- 3
urf_splitrule <- "extratrees"
urf_minnode <- 12
urf_ntrees <- 5000


#########################################
# 14. NPDR-Lman hyperparameters: need to tune
##########################################
bio_lman_npdr_ind <- which(colnames(train1.dat) %in% c("Analysis",man_feat,"biotic"))
bio_lman_train.dat <- train1.dat[,bio_lman_npdr_ind]
head(bio_lman_train.dat)
bio_lman_test.dat <- test1.dat[,bio_lman_npdr_ind]

####### tune RF, same procedure as before
rm.ind <- which(colnames(bio_lman_train.dat) =="Analysis")

# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(bio_lman_train.dat)[2]-2),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(1,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, savePredictions = T,
                        search="random",
                        allowParallel=T)
start<-Sys.time()
rf1<-train(biotic~., data=bio_lman_train.dat[,-rm.ind], 
           method="ranger",
           importance="none",
           metric="Accuracy", 
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(bio_lman_train.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 6.343647 mins

rf1$bestTune
#  mtry splitrule min.node.size
#    2 extratrees             1
# best accuracy
rf1$finalModel$prediction.error
# [1] 0.1428571
1-rf1$finalModel$prediction.error
# [1] 0.8571429
# saved tuned parameters
bestLman_mtry<-rf1$bestTune$mtry
bestLman_minNodeSize<-rf1$bestTune$min.node.size
bestLman_splitRule<-rf1$bestTune$splitrule


# tune the number of trees using best params from previous run
fileName<-"maxtrees_out1.csv" # for parallel output

tuneGrid<-expand.grid(.mtry=bestLman_mtry,
                      .splitrule=bestLman_splitRule,
                      .min.node.size=bestLman_minNodeSize)

trControl<-trainControl(method = "cv",number=5,
                        verboseIter = T,savePredictions = T,
                        search="random",
                        allowParallel=F # parallelize across number of trees
)


start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=bio_lman_train.dat[,-rm.ind],
            method="ranger",
            importance="none",
            metric="Accuracy", 
            trControl=trControl, 
            tuneGrid=tuneGrid,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(bio_lman_train.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 43.19692 secs

tree.results<-read.csv("maxtrees_out1.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
maxLman_acc<-tree.results[which.max(tree.results$Accuracy),]
maxLman_acc 
#   maxtrees  Accuracy
#1     3000 0.8642127
bestLman_maxTree<-5000#maxFull_acc$maxtrees



### run a final model
bio_lman_train.dat$biotic <- as.factor(bio_lman_train.dat$biotic)
rfLmanFinal.fit <- ranger(biotic~., bio_lman_train.dat[,-rm.ind], 
                               keep.inbag = TRUE,
                               num.trees=bestLman_maxTree, 
                               mtry=bestLman_mtry, 
                               importance="permutation", 
                               splitrule = bestLman_splitRule,
                               min.node.size=bestLman_minNodeSize,
                               class.weights = as.numeric(c(1/table(bio_lman_train.dat$biotic))),
                               scale.permutation.importance = T,
                               local.importance = T, num.threads=4)
sorted.imp<-sort(rfLmanFinal.fit$variable.importance,decreasing=T)
rf_lman.feat<-sorted.imp
names(rf_lman.feat)
# [1] "fluctanal_prop_r1" "motiftwo_entro3"   "avg_rR45CO244CO2"  "time_kl_shift"    
# [5] "avg_R45CO244CO2"   "avg_rR46CO244CO2"  "sd_d18O13C"        "walker_propcross"
rfLmanFinal.fit$confusion.matrix
#         predicted
# true    abiotic biotic
# abiotic      83      6
# biotic       12     39    

table(bio_lman_train.dat$biotic)
# abiotic  biotic 
#      89      51 
rfLmanFinal.fit$prediction.error
# [1] 0.1285714
1-rfLmanFinal.fit$prediction.error
# [1] 0.8714286

predLmanFinal.test<-predict(rfLmanFinal.fit,data=bio_lman_test.dat[,-rm.ind])
bio_lman_test.dat$biotic <- as.factor(bio_lman_test.dat$biotic)
confusionMatrix(predLmanFinal.test$predictions,bio_lman_test.dat$biotic)
# Accuracy : Accuracy : 0.9118 
table(bio_lman_test.dat$biotic, predLmanFinal.test$predictions)
#         predicted 
# true  abiotic biotic
# abiotic      22      0
# biotic        3      9     


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfLmanFinal.fit, "./replicates_data/bioAbio_LmanVarRF_bestFit_0.3p.rds")

# save predictions
lmanTrain.dat<-bio_lman_train.dat
lmanTrain.dat$pred<-rfLmanFinal.fit$predictions
lmanTest.dat<-bio_lman_test.dat
lmanTest.dat$pred<-predLmanFinal.test$predictions
write.table(lmanTrain.dat,"./replicates_data/bioAbio_LmanVarRF_train_0.3p.csv",quote=F,row.names=F,sep=",")
write.table(lmanTest.dat,"./replicates_data/bioAbio_LmanVarRF_test_0.3p.csv",quote=F,row.names=F,sep=",")


######################################
# 15. save NPDR-Lman hyperparameters
###################################
lman_mtry <- 2
lman_splitrule <- "extratrees"
lman_minnode <- 1
lman_ntrees <- 5000

####################################
# 16. setup RF hyperparameter dataframe
###################################
hyperparams.df <- as.data.frame(matrix(rep(NA,5*4),ncol=5))
colnames(hyperparams.df) <- c("var_space","mtry","splitrule","min_nodesize","ntrees")
hyperparams.df$var_space <- c("full_var","lurf_var","urf_var","lman_var")
hyperparams.df$mtry <- c(fullVar_mtry,lurf_mtry,urf_mtry,lman_mtry)
hyperparams.df$splitrule <- c(fullVar_splitrule,lurf_splitrule,urf_splitrule,lman_splitrule)
hyperparams.df$min_nodesize <- c(fullVar_minnode,lurf_minnode,urf_minnode,lman_minnode)
hyperparams.df$ntrees <- c(fullVar_ntrees,lurf_ntrees,urf_ntrees,lman_ntrees)
hyperparams.df
#   var_space mtry  splitrule min_nodesize ntrees
# 1  full_var   15  hellinger            7   5000
# 2  lurf_var    2  hellinger           18   5000
# 3   urf_var    3 extratrees           12   5000
# 4  lman_var    2 extratrees            1   5000


###########################################################################
# 17. RF replicates: full variable RF, NPDR-LURF RF, NPDR-URF RF, NPDR-LMan RF
###########################################################################
# need to grab correct col names for each RF variable space
# full var RF: train.list/test.list
# lurf var RF: lurf_train.list/lurf_test.list
# urf var RF: urf_train.list/urf_test.list
# lman var RF: lman_train.list/lman_test.list

# hyperparams.df
# train.list
test.list <- list(test1.dat,test2.dat,test3.dat,test4.dat,test5.dat)

# store results
fullVar_rf.list <- list()
fullVar_train_cm.list <- list()
fullVar_test_cm.list <- list()
fullVar_train.list <- list()
fullVar_test.list <- list()

lurf_rf.list <- list()
lurf_train_cm.list <- list()
lurf_test_cm.list <- list()
lurf_train.list <- list()
lurf_test.list <- list()

urf_rf.list <- list()
urf_train_cm.list <- list()
urf_test_cm.list <- list()
urf_train.list <- list()
urf_test.list <- list()

lman_rf.list <- list()
lman_train_cm.list <- list()
lman_test_cm.list <- list()
lman_train.list <- list()
lman_test.list <- list()

# RF for 4 variable spaces 5x (for each replicate) using tuned hyperparams
for(i in seq(1,length(train.list))){
  rm.ind <- 1 # rm Analysis num for RF
  
  ###########
  #### train and test sets
  # full variable space
  full_train.dat <- train.list[[i]] 
  #head(colnames(full_train.dat))
  full_test.dat <- test.list[[i]]
  
  # npdr-lurf space
  lurf_feat <- npdr_lurf.list[[i]]$features
  lurf_ind <- which(colnames(full_train.dat) %in% c("Analysis",lurf_feat,"biotic"))
  lurf_train.dat <- full_train.dat[,lurf_ind]
  lurf_test.dat <- full_test.dat[,lurf_ind]
  #colnames(lurf_test.dat)
  
  # npdr-urf space
  urf_feat <- npdr_urf.list[[i]]
  urf_ind <- which(colnames(full_train.dat) %in% c("Analysis",urf_feat,"biotic"))
  urf_train.dat <- full_train.dat[,urf_ind]
  urf_test.dat <- full_test.dat[,urf_ind]
  #colnames(urf_test.dat)
  
  # npdr-lman space
  lman_feat <- npdr_lman.list[[i]]$features
  lman_ind <- which(colnames(full_train.dat) %in% c("Analysis",lman_feat,"biotic"))
  lman_train.dat <- full_train.dat[,lman_ind]
  lman_test.dat <- full_test.dat[,lman_ind]
  #colnames(lman_test.dat)
  ##########
  
  ######## 
  # RF
  ########
  # full var RF
  full_train.dat$biotic <- as.factor(full_train.dat$biotic)
  full_test.dat$biotic <- as.factor(full_test.dat$biotic)
  
  fullVar.rf <- ranger(biotic~., full_train.dat[,-rm.ind], 
                                 keep.inbag = TRUE,
                                 num.trees=hyperparams.df[1,]$ntrees, 
                                 mtry=hyperparams.df[1,]$mtry, 
                                 importance="permutation", 
                                 splitrule = hyperparams.df[1,]$splitrule,
                                 min.node.size=hyperparams.df[1,]$min_nodesize,
                                 class.weights = as.numeric(c(1/table(full_train.dat$biotic))),
                                 scale.permutation.importance = T,
                                 local.importance = T, num.threads=4)
  fullVar_train.cm <- fullVar.rf$confusion.matrix
  fullVar_train.pred <- fullVar.rf$predictions
  full_train.dat$pred <- fullVar_train.pred
  fullVar_test.pred<-predict(fullVar.rf,data=full_test.dat[,-rm.ind])
  full_test.dat$pred <- fullVar_test.pred$predictions
  fullVar_test.cm <- table(full_test.dat$biotic, fullVar_test.pred$predictions)
  
  fullVar_rf.list[[i]] <- fullVar.rf
  fullVar_train.list[[i]] <- full_train.dat
  fullVar_train_cm.list[[i]] <- fullVar_train.cm
  fullVar_test_cm.list[[i]] <- fullVar_test.cm
  fullVar_test.list[[i]] <- full_test.dat
  
  
  
  # npdr-lurf RF
  lurf_train.dat$biotic <- as.factor(lurf_train.dat$biotic)
  lurf_test.dat$biotic <- as.factor(lurf_test.dat$biotic)
  
  lurf.rf <- ranger(biotic~., lurf_train.dat[,-rm.ind], 
                    keep.inbag = TRUE,
                    num.trees=hyperparams.df[2,]$ntrees, 
                    mtry=hyperparams.df[2,]$mtry, 
                    importance="permutation", 
                    splitrule = hyperparams.df[2,]$splitrule,
                    min.node.size=hyperparams.df[2,]$min_nodesize,
                    class.weights = as.numeric(c(1/table(lurf_train.dat$biotic))),
                    scale.permutation.importance = T,
                    local.importance = T, num.threads=4)
  lurf_train.cm <- lurf.rf$confusion.matrix
  lurf_train.pred <- lurf.rf$predictions
  lurf_train.dat$pred <- lurf_train.pred
  lurf_test.pred<-predict(lurf.rf,data=lurf_test.dat[,-rm.ind])
  lurf_test.dat$pred <- lurf_test.pred$predictions
  lurf_test.cm <- table(lurf_test.dat$biotic, lurf_test.pred$predictions)
  
  lurf_rf.list[[i]] <- lurf.rf
  lurf_train.list[[i]] <- lurf_train.dat
  lurf_train_cm.list[[i]] <- lurf_train.cm
  lurf_test_cm.list[[i]] <- lurf_test.cm
  lurf_test.list[[i]] <- lurf_test.dat
  
  
  # npdr-urf RF
  urf_train.dat$biotic <- as.factor(urf_train.dat$biotic)
  urf_test.dat$biotic <- as.factor(urf_test.dat$biotic)
  
  urf.rf <- ranger(biotic~., urf_train.dat[,-rm.ind], 
                    keep.inbag = TRUE,
                    num.trees=hyperparams.df[3,]$ntrees, 
                    mtry=hyperparams.df[3,]$mtry, 
                    importance="permutation", 
                    splitrule = hyperparams.df[3,]$splitrule,
                    min.node.size=hyperparams.df[3,]$min_nodesize,
                    class.weights = as.numeric(c(1/table(urf_train.dat$biotic))),
                    scale.permutation.importance = T,
                    local.importance = T, num.threads=4)
  urf_train.cm <- urf.rf$confusion.matrix
  urf_train.pred <- urf.rf$predictions
  urf_train.dat$pred <- urf_train.pred
  urf_test.pred<-predict(urf.rf,data=urf_test.dat[,-rm.ind])
  urf_test.dat$pred <- urf_test.pred$predictions
  urf_test.cm <- table(urf_test.dat$biotic, urf_test.pred$predictions)
  
  urf_rf.list[[i]] <- urf.rf
  urf_train.list[[i]] <- urf_train.dat
  urf_train_cm.list[[i]] <- urf_train.cm
  urf_test_cm.list[[i]] <- urf_test.cm
  urf_test.list[[i]] <- urf_test.dat
  
  
  
  # npdr-lman RF
  lman_train.dat$biotic <- as.factor(lman_train.dat$biotic)
  lman_test.dat$biotic <- as.factor(lman_test.dat$biotic)
  
  lman.rf <- ranger(biotic~., lman_train.dat[,-rm.ind], 
                   keep.inbag = TRUE,
                   num.trees=hyperparams.df[4,]$ntrees, 
                   mtry=hyperparams.df[4,]$mtry, 
                   importance="permutation", 
                   splitrule = hyperparams.df[4,]$splitrule,
                   min.node.size=hyperparams.df[4,]$min_nodesize,
                   class.weights = as.numeric(c(1/table(lman_train.dat$biotic))),
                   scale.permutation.importance = T,
                   local.importance = T, num.threads=4)
  lman_train.cm <- lman.rf$confusion.matrix
  lman_train.pred <- lman.rf$predictions
  lman_train.dat$pred <- lman_train.pred
  lman_test.pred<-predict(lman.rf,data=lman_test.dat[,-rm.ind])
  lman_test.dat$pred <- lman_test.pred$predictions
  lman_test.cm <- table(lman_test.dat$biotic, lman_test.pred$predictions)
  
  lman_rf.list[[i]] <- lman.rf
  lman_train.list[[i]] <- lman_train.dat
  lman_train_cm.list[[i]] <- lman_train.cm
  lman_test_cm.list[[i]] <- lman_test.cm
  lman_test.list[[i]] <- lman_test.dat
  
  
}
################################
# 18. inspect RF replicate results
##############################


##### training and testing cms - train/test set 1

### full Var
fullVar_train_cm.list[[1]]
# predicted
# true     abiotic biotic
# abiotic      79     10
# biotic        7     44

fullVar_test_cm.list[[1]]
#       abiotic biotic
# abiotic      20      2
# biotic        0     12

# save results to file
saveRDS(fullVar_rf.list[[1]], "./replicates_data/bioAbio_fullVarRF_set1_rf.rds")
write.table(fullVar_train.list[[1]],"./replicates_data/bioAbio_fullVarRF_set1_train.csv",quote=F,row.names=F,sep=",")
write.table(fullVar_test.list[[1]],"./replicates_data/bioAbio_fullVarRF_set1_test.csv",quote=F,row.names=F,sep=",")


### lurf 
lurf_train_cm.list[[1]]
# predicted
# true     abiotic biotic
# abiotic      77     12
# biotic        8     43

lurf_test_cm.list[[1]]
#        abiotic biotic
# abiotic      19      3
# biotic        2     10


# save results to file
saveRDS(lurf_rf.list[[1]], "./replicates_datas/bioAbio_lurfVarRF_set1_rf.rds")
write.table(lurf_train.list[[1]],"./replicates_data/bioAbio_lurfVarRF_set1_train.csv",quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[1]],"./replicates_data/bioAbio_lurfVarRF_set1_test.csv",quote=F,row.names=F,sep=",")


### urf
urf_train_cm.list[[1]]
# predicted
# true    abiotic biotic
# abiotic      81      8
# biotic        5     46

urf_test_cm.list[[1]]
#         abiotic biotic
# abiotic      19      3
# biotic        0     12

# save results to file
saveRDS(urf_rf.list[[1]], "./replicates_data/bioAbio_urfVarRF_set1_rf.rds")
write.table(urf_train.list[[1]],"./replicates_data/bioAbio_urfVarRF_set1_train.csv",quote=F,row.names=F,sep=",")
write.table(urf_test.list[[1]],"./replicates_data/bioAbio_urfVarRF_set1_test.csv",quote=F,row.names=F,sep=",")


### lman
lman_train_cm.list[[1]]
# predicted
# trueabiotic biotic
# abiotic      81      8
# biotic       12     39

lman_test_cm.list[[1]]
#         abiotic biotic
# abiotic      20      2
# biotic        0     12

# save results to file
saveRDS(lman_rf.list[[1]], "./replicates_data/bioAbio_lmanVarRF_set1_rf.rds")
write.table(lman_train.list[[1]],"./replicates_data/bioAbio_lmanVarRF_set1_train.csv",quote=F,row.names=F,sep=",")
write.table(lman_test.list[[1]],"./replicates_data/bioAbio_lmanVarRF_set1_test.csv",quote=F,row.names=F,sep=",")




##### training and testing cms - train/test set 2

### full Var
fullVar_train_cm.list[[2]]
# predicted
# true     abiotic biotic
# abiotic      83      6
# biotic        6     45

fullVar_test_cm.list[[2]]
#       abiotic biotic
# abiotic      21      1
# biotic        2     10

# save results to file
saveRDS(fullVar_rf.list[[2]], "./replicates_data/bioAbio_fullVarRF_set2_rf.rds")
write.table(fullVar_train.list[[2]],"./replicates_data/bioAbio_fullVarRF_set2_train.csv",quote=F,row.names=F,sep=",")
write.table(fullVar_test.list[[2]],"./replicates_data/bioAbio_fullVarRF_set2_test.csv",quote=F,row.names=F,sep=",")



### lurf 
lurf_train_cm.list[[2]]
# predicted
# true     abiotic biotic
# abiotic      82      7
# biotic        7     44

lurf_test_cm.list[[2]]
#        abiotic biotic
# abiotic      19      3
# biotic        3      9

# save results to file
saveRDS(lurf_rf.list[[2]], "./replicates_data/bioAbio_lurfVarRF_set2_rf.rds")
write.table(lurf_train.list[[2]],"./replicates_data/bioAbio_lurfVarRF_set2_train.csv",quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[2]],"./replicates_data/bioAbio_lurfVarRF_set2_test.csv",quote=F,row.names=F,sep=",")



### urf
urf_train_cm.list[[2]]
# predicted
# true  abiotic biotic
# abiotic      81      8
# biotic        6     45

urf_test_cm.list[[2]]
#      abiotic biotic
# abiotic      19      3
# biotic        4      8

# save results to file
saveRDS(urf_rf.list[[2]], "./replicates_data/bioAbio_urfVarRF_set2_rf.rds")
write.table(urf_train.list[[2]],"./replicates_data/bioAbio_urfVarRF_set2_train.csv",quote=F,row.names=F,sep=",")
write.table(urf_test.list[[2]],"./replicates_data/bioAbio_urfVarRF_set2_test.csv",quote=F,row.names=F,sep=",")



### lman
lman_train_cm.list[[2]]
# predicted
# true    abiotic biotic
# abiotic      80      9
# biotic       14     37

lman_test_cm.list[[2]]
#    abiotic biotic
# abiotic      19      3
# biotic        3      9


# save results to file
saveRDS(lman_rf.list[[2]], "./replicates_data/bioAbio_lmanVarRF_set2_rf.rds")
write.table(lman_train.list[[2]],"./replicates_data/bioAbio_lmanVarRF_set2_train.csv",quote=F,row.names=F,sep=",")
write.table(lman_test.list[[2]],"./replicates_data/bioAbio_lmanVarRF_set2_test.csv",quote=F,row.names=F,sep=",")




##### training and testing cms - train/test set 3

### full Var
fullVar_train_cm.list[[3]]
# predicted
# true   abiotic biotic
# abiotic      83      6
# biotic       10     41

fullVar_test_cm.list[[3]]
#         abiotic biotic
# abiotic      21      1
# biotic        0     12

# save results to file
saveRDS(fullVar_rf.list[[3]], "./replicates_data/bioAbio_fullVarRF_set3_rf.rds")
write.table(fullVar_train.list[[3]],"./replicates_data/bioAbio_fullVarRF_set3_train.csv",quote=F,row.names=F,sep=",")
write.table(fullVar_test.list[[3]],"./replicates_data/bioAbio_fullVarRF_set3_test.csv",quote=F,row.names=F,sep=",")



### lurf 
lurf_train_cm.list[[3]]
# predicted
# true   abiotic biotic
# abiotic      78     11
# biotic        7     44

lurf_test_cm.list[[3]]
#        abiotic biotic
# abiotic      17      5
# biotic        2     10


# save results to file
saveRDS(lurf_rf.list[[3]], "./replicates_data/bioAbio_lurfVarRF_set3_rf.rds")
write.table(lurf_train.list[[3]],"./replicates_data/bioAbio_lurfVarRF_set3_train.csv",quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[3]],"./replicates_data/bioAbio_lurfVarRF_set3_test.csv",quote=F,row.names=F,sep=",")



### urf
urf_train_cm.list[[3]]
# predicted
# true    abiotic biotic
# abiotic      82      7
# biotic        4     47

urf_test_cm.list[[3]]
#        abiotic biotic
# abiotic      20      2
# biotic        0     12

# save results to file
saveRDS(urf_rf.list[[3]], "./replicates_data/bioAbio_urfVarRF_set3_rf.rds")
write.table(urf_train.list[[3]],"./replicates_data/bioAbio_urfVarRF_set3_train.csv",quote=F,row.names=F,sep=",")
write.table(urf_test.list[[3]],"./replicates_data/bioAbio_urfVarRF_set3_test.csv",quote=F,row.names=F,sep=",")



### lman
lman_train_cm.list[[3]]
# predicted
# true   abiotic biotic
# abiotic      83      6
# biotic        9     42

lman_test_cm.list[[3]]
#        abiotic biotic
# abiotic      20      2
# biotic        2     10


# save results to file
saveRDS(lman_rf.list[[3]], "./replicates_data/bioAbio_lmanVarRF_set3_rf.rds")
write.table(lman_train.list[[3]],"./replicates_data/bioAbio_lmanVarRF_set3_train.csv",quote=F,row.names=F,sep=",")
write.table(lman_test.list[[3]],"./replicates_data/bioAbio_lmanVarRF_set3_test.csv",quote=F,row.names=F,sep=",")




##### training and testing cms - train/test set 4

### full Var
fullVar_train_cm.list[[4]]
# predicted
# true   abiotic biotic
  # abiotic      81      8
  # biotic        8     43

fullVar_test_cm.list[[4]]
#         abiotic biotic
# abiotic      22      0
# biotic        1     11

# save results to file
saveRDS(fullVar_rf.list[[4]], "./replicates_data/bioAbio_fullVarRF_set4_rf.rds")
write.table(fullVar_train.list[[4]],"./replicates_data/bioAbio_fullVarRF_set4_train.csv",quote=F,row.names=F,sep=",")
write.table(fullVar_test.list[[4]],"./replicates_data/bioAbio_fullVarRF_set4_test.csv",quote=F,row.names=F,sep=",")



### lurf 
lurf_train_cm.list[[4]]
# predicted
# true      abiotic biotic
# abiotic      75     14
# biotic       12     39

lurf_test_cm.list[[4]]
#         abiotic biotic
# abiotic      20      2
# biotic        2     10


# save results to file
saveRDS(lurf_rf.list[[4]], "./replicates_data/bioAbio_lurfVarRF_set4_rf.rds")
write.table(lurf_train.list[[4]],"./replicates_data/bioAbio_lurfVarRF_set4_train.csv",quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[4]],"./replicates_data/bioAbio_lurfVarRF_set4_test.csv",quote=F,row.names=F,sep=",")




### urf
urf_train_cm.list[[4]]
# predicted
# true    abiotic biotic
# abiotic      83      6
# biotic        6     45

urf_test_cm.list[[4]]
#       abiotic biotic
# abiotic      21      1
# biotic        1     11

# save results to file
saveRDS(urf_rf.list[[4]], "./replicates_data/bioAbio_urfVarRF_set4_rf.rds")
write.table(urf_train.list[[4]],"./replicates_data/bioAbio_urfVarRF_set4_train.csv",quote=F,row.names=F,sep=",")
write.table(urf_test.list[[4]],"./replicates_data/bioAbio_urfVarRF_set4_test.csv",quote=F,row.names=F,sep=",")




### lman
lman_train_cm.list[[4]]
# predicted
# true  abiotic biotic
# abiotic      80      9
# biotic       12     39

lman_test_cm.list[[4]]
#         abiotic biotic
# abiotic      22      0
# biotic        2     10

# save results to file
saveRDS(lman_rf.list[[4]], "./replicates_data/bioAbio_lmanVarRF_set4_rf.rds")
write.table(lman_train.list[[4]],"./replicates_data/bioAbio_lmanVarRF_set4_train.csv",quote=F,row.names=F,sep=",")
write.table(lman_test.list[[4]],"./replicates_data/bioAbio_lmanVarRF_set4_test.csv",quote=F,row.names=F,sep=",")





##### training and testing cms - train/test set 5

### full Var
fullVar_train_cm.list[[5]]
# predicted
# true  abiotic biotic
# abiotic      83      6
# biotic        7     44

fullVar_test_cm.list[[5]]
#            abiotic biotic
# abiotic      22      0
# biotic        2     10

# save results to file
saveRDS(fullVar_rf.list[[5]], "./replicates_data/bioAbio_fullVarRF_set5_rf.rds")
write.table(fullVar_train.list[[5]],"./replicates_data/bioAbio_fullVarRF_set5_train.csv",quote=F,row.names=F,sep=",")
write.table(fullVar_test.list[[5]],"./replicates_data/bioAbio_fullVarRF_set5_test.csv",quote=F,row.names=F,sep=",")



### lurf 
lurf_train_cm.list[[5]]
# predicted
# true   abiotic biotic
# abiotic      77     12
# biotic       10     41

lurf_test_cm.list[[5]]
#          abiotic biotic
# abiotic      22      0
# biotic        3      9


# save results to file
saveRDS(lurf_rf.list[[5]], "./replicates_data/bioAbio_lurfVarRF_set5_rf.rds")
write.table(lurf_train.list[[5]],"./replicates_data/bioAbio_lurfVarRF_set5_train.csv",quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[5]],"./replicates_data/bioAbio_lurfVarRF_set5_test.csv",quote=F,row.names=F,sep=",")



### urf
urf_train_cm.list[[5]]
# predicted
# true    abiotic biotic
# abiotic      81      8
# biotic        6     45

urf_test_cm.list[[5]]
#       abiotic biotic
# abiotic      22      0
# biotic        2     10


# save results to file
saveRDS(urf_rf.list[[5]], "./replicates_data/bioAbio_urfVarRF_set5_rf.rds")
write.table(urf_train.list[[5]],"./replicates_data/bioAbio_urfVarRF_set5_train.csv",quote=F,row.names=F,sep=",")
write.table(urf_test.list[[5]],"./replicates_data/bioAbio_urfVarRF_set5_test.csv",quote=F,row.names=F,sep=",")





### lman
lman_train_cm.list[[5]]
# predicted
# true  abiotic biotic
# abiotic      80      9
# biotic       10     41

lman_test_cm.list[[5]]
#         abiotic biotic
# abiotic      22      0
# biotic        5      7


# save results to file
saveRDS(lman_rf.list[[5]], "./replicates_data/bioAbio_lmanVarRF_set5_rf.rds")
write.table(lman_train.list[[5]],"./replicates_data/bioAbio_lmanVarRF_set5_train.csv",quote=F,row.names=F,sep=",")
write.table(lman_test.list[[5]],"./replicates_data/bioAbio_lmanVarRF_set5_test.csv",quote=F,row.names=F,sep=",")



##################################################################################
# 19. Explore "reduced redundancy" variable space: find highly correlated variables
##################################################################################
# remove highly correlated variables (>0.99) and compare
head(colnames(tsms.dat))
#[1] "Analysis"    "x_acf1"      "x_acf10"     "diff1_acf1"  "diff1_acf10" "diff2_acf1" 
tail(colnames(tsms.dat))
# [1] "avg_d18O13C"       "sd_d18O13C"        "avg_calib_d18O16O" "sd_calib_d18O16O" 
# [5] "biotic"            "dataset"
rm.ind <- which(colnames(tsms.dat) %in% c("Analysis","biotic","dataset"))
cor.dat <- cor(tsms.dat[,-rm.ind])
diag(cor.dat)<-0
head(cor.dat)[,1:6]
#                  x_acf1     x_acf10  diff1_acf1 diff1_acf10 diff2_acf1 diff2_acf10
# x_acf1       0.00000000  0.94932011  0.09791779  0.37213554 -0.7306787  -0.9380885
# x_acf10      0.94932011  0.00000000 -0.21851013  0.06805399 -0.9030994  -0.9723683
# diff1_acf1   0.09791779 -0.21851013  0.00000000  0.95022472  0.5821875   0.1530902
# diff1_acf10  0.37213554  0.06805399  0.95022472  0.00000000  0.3172077  -0.1400930
# diff2_acf1  -0.73067867 -0.90309938  0.58218748  0.31720772  0.0000000   0.8918093
# diff2_acf10 -0.93808855 -0.97236825  0.15309022 -0.14009299  0.8918093   0.0000000


#########################################
# 20. remove highly correlated variables
#########################################
## the following is a lot of work but worth investigating

cor_vec.list <- list()
cor_vec_ind <- 1
for(i in seq(1,dim(cor.dat)[2])){
  curr_col <- colnames(cor.dat)[i]
  curr_col
  curr_cor_vec <- cor.dat[,i]
  head(curr_cor_vec)
  high_cor_ind <- which(abs(curr_cor_vec)>=0.99) # find vars with > 99% correlation
  row_names <- names(curr_cor_vec[high_cor_ind])
  high_cor_vec <- curr_cor_vec[high_cor_ind]
  # add to list of highly correlated variables
  if(length(row_names)>0){
    for(j in seq(1,length(row_names))){
      cor_num <- high_cor_vec[j]
      cor_vec <- c(row_names[j],curr_col,cor_num)
      cor_vec.list[[cor_vec_ind]] <- cor_vec
      cor_vec_ind <- cor_vec_ind+1
    }
  }
  
}

length(cor_vec.list)
# 448
cor_vec.list[[1]]
#  e_acf10 
# "e_acf10"            "x_acf1" "0.990212869670723" 
cor_vec.list[[2]]
# std1st_der 
# "std1st_der"             "x_acf1" "-0.998493239063819" 

cor_vec.df <- as.data.frame(do.call(rbind,cor_vec.list))
dim(cor_vec.df)
# [1] 448   3
colnames(cor_vec.df) <- c("var1","var2","cor")
cor_vec.df$cor <- as.numeric(cor_vec.df$cor)
head(cor_vec.df)
#         var1    var2        cor
# 1    e_acf10  x_acf1  0.9902129
# 2 std1st_der  x_acf1 -0.9984932
# 3    entropy x_acf10 -0.9910630
# 4       ac_9 x_acf10  0.9959292
# 5    x_acf10 entropy -0.9910630
# 6       ac_9 entropy -0.9938500

length(unique(cor_vec.df$var2))
# [1] 64

# put all vars associated with a unique var 2 in a single vector 
cor_var.list <- list()
for(i in seq(1,length(unique(cor_vec.df$var2)))){
  cor_vec <- c()
  curr_var <- unique(cor_vec.df$var2)[i]
  var2.ind <- which(cor_vec.df$var2==curr_var)
  var1.vec <- cor_vec.df[var2.ind,]
  var1.vec
  cor_vec <- c(curr_var,var1.vec$var1)
  cor_vec
  cor_var.list[[i]] <- cor_vec
}
length(cor_var.list)


keep.vec <- c()
rm.vec <- c()

## 1.
cor_var.list[[1]]
# [1] "x_acf1"     "e_acf10"    "std1st_der"
cor_var.list[[8]]
# [1] "e_acf10"    "x_acf1"     "std1st_der"
cor_var.list[[13]]
#[1] "std1st_der" "x_acf1"     "e_acf10"

# keep "std1st_der"
keep.vec <- c("std1st_der")
rm.vec <- c("x_acf1", "e_acf10")

## 2.
cor_var.list[[2]]
# [1] "x_acf10" "entropy" "ac_9" 
cor_var.list[[3]]
# [1] "entropy" "x_acf10" "ac_9"

# keep "entropy"
keep.vec <- c(keep.vec,"entropy")
rm.vec <- c(rm.vec,"x_acf10","ac_9")

## 3.
cor_var.list[[4]]
# [1] "time_level_shift" "time_var_shift" 
cor_var.list[[5]]
# [1] "time_var_shift"   "time_level_shift"

# keep "time_level_shift"
keep.vec <- c(keep.vec,"time_level_shift")
rm.vec <- c(rm.vec, "time_var_shift")

## 4.
cor_var.list[[6]]
# [1] "diff1x_pacf5" "diff2x_pacf5"
cor_var.list[[7]]
# [1] "diff2x_pacf5" "diff1x_pacf5"

# keep "diff2x_pacf5"
keep.vec <- c(keep.vec,"diff2x_pacf5")
rm.vec <- c(rm.vec,"diff1x_pacf5")

## 5.
cor_var.list[[9]]
# [1] "unitroot_pp" "ac_9"  

rm.vec
# [1] "x_acf1"         "e_acf10"        "x_acf10"        "ac_9"           "time_var_shift"
# [6] "diff1x_pacf5"

# "ac_9" already in rm.vec - rm both
rm.vec <- c(rm.vec, "unitroot_pp")
rm.vec
# [1] "x_acf1"         "e_acf10"        "x_acf10"        "ac_9"           "time_var_shift"
# [6] "diff1x_pacf5"   "unitroot_pp" 
keep.vec
# [1] "std1st_der"       "entropy"          "time_level_shift" "diff2x_pacf5"    


## 6.
cor_var.list[[10]] # all accounted for in keep.vec or rm.vec
#  [1] "ac_9"        "x_acf10"*     "entropy"-keep     "unitroot_pp"*


## 7.
cor_var.list[[11]]
# [1] "sampenc"      "sampen_first"
cor_var.list[[12]]
# [1] "sampen_first" "sampenc" 

# keep "sampenc"
keep.vec <- c(keep.vec,"sampenc")
rm.vec <- c(rm.vec,"sampen_first")

## 8. 
cor_var.list[[14]]
# [1] "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"        "avg_rIntensity44" 
# [5] "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll" "avg_Intensity44"  
# [9] "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea"

cor_var.list[[16]]
# [1] "avg_Ampl45"        "avg_Ampl44"        "avg_Ampl46"        "avg_rIntensity44" 
# [5] "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll" "avg_Intensity44"  
# [9] "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea"

cor_var.list[[18]]
# [1] "avg_Ampl46"        "avg_Ampl44"        "avg_Ampl45"        "avg_rIntensity44" 
# [5] "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll" "avg_Intensity44"  
# [9] "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea" 

cor_var.list[[20]]
# [1] "avg_rIntensity44"  "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll" "avg_Intensity44"  
# [9] "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea"

cor_var.list[[22]]
# [1] "avg_rIntensity45"  "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity44"  "avg_rIntensity46"  "avg_rIntensityAll" "avg_Intensity44"  
# [9] "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea"

cor_var.list[[24]]
# [1] "avg_rIntensity46"  "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity44"  "avg_rIntensity45"  "avg_rIntensityAll" "avg_Intensity44"  
# [9] "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea"

cor_var.list[[26]]
# [1] "avg_rIntensityAll" "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity44"  "avg_rIntensity45"  "avg_rIntensity46"  "avg_Intensity44"  
# [9] "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea"

cor_var.list[[28]]
# [1] "avg_Intensity44"   "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity44"  "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll"
# [9] "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea"

cor_var.list[[30]]
# [1] "avg_Intensity45"   "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity44"  "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll"
# [9] "avg_Intensity44"   "avg_Intensity46"   "avg_IntensityAll"  "avg_pkArea" 

cor_var.list[[32]]
# [1] "avg_Intensity46"   "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity44"  "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll"
# [9] "avg_Intensity44"   "avg_Intensity45"   "avg_IntensityAll"  "avg_pkArea" 

cor_var.list[[34]]
# [1] "avg_IntensityAll"  "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity44"  "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll"
# [9] "avg_Intensity44"   "avg_Intensity45"   "avg_Intensity46"   "avg_pkArea" 

cor_var.list[[62]]
# [1] "avg_pkArea"        "avg_Ampl44"        "avg_Ampl45"        "avg_Ampl46"       
# [5] "avg_rIntensity44"  "avg_rIntensity45"  "avg_rIntensity46"  "avg_rIntensityAll"
# [9] "avg_Intensity44"   "avg_Intensity45"   "avg_Intensity46"   "avg_IntensityAll"

# keep avg_pkArea
keep.vec <- c(keep.vec,"avg_pkArea")
rm.vec <- c(rm.vec,"avg_Ampl44","avg_Ampl46","avg_rIntensity44",
            "avg_rIntensity45","avg_rIntensity46","avg_rIntensityAll",
            "avg_Intensity44","avg_Intensity45","avg_Intensity46","avg_IntensityAll",
            "avg_Ampl45")



## 9.
cor_var.list[[15]]
# [1] "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"        "sd_rIntensity44" 
# [5] "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll" "sd_Intensity44"  
# [9] "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea"

cor_var.list[[17]]
# [1] "sd_Ampl45"        "sd_Ampl44"        "sd_Ampl46"        "sd_rIntensity44" 
# [5] "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll" "sd_Intensity44"  
# [9] "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea"

cor_var.list[[19]]
# [1] "sd_Ampl46"        "sd_Ampl44"        "sd_Ampl45"        "sd_rIntensity44" 
# [5] "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll" "sd_Intensity44"  
# [9] "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea"

cor_var.list[[21]]
# [1] "sd_rIntensity44"  "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll" "sd_Intensity44"  
# [9] "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea"

cor_var.list[[23]]
# [1] "sd_rIntensity45"  "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity44"  "sd_rIntensity46"  "sd_rIntensityAll" "sd_Intensity44"  
# [9] "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea"

cor_var.list[[25]]
# [1] "sd_rIntensity46"  "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity44"  "sd_rIntensity45"  "sd_rIntensityAll" "sd_Intensity44"  
# [9] "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea" 

cor_var.list[[27]]
# [1] "sd_rIntensityAll" "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity44"  "sd_rIntensity45"  "sd_rIntensity46"  "sd_Intensity44"  
# [9] "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea" 

cor_var.list[[29]]
# [1] "sd_Intensity44"   "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity44"  "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll"
# [9] "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea"

cor_var.list[[31]]
# [1] "sd_Intensity45"   "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity44"  "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll"
# [9] "sd_Intensity44"   "sd_Intensity46"   "sd_IntensityAll"  "sd_pkArea" 

cor_var.list[[33]]
# [1] "sd_Intensity46"   "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity44"  "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll"
# [9] "sd_Intensity44"   "sd_Intensity45"   "sd_IntensityAll"  "sd_pkArea" 

cor_var.list[[35]]
# [1] "sd_IntensityAll"  "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity44"  "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll"
# [9] "sd_Intensity44"   "sd_Intensity45"   "sd_Intensity46"   "sd_pkArea" 

cor_var.list[[63]]
# [1] "sd_pkArea"        "sd_Ampl44"        "sd_Ampl45"        "sd_Ampl46"       
# [5] "sd_rIntensity44"  "sd_rIntensity45"  "sd_rIntensity46"  "sd_rIntensityAll"
# [9] "sd_Intensity44"   "sd_Intensity45"   "sd_Intensity46"   "sd_IntensityAll"



# keep "sd_pkArea"
keep.vec <- c(keep.vec,"sd_pkArea")
rm.vec <- c(rm.vec,"sd_Ampl44","sd_Ampl45","sd_Ampl46","sd_rIntensity44",
            "sd_rIntensity45","sd_rIntensity46","sd_rIntensityAll","sd_Intensity44",
            "sd_Intensity45","sd_Intensity46","sd_IntensityAll")



## 10.
cor_var.list[[36]]
# [1] "sd_rR45CO244CO2" "sd_R45CO244CO2"  "sd_rd45CO244CO2" "sd_d45CO244CO2" 
# [5] "sd_R13C12C"      "sd_d13C12C"   

cor_var.list[[39]]
# [1] "sd_R45CO244CO2"  "sd_rR45CO244CO2" "sd_rd45CO244CO2" "sd_d45CO244CO2" 
# [5] "sd_R13C12C"      "sd_d13C12C"

cor_var.list[[41]]
# [1] "sd_rd45CO244CO2" "sd_rR45CO244CO2" "sd_R45CO244CO2"  "sd_d45CO244CO2" 
# [5] "sd_R13C12C"      "sd_d13C12C"

cor_var.list[[43]]
# [1] "sd_d45CO244CO2"  "sd_rR45CO244CO2" "sd_R45CO244CO2"  "sd_rd45CO244CO2"
# [5] "sd_R13C12C"      "sd_d13C12C" 

cor_var.list[[51]]
# [1] "sd_R13C12C"      "sd_rR45CO244CO2" "sd_R45CO244CO2"  "sd_rd45CO244CO2"
# [5] "sd_d45CO244CO2"  "sd_d13C12C"

cor_var.list[[53]]
# [1] "sd_d13C12C"      "sd_rR45CO244CO2" "sd_R45CO244CO2"  "sd_rd45CO244CO2"
# [5] "sd_d45CO244CO2"  "sd_R13C12C"



# keep "sd_d13C12C"
keep.vec <- c(keep.vec,"sd_d13C12C")
rm.vec <- c(rm.vec, "sd_rR45CO244CO2", "sd_R45CO244CO2", "sd_rd45CO244CO2",
            "sd_d45CO244CO2", "sd_R13C12C")


## 11. 
cor_var.list[[37]]
# [1] "sd_rR46CO244CO2"  "sd_R46CO244CO2"   "sd_rd46CO244CO2"  "sd_d46CO244CO2"  
# [5] "sd_R18O16O"       "sd_d18O16O"       "sd_R17O16O"       "sd_d17O16O"      
# [9] "sd_calib_d18O16O"

cor_var.list[[45]]
# [1] "sd_R46CO244CO2"   "sd_rR46CO244CO2"  "sd_rd46CO244CO2"  "sd_d46CO244CO2"  
# [5] "sd_R18O16O"       "sd_d18O16O"       "sd_R17O16O"       "sd_d17O16O"      
# [9] "sd_calib_d18O16O"

cor_var.list[[47]]
# [1] "sd_rd46CO244CO2"  "sd_rR46CO244CO2"  "sd_R46CO244CO2"   "sd_d46CO244CO2"  
# [5] "sd_R18O16O"       "sd_d18O16O"       "sd_R17O16O"       "sd_d17O16O"      
# [9] "sd_calib_d18O16O"

cor_var.list[[49]]
# [1] "sd_d46CO244CO2"   "sd_rR46CO244CO2"  "sd_R46CO244CO2"   "sd_rd46CO244CO2" 
# [5] "sd_R18O16O"       "sd_d18O16O"       "sd_R17O16O"       "sd_d17O16O"      
# [9] "sd_calib_d18O16O"

cor_var.list[[55]]
# [1] "sd_R18O16O"       "sd_rR46CO244CO2"  "sd_R46CO244CO2"   "sd_rd46CO244CO2" 
# [5] "sd_d46CO244CO2"   "sd_d18O16O"       "sd_R17O16O"       "sd_d17O16O"      
# [9] "sd_calib_d18O16O"

cor_var.list[[57]]
# [1] "sd_d18O16O"       "sd_rR46CO244CO2"  "sd_R46CO244CO2"   "sd_rd46CO244CO2" 
# [5] "sd_d46CO244CO2"   "sd_R18O16O"       "sd_R17O16O"       "sd_d17O16O"      
# [9] "sd_calib_d18O16O"

cor_var.list[[59]]
# [1] "sd_R17O16O"       "sd_rR46CO244CO2"  "sd_R46CO244CO2"   "sd_rd46CO244CO2" 
# [5] "sd_d46CO244CO2"   "sd_R18O16O"       "sd_d18O16O"       "sd_d17O16O"      
# [9] "sd_calib_d18O16O"

cor_var.list[[61]]
# [1] "sd_d17O16O"       "sd_rR46CO244CO2"  "sd_R46CO244CO2"   "sd_rd46CO244CO2" 
# [5] "sd_d46CO244CO2"   "sd_R18O16O"       "sd_d18O16O"       "sd_R17O16O"      
# [9] "sd_calib_d18O16O"

cor_var.list[[64]]
# [1] "sd_calib_d18O16O" "sd_rR46CO244CO2"  "sd_R46CO244CO2"   "sd_rd46CO244CO2" 
# [5] "sd_d46CO244CO2"   "sd_R18O16O"       "sd_d18O16O"       "sd_R17O16O"      
# [9] "sd_d17O16O" 

# keep "sd_calib_d18O16O"
keep.vec <- c(keep.vec,"sd_calib_d18O16O")
rm.vec <- c(rm.vec,"sd_rR46CO244CO2","sd_R46CO244CO2", "sd_rd46CO244CO2",
            "sd_d46CO244CO2","sd_R18O16O","sd_d18O16O","sd_R17O16O","sd_d17O16O")



## 12.
cor_var.list[[38]]
# [1] "avg_R45CO244CO2"  "avg_rd45CO244CO2" "avg_d45CO244CO2"  "avg_R13C12C"     
# [5] "avg_d13C12C" 

cor_var.list[[52]]
# [1] "avg_d13C12C"      "avg_R45CO244CO2"  "avg_rd45CO244CO2" "avg_d45CO244CO2" 
# [5] "avg_R13C12C"

cor_var.list[[40]]
# [1] "avg_rd45CO244CO2" "avg_R45CO244CO2"  "avg_d45CO244CO2"  "avg_R13C12C"     
# [5] "avg_d13C12C"

cor_var.list[[42]]
# [1] "avg_d45CO244CO2"  "avg_R45CO244CO2"  "avg_rd45CO244CO2" "avg_R13C12C"     
# [5] "avg_d13C12C" 

cor_var.list[[50]]
# [1] "avg_R13C12C"      "avg_R45CO244CO2"  "avg_rd45CO244CO2" "avg_d45CO244CO2" 
# [5] "avg_d13C12C" 


# keep "avg_d13C12C"
keep.vec <- c(keep.vec, "avg_d13C12C")
rm.vec <- c(rm.vec, "avg_R13C12C", "avg_R45CO244CO2", "avg_rd45CO244CO2",
            "avg_d45CO244CO2")


## 13.
cor_var.list[[48]]
# [1] "avg_d46CO244CO2"  "avg_R46CO244CO2"  "avg_rd46CO244CO2" "avg_R18O16O"     
# [5] "avg_d18O16O"      "avg_R17O16O"      "avg_d17O16O"

cor_var.list[[46]]
# [1] "avg_rd46CO244CO2" "avg_R46CO244CO2"  "avg_d46CO244CO2"  "avg_R18O16O"     
# [5] "avg_d18O16O"      "avg_R17O16O"      "avg_d17O16O"  

cor_var.list[[44]]
# [1] "avg_R46CO244CO2"  "avg_rd46CO244CO2" "avg_d46CO244CO2"  "avg_R18O16O"     
# [5] "avg_d18O16O"      "avg_R17O16O"      "avg_d17O16O" 

cor_var.list[[54]]
# [1] "avg_R18O16O"      "avg_R46CO244CO2"  "avg_rd46CO244CO2" "avg_d46CO244CO2" 
# [5] "avg_d18O16O"      "avg_R17O16O"      "avg_d17O16O"

cor_var.list[[56]]
# [1] "avg_d18O16O"      "avg_R46CO244CO2"  "avg_rd46CO244CO2" "avg_d46CO244CO2" 
# [5] "avg_R18O16O"      "avg_R17O16O"      "avg_d17O16O" 

cor_var.list[[58]]
# [1] "avg_R17O16O"      "avg_R46CO244CO2"  "avg_rd46CO244CO2" "avg_d46CO244CO2" 
# [5] "avg_R18O16O"      "avg_d18O16O"      "avg_d17O16O" 

cor_var.list[[60]]
# [1] "avg_d17O16O"      "avg_R46CO244CO2"  "avg_rd46CO244CO2" "avg_d46CO244CO2" 
# [5] "avg_R18O16O"      "avg_d18O16O"      "avg_R17O16O" 

 
# keep "avg_d18O16O"
keep.vec <- c(keep.vec, "avg_d18O16O")
rm.vec <- c(rm.vec, "avg_d17O16O", "avg_R46CO244CO2", "avg_rd46CO244CO2",
            "avg_d46CO244CO2", "avg_R18O16O",  "avg_R17O16O")


####################################################################################
# 21. URFP with highly correlated features removed (reduced redundancy feature space)
###################################################################################

#### new distance matrix 
# amend train.list/test.list to remove the highly correlated vars
head(colnames(train1.dat))
# [1] "Analysis"    "x_acf1"      "x_acf10"     "diff1_acf1"  "diff1_acf10" "diff2_acf1" 
tail(colnames(train1.dat))
# [1] "sd_pkArea"         "avg_d18O13C"       "sd_d18O13C"        "avg_calib_d18O16O"
# [5] "sd_calib_d18O16O"  "biotic" 


rm_vec.ind <- which(colnames(train1.dat) %in% rm.vec)

# create new training list 
new_train.list <- list()
new_test.list <- list()

for(i in seq(1,length(train.list))){
  new_train <- train.list[[i]][,-rm_vec.ind]
  new_test <- test.list[[i]][,-rm_vec.ind]
  new_train.list[[i]] <- new_train
  new_test.list[[i]]<- new_test
}

length(new_train.list)
length(new_test.list)


####### URFP
rm.ind <- which(colnames(new_train.list[[1]]) %in% c("Analysis","biotic"))

new_prox.list <- list()
for(i in seq(1,length(new_train.list))){
  synth <- as.data.frame(lapply(new_train.list[[i]][,-rm.ind], 
                                function(x) {sample(x, length(x), replace = TRUE)}))
  synth.dat<-rbind(data.frame(y="real",new_train.list[[i]][,-rm.ind]),
                   data.frame(y="synth",synth))
  synth.dat$y<-as.factor(synth.dat$y)
  # URF
  urf.fit <- ranger(y~., synth.dat, keep.inbag = TRUE,
                    num.trees=5000, mtry=2, 
                    importance="none",
                    local.importance = F, 
                    num.threads=4)
  prox <- extract_proximity_oob(urf.fit, synth.dat)[1:nrow(new_train.list[[i]]), 
                                                    1:nrow(new_train.list[[i]])]
  urfp.dist<-sqrt(1-prox)
  fileName <- paste("./replicates_data/dist_new","_",i,".csv",sep="")
  
  write.table(urfp.dist,fileName,row.names=F,col.names=F,quote=F,sep=",")
  new_prox.list[[i]] <- urfp.dist
}




#################################################################
# 22. NPDR-URF for data with highly correlated variables removed
################################################################

rm.ind <- which(colnames(new_train.list[[1]]) %in% c("Analysis"))

new_npdr_urf.list<-list()
for(i in seq(1,length(new_prox.list))){
  bio_npdr_urf <- npdr::npdr("biotic", new_train.list[[i]][-rm.ind],
                             regression.type="binomial",
                             attr.diff.type="numeric-abs",
                             nbd.method="relieff",
                             nbd.metric = "precomputed",
                             external.dist=new_prox.list[[i]],
                             knn=knnSURF.balanced(new_train.list[[i]]$biotic, 
                                                  sd.frac = .5),
                             use.glmnet = F, glmnet.alpha = 1, 
                             glmnet.lower = 0, glmnet.lam="lambda.min",
                             neighbor.sampling="none", dopar.nn = F, dopar.reg=F,
                             padj.method="bonferroni", verbose=T)
  bio_npdr_urf_scores<-bio_npdr_urf %>% filter(pval.adj<.05) %>% pull(att)
  bio_npdr_urf_scores
  new_npdr_urf.list[[i]]<-bio_npdr_urf_scores
  fileName <- paste("./replicates_data/bio_npdr_new_urf_scores","_",i,".csv",sep="")
  write.table(bio_npdr_urf_scores,fileName,row.names=F,quote=F,sep=",")
  
}

length(new_npdr_urf.list)
# [1] 5

# 1
new_npdr_urf.list[[1]]
# [1] "avg_rR45CO244CO2"  "avg_d13C12C"       "time_kl_shift"     "fluctanal_prop_r1"
# [5] "diff2_acf1"        "walker_propcross"

# 2
new_npdr_urf.list[[2]]
# [1] "avg_d13C12C"       "avg_rR45CO244CO2"  "fluctanal_prop_r1" "sd_d18O13C"       
# [5] "diff2_acf1" 

# 3
new_npdr_urf.list[[3]]
# [1] "avg_rR45CO244CO2"  "diff2_acf1"        "fluctanal_prop_r1" "trend"            
# [5] "avg_d13C12C"

# 4
new_npdr_urf.list[[4]]
# [1] "avg_rR45CO244CO2"  "avg_d13C12C"       "diff2_acf1"        "fluctanal_prop_r1"

# 5
new_npdr_urf.list[[5]]
#[1] "avg_rR45CO244CO2"  "fluctanal_prop_r1" "sd_d18O13C"        "diff2_acf1"       
#[5] "avg_d13C12C"       "avg_rR46CO244CO2"  "time_kl_shift"






##########################################################
# 23. tune the RR variable model: get hyperparameters
##########################################################


# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(new_train.list[[1]])[2]-3),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(5,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, savePredictions = T,
                        search="random",
                        allowParallel=T)

start<-Sys.time()
rf3<-train(biotic~., data=new_train.list[[1]][,-rm.ind], 
           method="ranger",
           importance="none",
           metric="Accuracy", 
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(new_train.list[[1]]$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 20.30494 mins

rf3$bestTune
#  mtry  splitrule min.node.size
#157    4 extratrees             5

# best accuracy
rf3$finalModel$prediction.error
# [1] 0.1
1-rf3$finalModel$prediction.error
# [1] 0.9172414

# saved tuned parameters
newFull_mtry<-rf3$bestTune$mtry
newFull_minNodeSize<-rf3$bestTune$min.node.size
newFull_splitRule<-rf3$bestTune$splitrule


# tune the number of trees using best params from previous run
fileName<-"maxtrees_out3.csv" # for parallel output

tuneGrid<-expand.grid(.mtry=newFull_mtry,
                      .splitrule=newFull_splitRule,
                      .min.node.size=newFull_minNodeSize)

trControl<-trainControl(method = "cv",number=5,
                        verboseIter = T,savePredictions = T,
                        search="random",
                        allowParallel=F # parallelize across number of trees
)

start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=new_train.list[[1]][,-rm.ind],
            method="ranger",
            importance="none", 
            metric="Accuracy", 
            trControl=trControl, 
            tuneGrid=tuneGrid,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(new_train.list[[1]]$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
#Time difference of 47.13818 secs

tree.results<-read.csv("maxtrees_out3.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
max_acc<-tree.results[which.max(tree.results$Accuracy),]
max_acc 
#   maxtrees  Accuracy
# 1     3000 0.907599
newFull_maxTree<-5000#maxFull_acc$maxtrees



### run a final model
new_train.list[[1]]$biotic <- as.factor(new_train.list[[1]]$biotic)
rfFinal.fit <- ranger(biotic~., new_train.list[[1]][,-rm.ind], 
                               keep.inbag = TRUE,
                               num.trees=newFull_maxTree, 
                               mtry=newFull_mtry, 
                               importance="permutation", 
                               splitrule = newFull_splitRule,
                               min.node.size=newFull_minNodeSize,
                               class.weights = as.numeric(c(1/table(new_train.list[[1]]$biotic))),
                               scale.permutation.importance = T,
                               local.importance = T, num.threads=4)
sorted.imp<-sort(rfFinal.fit$variable.importance,decreasing=T)
rf.feat<-sorted.imp[1:10] # save top nine for further analysis
names(rf.feat) # top 10
# [1] "max_kl_shift"       "fluctanal_prop_r1"  "avg_rR45CO244CO2"   "localsimple_taures"
# [5] "time_level_shift"   "diff2_acf10"        "diff2x_pacf5"       "diff2_acf1"        
# [9] "max_var_shift"      "x_pacf5"   

rfFinal.fit$confusion.matrix
#         predicted
# true      abiotic biotic
# abiotic      81      8
# biotic        9     42


rfFinal.fit$prediction.error
# [1] 0.1214286
1-rfFinal.fit$prediction.error
# [1] 0.8785714

predFinal.test<-predict(rfFinal.fit,data=new_test.list[[1]][,-rm.ind])
new_test.list[[1]]$biotic <- as.factor(new_test.list[[1]]$biotic)
confusionMatrix(predFinal.test$predictions,new_test.list[[1]]$biotic)
# Accuracy : 0.8824 


table(new_test.list[[1]]$biotic, predFinal.test$predictions)
#          predicted
# true    abiotic biotic
# abiotic      18      4
# biotic        0     12    


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfFinal.fit, "./replicates_data/bioAbio_noCorVarRF_bestFit.rds")

# save predictions
fullTrain.dat<-new_train.list[[1]]
fullTrain.dat$pred<-rfFinal.fit$predictions
fullTest.dat<-new_test.list[[1]]
fullTest.dat$pred<-predFinal.test$predictions
write.table(fullTrain.dat,"./replicates_data/bioAbio_noCorVarRF_train.csv",quote=F,row.names=F,sep=",")
write.table(fullTest.dat,"./replicates_data/bioAbio_noCorVarRF_test.csv",quote=F,row.names=F,sep=",")



################################################
# 24. RR with test/train split run 0: set up data 
################################################

## read in data again if needed
bio_0.3p_full_train.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_train_0.3p.csv")
bio_0.3p_full_test.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_test_0.3p.csv")
bio_0.3p_full.dat <- rbind.data.frame(bio_0.3p_full_train.dat,bio_0.3p_full_test.dat)
dim(bio_0.3p_full.dat)
# [1] 174 108
head(colnames(bio_0.3p_full.dat))
# [1] "Analysis"    "x_acf1"      "x_acf10"     "diff1_acf1"  "diff1_acf10"
# [6] "diff2_acf1" 
tail(colnames(bio_0.3p_full.dat))
# [1] "sd_d18O13C"        "avg_calib_d18O16O" "sd_calib_d18O16O"  "biotic"           
# [5] "dataset"           "pred" 


bio_noCor_0.3p_train.dat <- read.csv("./replicates_data/bioAbio_noCorVarRF_train.csv")
bio_noCor_0.3p_test.dat <- read.csv("./replicates_data/bioAbio_noCorVarRF_test.csv")
bio_noCor_0.3p.dat <- rbind.data.frame(bio_noCor_0.3p_train.dat,bio_noCor_0.3p_test.dat)
dim(bio_noCor_0.3p.dat)
# [1] 174  54
colnames(bio_noCor_0.3p.dat)
# [1] "Analysis"                   "diff1_acf1"                 "diff1_acf10"               
# [4] "diff2_acf1"                 "diff2_acf10"                "ARCH.LM"                   
# [7] "entropy"                    "flat_spots"                 "arch_acf"                  
# [10] "garch_acf"                  "arch_r2"                    "garch_r2"                  
# [13] "alpha"                      "beta"                       "lumpiness"                 
# [16] "max_kl_shift"               "time_kl_shift"              "max_level_shift"           
# [19] "time_level_shift"           "max_var_shift"              "nonlinearity"              
# [22] "x_pacf5"                    "diff2x_pacf5"               "stability"                 
# [25] "trend"                      "spike"                      "linearity"                 
# [28] "curvature"                  "e_acf1"                     "unitroot_kpss"             
# [31] "firstzero_ac"               "fluctanal_prop_r1"          "histogram_mode"            
# [34] "localsimple_taures"         "motiftwo_entro3"            "outlierinclude_mdrmd"      
# [37] "sampenc"                    "std1st_der"                 "trev_num"                  
# [40] "spreadrandomlocal_meantaul" "walker_propcross"           "avg_rR45CO244CO2"          
# [43] "avg_rR46CO244CO2"           "avg_d13C12C"                "sd_d13C12C"                
# [46] "avg_d18O16O"                "avg_pkArea"                 "sd_pkArea"                 
# [49] "avg_d18O13C"                "sd_d18O13C"                 "avg_calib_d18O16O"         
# [52] "sd_calib_d18O16O"           "biotic"                     "pred"

redCorFeat <- colnames(bio_noCor_0.3p.dat)[-54]

# get train analyses
train.an <- bio_0.3p_full_train.dat$Analysis
head(train.an)
# [1] 2961 2962 2963 2965 2966 2968
length(train.an)
# [1] 140

test.an <- bio_0.3p_full_test.dat$Analysis
head(test.an)
# [1] 2960 2992 2996 3231 3232 3255
length(test.an)
# [1] 34

train.ind <- which(bio_noCor_0.3p.dat$Analysis %in% train.an)
length(train.ind)
# [1] 140
test.ind <- which(bio_noCor_0.3p.dat$Analysis %in% test.an)
length(test.ind)
# [1] 34

bio_lowCor_train.dat <- bio_noCor_0.3p.dat[train.ind,]
head(bio_lowCor_train.dat)
# remove pred col
which(colnames(bio_lowCor_train.dat)=="pred")
# 54
bio_lowCor_train.dat <- bio_lowCor_train.dat[,-54]
colnames(bio_lowCor_train.dat)

bio_lowCor_test.dat <- bio_noCor_0.3p.dat[test.ind,] 
head(bio_lowCor_test.dat)
# remove pred col - same as train
bio_lowCor_test.dat <- bio_lowCor_test.dat[,-54]
colnames(bio_lowCor_test.dat)

# remove Analysis and biotic for RF
rm.ind <- which(colnames(bio_noCor_0.3p.dat) %in% c("Analysis","pred"))
rm.ind
# [1]  1 54



#########################################
# 25. tune the model: RR variables, run 0
#########################################


# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(bio_lowCor_train.dat)[2]-3),by=1),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(1,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, 
                        search="random",
                        allowParallel=T)

start<-Sys.time()
rfa<-train(biotic~., data=bio_lowCor_train.dat[,-rm.ind], 
           method="ranger",
           importance="none",
           metric="Accuracy",
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(bio_lowCor_train.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 25.69949 mins

rfa$bestTune
#  mtry  splitrule min.node.size
#541    8 extratrees             1

# best accuracy
rfa$finalModel$prediction.error
# [1] 0.1214286
1-rfa$finalModel$prediction.error
# [1] 0.8785714

# saved tuned parameters
lowCor_mtry<-rfa$bestTune$mtry
lowCor_minNodeSize<-rfa$bestTune$min.node.size
lowCor_splitRule<-rfa$bestTune$splitrule

# tune the number of trees using best params from previous run
fileName<-"maxtrees_outa.csv" # for parallel output

tuneGrid<-expand.grid(.mtry=lowCor_mtry,
                      .splitrule=lowCor_splitRule,
                      .min.node.size=lowCor_minNodeSize)

trControl<-trainControl(method = "cv",number=5,
                        verboseIter = T, savePredictions=TRUE,
                        search="random",
                        allowParallel=F # parallelize across number of trees
)

start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=bio_lowCor_train.dat[,-rm.ind],
            method="ranger",
            importance="none", #"none"?
            metric="Accuracy", 
            trControl=trControl, 
            tuneGrid=tuneGrid,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(bio_lowCor_train.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 1.024512 mins

tree.results<-read.csv("maxtrees_outa.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
max_acc<-tree.results[which.max(tree.results$Accuracy),]
max_acc 
#  maxtrees  Accuracy
#1     3000 0.8938241
lowCor_maxTree<-5000#maxFull_acc$maxtrees



### run a final model
bio_lowCor_train.dat$biotic <- as.factor(bio_lowCor_train.dat$biotic)
rfFinal.fit <- ranger(biotic~., bio_lowCor_train.dat[,-rm.ind], 
                      keep.inbag = TRUE,
                      num.trees=lowCor_maxTree, 
                      mtry=lowCor_mtry, 
                      importance="permutation", 
                      splitrule = lowCor_splitRule,
                      min.node.size=lowCor_minNodeSize,
                      class.weights = as.numeric(c(1/table(bio_lowCor_train.dat$biotic))),
                      scale.permutation.importance = T,
                      local.importance = T, num.threads=4)
sorted.imp<-sort(rfFinal.fit$variable.importance,decreasing=TRUE)
rf.feat<-sorted.imp[1:10] # save top nine for further analysis
names(rf.feat) # top 10
# [1] "max_kl_shift"       "localsimple_taures" "fluctanal_prop_r1"  "avg_rR45CO244CO2"  
# [5] "time_level_shift"   "avg_d13C12C"        "diff2x_pacf5"       "diff2_acf10"       
# [9] "diff2_acf1"         "max_level_shift"   
rfFinal.fit$confusion.matrix 
#         predicted
# true      abiotic biotic
# abiotic      80      9
# biotic       10     41

rfFinal.fit$prediction.error 
# [1] 0.1357143
1-rfFinal.fit$prediction.error
# [[1] 0.8642857

predFinal.test<-predict(rfFinal.fit,data=bio_lowCor_test.dat[,-rm.ind])
bio_lowCor_test.dat[,-rm.ind]$biotic <- as.factor(bio_lowCor_test.dat[,-rm.ind]$biotic)
confusionMatrix(predFinal.test$predictions,bio_lowCor_test.dat[,-rm.ind]$biotic)
#     Reference
# Prediction  Reference
# Prediction abiotic biotic
# abiotic      22      3
# biotic        0      9
#    


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfFinal.fit, "./replicates_data/bioAbio_lowCorVarRF_bestFit.rds")

# save predictions
fullTrain.dat<-bio_lowCor_train.dat
fullTrain.dat$pred<-rfFinal.fit$predictions
fullTest.dat<-bio_lowCor_test.dat
fullTest.dat$pred<-predFinal.test$predictions
write.table(fullTrain.dat,"./replicates_data/bioAbio_lowCorVarRF_train.csv",quote=F,row.names=F,sep=",")
write.table(fullTest.dat,"./replicates_data/bioAbio_lowCorVarRF_test.csv",quote=F,row.names=F,sep=",")



###########################################################
# 26. RF for deltas - uncalibrated: run 0
#########################################################

# uncalibrated first
delta.ind <- which(colnames(bio_lowCor_train.dat) %in% c("avg_d18O16O","avg_d13C12C","biotic"))


deltas_train.dat <- bio_lowCor_train.dat[,delta.ind]
deltas_test.dat <- bio_lowCor_test.dat[,delta.ind]
colnames(deltas_train.dat)
#rm_ind_delta <- 3 #biotic



# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(deltas_train.dat)[2]-2)),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(1,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, 
                        search="random",
                        allowParallel=T)

start<-Sys.time()
rfb<-train(biotic~., data=deltas_train.dat, 
           method="ranger",
           importance="none",
           metric="Accuracy",
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(deltas_train.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 2.398593 mins

rfb$bestTune
#  mtry  splitrule min.node.size
# 4    2 extratrees             4

# best accuracy
rfb$finalModel$prediction.error
# [1] 0.2571429
1-rfb$finalModel$prediction.error
# [1] 0.7428571

# saved tuned parameters
deltas_mtry<-rfb$bestTune$mtry
deltas_minNodeSize<-rfb$bestTune$min.node.size
deltas_splitRule<-rfb$bestTune$splitrule

# tune the number of trees using best params from previous run
fileName<-"maxtrees_outb.csv" # for parallel output

tuneGrid<-expand.grid(.mtry=deltas_mtry,
                      .splitrule=deltas_splitRule,
                      .min.node.size=deltas_minNodeSize)

trControl<-trainControl(method = "cv",number=5,
                        verboseIter = T, savePredictions=TRUE,
                        search="random",
                        allowParallel=F # parallelize across number of trees
)

start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=deltas_train.dat,
            method="ranger",
            importance="none", 
            metric="Accuracy", 
            trControl=trControl, 
            tuneGrid=tuneGrid,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(deltas_train.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 46.13441 secs

tree.results<-read.csv("maxtrees_outb.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
max_acc<-tree.results[which.max(tree.results$Accuracy),]
max_acc 
#   maxtrees  Accuracy
#24    13000 0.7439792
deltas_maxTree<-max_acc$maxtrees #5000



### run a final model
deltas_train.dat$biotic <- as.factor(deltas_train.dat$biotic)

rfFinal.fit <- ranger(biotic~., deltas_train.dat, 
                      keep.inbag = TRUE,
                      num.trees=deltas_maxTree, 
                      mtry=deltas_mtry, 
                      importance="permutation", 
                      splitrule = deltas_splitRule,
                      min.node.size=deltas_minNodeSize,
                      class.weights = as.numeric(c(1/table(deltas_train.dat$biotic))),
                      scale.permutation.importance = T,
                      local.importance = T, num.threads=4)
sorted.imp<-sort(rfFinal.fit$variable.importance,decreasing=TRUE)
rf.feat<-sorted.imp
names(rf.feat) 
#  [1] "avg_d13C12C" "avg_d18O16O"

rfDelta.fit <- rfFinal.fit
rfDelta.fit$confusion.matrix 
#   predicted
# true      abiotic biotic
# abiotic      69     20
# biotic       18     33

rfDelta.fit$prediction.error 
# [1] 0.2714286
1-rfDelta.fit$prediction.error
# [1] 0.7285714

predFinal.test<-predict(rfDelta.fit,data=deltas_test.dat)
deltas_test.dat$biotic <- as.factor(deltas_test.dat$biotic)
confusionMatrix(predFinal.test$predictions,deltas_test.dat$biotic)
# Accuracy : 0.6176 
#    Reference
#  Prediction 

table(deltas_test.dat$biotic, predFinal.test$predictions) 
#          predicted
# true    abiotic biotic
# abiotic      16      7
# biotic        6      5  


#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfDelta.fit, "./replicates_data/bioAbio_deltaVarRF_bestFit.rds")

# save predictions
fullTrain.dat<-deltas_train.dat
fullTrain.dat$pred<-rfDelta.fit$predictions
fullTest.dat<-deltas_test.dat
fullTest.dat$pred<-predFinal.test$predictions
write.table(fullTrain.dat,"./replicates_data/bioAbio_deltaVarRF_train.csv",quote=F,row.names=F,sep=",")
write.table(fullTest.dat,"./replicates_data/bioAbio_deltaVarRF_test.csv",quote=F,row.names=F,sep=",")



#####################################
# 27. RF for calibrated deltas: run 0
####################################
calib_delta.ind <- which(colnames(bio_lowCor_train.dat) %in% c("avg_calib_d18O16O","avg_d13C12C","biotic"))


calib_deltas_train.dat <- bio_lowCor_train.dat[,calib_delta.ind]
calib_deltas_test.dat <- bio_lowCor_test.dat[,calib_delta.ind]
colnames(calib_deltas_train.dat)
# [1] "avg_d13C12C"       "avg_calib_d18O16O" "biotic"    


# define tuning grid
tuneGrid<-expand.grid(
  .mtry=seq(2,1.5*sqrt(dim(calib_deltas_train.dat)[2]-2)),
  .splitrule=c("extratrees","gini","hellinger"),
  .min.node.size=seq(1,30))

trControl<-trainControl(method = "cv", number=5,
                        verboseIter = T, 
                        search="random",
                        allowParallel=T)

start<-Sys.time()
rfc<-train(biotic~., data=calib_deltas_train.dat, 
           method="ranger",
           importance="none",
           metric="Accuracy",
           trControl=trControl, 
           tuneGrid=tuneGrid,
           num.trees=5000,
           class.weights = as.numeric(c(1/table(calib_deltas_train.dat$biotic))),
           verbose=T)
end<-Sys.time()
rf.time<-end-start
rf.time
# Time difference of 2.424989 mins

rfc$bestTune
#  mtry  splitrule min.node.size
#3    2 extratrees             3 

# best accuracy
rfc$finalModel$prediction.error
# [1] 0.2571429
1-rfc$finalModel$prediction.error
# [1] 0.7428571

# saved tuned parameters
calib_deltas_mtry<-rfc$bestTune$mtry
calib_deltas_minNodeSize<-rfc$bestTune$min.node.size
calib_deltas_splitRule<-rfc$bestTune$splitrule

# tune the number of trees using best params from previous run
fileName<-"maxtrees_outc.csv" # for parallel output

tuneGrid<-expand.grid(.mtry=calib_deltas_mtry,
                      .splitrule=calib_deltas_splitRule,
                      .min.node.size=calib_deltas_minNodeSize)

trControl<-trainControl(method = "cv",number=5,
                        verboseIter = T, savePredictions=TRUE,
                        search="random",
                        allowParallel=F # parallelize across number of trees
)


start<-Sys.time()
foreach(max_tree=seq(3000,15000,by=1000)) %dopar% {
  set.seed(1234)
  rf<-train(biotic~., data=calib_deltas_train.dat,
            method="ranger",
            importance="none",
            metric="Accuracy", 
            trControl=trControl, 
            tuneGrid=tuneGrid,
            num.trees=max_tree,
            class.weights = as.numeric(c(1/table(calib_deltas_train.dat$biotic))),
            verbose=T
  )
  curr_acc<-cbind.data.frame(max_tree,rf$results$Accuracy)
  write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
}
end<-Sys.time()
rf.time<-end-start
rf.time
#Time difference of 51.02032 secs

tree.results<-read.csv("maxtrees_outc.csv",header=F)
colnames(tree.results)<-c("maxtrees","Accuracy")
max_acc<-tree.results[which.max(tree.results$Accuracy),]
max_acc 
#    maxtrees  Accuracy
# 1     3000 0.7506477
calib_deltas_maxTree<-5000#maxFull_acc$maxtrees #5000



### run a final model
calib_deltas_train.dat$biotic <- as.factor(calib_deltas_train.dat$biotic)

rfFinal.fit <- ranger(biotic~., calib_deltas_train.dat, 
                      keep.inbag = TRUE,
                      num.trees=calib_deltas_maxTree, 
                      mtry=calib_deltas_mtry, 
                      importance="permutation", 
                      splitrule = calib_deltas_splitRule,
                      min.node.size=calib_deltas_minNodeSize,
                      class.weights = as.numeric(c(1/table(calib_deltas_train.dat$biotic))),
                      scale.permutation.importance = T,
                      local.importance = T, num.threads=4)
sorted.imp<-sort(rfFinal.fit$variable.importance,decreasing=TRUE)
rf.feat<-sorted.imp 
names(rf.feat) 
#[1] "avg_d13C12C"       "avg_calib_d18O16O"

rfFinal.fit$confusion.matrix 
#        predicted
# true    abiotic biotic
# abiotic      74     15
# biotic       21     30    

rfFinal.fit$prediction.error 
# [1] 0.2571429
1-rfFinal.fit$prediction.error
# [1] 0.7428571

predFinal.test<-predict(rfFinal.fit,data=calib_deltas_test.dat)
calib_deltas_test.dat$biotic <- as.factor(calib_deltas_test.dat$biotic)
confusionMatrix(predFinal.test$predictions,calib_deltas_test.dat$biotic)
#              Reference
# Prediction abiotic biotic
# abiotic      19      6
# biotic        3      6 

table(calib_deltas_test.dat$biotic, predFinal.test$predictions) 
#          predicted
# true    abiotic biotic
# abiotic      19      3
# biotic        6      6
# Accuracy : 0.7353 

#### save model, test, train and predicted values
# save best model, training and testing data
saveRDS(rfCalibDelta.fit, "./replicates_data/bioAbio_CalibDeltaVarRF_bestFit.rds")

# save predictions
fullTrain.dat<-calib_deltas_train.dat
fullTrain.dat$pred<-rfFinal.fit$predictions
fullTest.dat<-calib_deltas_test.dat
fullTest.dat$pred<-predFinal.test$predictions
write.table(fullTrain.dat,"./replicates_data/bioAbio_calibDeltaVarRF_train.csv",quote=F,row.names=F,sep=",")
write.table(fullTest.dat,"./replicates_data/bioAbio_calibDeltaVarRF_test.csv",quote=F,row.names=F,sep=",")





###################################################
# 28. function to tune model for new random splits 
###################################################


tuneModelHyperParams <- function(train.df, remove.ind, classification=TRUE,
                                 maxNode=30, outcome.ind, maxTrees1=5000,
                                 maxTrees2=15000, nfolds=5, accMetric="Accuracy",
                                 useMinTrees=TRUE, minTrees=5000,
                                 fileName="maxTrees_outx.csv"){
  ret.list <- list()
  numVars <- dim(train.df)[2] - length(remove.ind)
  
  # set tuning grid for three hyperparams
  if(classification){
    mtry.vec <- seq(2,2*sqrt(numVars))
    min_node.vec <- seq(1,maxNode)
    splitrule.vec <- c("extratrees","gini","hellinger")
  }else{# regression
    mtry.vec <- seq(2,floor(numVars)/3)
    min_node.vec <- seq(2,maxNode)
    splitrule.vec <- c("extratrees","variance","beta") #maxstat?
  }
  
  # define tuning grid
  tuneGrid1<-expand.grid(
    .mtry = mtry.vec,
    .splitrule = splitrule.vec,
    .min.node.size= min_node.vec)
  
  trControl1<-trainControl(method = "cv", 
                           number=nfolds,
                          verboseIter = T, 
                          search="random",
                          allowParallel=T)
 
  rf1<-train(x=train.df[,-remove.ind], 
             y=train.df[,outcome.ind],
             method="ranger",
             importance="none",
             metric=accMetric,
             trControl=trControl1, 
             tuneGrid=tuneGrid1,
             num.trees=maxTrees1,
             class.weights = as.numeric(c(1/table(train.df[,outcome.ind]))),
             verbose=T)
  
  
  bestTune1 <- rf1$bestTune
  
  if(classification){
    # best accuracy
    bestAcc1 <- 1-rf1$finalModel$prediction.error
  }else{
    # use RMSE
    bestAcc1 <- rf1$finalModel$RMSE
  }
  
  ret.list$tunedCM1 <- rf1$finalModel$confusion.matrix
  ret.list$tunedHyperparams <- bestTune1
  ret.list$bestAcc1 <- bestAcc1
  
  # saved tuned parameters
  best_mtry<-rf1$bestTune$mtry
  best_minNodeSize<-rf1$bestTune$min.node.size
  best_splitRule<-rf1$bestTune$splitrule
  
  # tune the number of trees using best params from previous run
  #fileName<-"maxtrees_outx.csv" # for parallel output

  tuneGrid2<-expand.grid(.mtry=best_mtry,
                        .splitrule=best_splitRule,
                        .min.node.size=best_minNodeSize)
  
  trControl2<-trainControl(method = "cv",
                           number=nfolds,
                          verboseIter = T, 
                          savePredictions=TRUE,
                          search="random",
                          allowParallel=F # parallelize across number of trees
                          )
  
  foreach(max_tree=seq(3000,maxTrees2,by=1000)) %dopar% {
    set.seed(1234)
    rf2<-train(x=train.df[,-remove.ind],
              y=train.df[,outcome.ind],
              method="ranger",
              importance="none",
              metric=accMetric, 
              trControl=trControl2, 
              tuneGrid=tuneGrid2,
              num.trees=max_tree,
              class.weights = as.numeric(c(1/table(train.df[,outcome.ind]))),
              verbose=T)
    if(classification){
      curr_acc<-cbind.data.frame(max_tree,rf2$results$Accuracy)
    }else{
      curr_acc<-cbind.data.frame(max_tree,rf2$results$RMSE)
    }
    write.table(curr_acc,fileName,quote=F,append=T,col.names=F,row.names=F,sep=",")
  }
  
  tree.results<-read.csv(fileName,header=F)
  colnames(tree.results)<-c("maxtrees","Accuracy")
  if(classification){
    max_acc<-tree.results[which.max(tree.results$Accuracy),]
  }else{
    max_acc<-tree.results[which.min(tree.results$Accuracy),]
  }
  
  maxTrees <- max_acc$maxtrees
  
  ret.list$tuneTrees <- maxTrees
  ret.list$tuneTreeAcc <- max_acc$Accuracy
  
  # use at least 5000 trees if tuned amt below this
  if(maxTrees < 5000 && useMinTrees){
    maxTrees = minTrees
  }
  
  ret.list$numTrees <- maxTrees
  file.remove(fileName)
  return(ret.list)
}


################################################################################
# 29. RF Replicates for selected feature spaces: incorporate new spaces (deltas)
###############################################################################
# read in data and previous predictions
train1.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set1_train.csv")
dim(train1.dat)
#[1] 140 107
head(colnames(train1.dat))
# [1] "Analysis"    "x_acf1"      "x_acf10"     "diff1_acf1"  "diff1_acf10"
# [6] "diff2_acf1" 
tail(colnames(train1.dat))
# [1] "avg_d18O13C"       "sd_d18O13C"        "avg_calib_d18O16O" "sd_calib_d18O16O" 
# [5] "biotic"            "pred" 
test1.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set1_test.csv")

# rm pred
which(colnames(train1.dat)=="pred")
# [1] 107
train1.dat <- train1.dat[,-107]
test1.dat <- test1.dat[,-107]

train2.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set2_train.csv")
test2.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set2_test.csv")
train2.dat <- train2.dat[,-107]
test2.dat <- test2.dat[,-107]
dim(train2.dat)
#[1] 140 106

train3.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set3_train.csv")
test3.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set3_test.csv")
train3.dat <- train3.dat[,-107]
test3.dat <- test3.dat[,-107]
dim(train3.dat)
#[1] 140 106

train4.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set4_train.csv")
test4.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set4_test.csv")
train4.dat <- train4.dat[,-107]
test4.dat <- test4.dat[,-107]
dim(train4.dat)
#[1] 140 106

train5.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set5_train.csv")
test5.dat <- read.csv("./replicates_data/bioAbio_fullVarRF_set5_test.csv")
train5.dat <- train5.dat[,-107]
test5.dat <- test5.dat[,-107]
dim(train3.dat)
#[1] 140 106


### for loop: match analysis numbers in data
# rm pred var, Analysis, bio (get indices)
### RF loop

train.list <- list(train1.dat, train2.dat, train3.dat, train4.dat, train5.dat)
test.list <- list(test1.dat,test2.dat,test3.dat,test4.dat,test5.dat)

# create new training and testing lists for reduced correlation features, 
# NPDR-LURF, NPDR-LMan, rrTSMS, Deltas, CDeltas
lurf_train.list <- list()
lurf_test.list <- list()

lman_train.list <- list()
lman_test.list <- list()

rr_train.list <- list()
rr_test.list <- list()

deltas_train.list <- list()
deltas_test.list <- list()

cDeltas_train.list <- list()
cDeltas_test.list <- list()

# rrTSMS
redCor.ind <- which(colnames(train.list[[1]]) %in% redCorFeat)
for(i in seq(1,length(train.list))){
  new_train <- train.list[[i]][redCor.ind]
  new_test <- test.list[[i]][,redCor.ind]
  rr_train.list[[i]] <- new_train
  rr_test.list[[i]]<- new_test
}


# deltas
deltas_feat <- c("avg_d18O16O","avg_d13C12C")
deltas.ind <- c("Analysis",deltas_feat,"biotic")
calib_deltas_feat <- c("avg_calib_d18O16O","avg_d13C12C")
calibDeltas.ind <- c("Analysis",calib_deltas_feat,"biotic")
for(i in seq(1,length(train.list))){
  deltas_train <- train.list[[i]][deltas.ind]
  deltas_test <- test.list[[i]][,deltas.ind]
  deltas_train.list[[i]] <- deltas_train
  deltas_test.list[[i]]<- deltas_test
  
  calib_deltas_train <- train.list[[i]][calibDeltas.ind]
  calib_deltas_test <- test.list[[i]][,calibDeltas.ind]
  calib_deltas_train.list[[i]] <- calib_deltas_train
  calib_deltas_test.list[[i]]<- calib_deltas_test
  
}


### lman and lurf features for the 5 sets
lman1 <- read.csv("./replicates_data/bio_npdr_lman_scores_1.csv")
lman1 <- lman1$features
lman2 <- read.csv("./replicates_data/bio_npdr_lman_scores_2.csv")
lman2 <- lman2$features
lman3 <- read.csv("./replicates_data/bio_npdr_lman_scores_3.csv")
lman3 <- lman3$features
lman4 <- read.csv("./replicates_data/bio_npdr_lman_scores_4.csv")
lman4 <- lman4$features
lman5 <- read.csv("./replicates_data/bio_npdr_lman_scores_5.csv")
lman5 <- lman5$features

lman_feat.list <- list(lman1, lman2, lman3, lman4, lman5)

lurf1 <- read.csv("./replicates_data/bio_npdr_lurf_scores_1.csv")
lurf1 <- lurf1$features
lurf2 <- read.csv("./replicates_data/bio_npdr_lurf_scores_2.csv")
lurf2 <- lurf2$features
lurf3 <- read.csv("./replicates_data/bio_npdr_lurf_scores_3.csv")
lurf3 <- lurf3$features
lurf4 <- read.csv("./replicates_data/bio_npdr_lurf_scores_4.csv")
lurf4 <- lurf4$features
lurf5 <- read.csv("./replicates_data/bio_npdr_lurf_scores_5.csv")
lurf5 <- lurf5$features

lurf_feat.list <- list(lurf1,lurf2,lurf3,lurf4,lurf5)

## make lists of train/test data: lurf/lman
for(i in seq(1,length(train.list))){
  # get feature indices
  lurf.ind <- which(colnames(train.list[[i]]) %in% 
                   c("Analysis",lurf_feat.list[[i]],"biotic"))
  lman.ind <- which(colnames(train.list[[i]]) %in%
                    c("Analysis",lman_feat.list[[i]],"biotic"))
  # create train and test data
  lurf_train <- train.list[[i]][,lurf.ind]
  lurf_test <- test.list[[i]][,lurf.ind]
  lurf_train.list[[i]] <- lurf_train
  lurf_test.list[[i]]<- lurf_test
  
  lman_train <- train.list[[i]][,lman.ind]
  lman_test <- test.list[[i]][,lman.ind]
  lman_train.list[[i]] <- lman_train
  lman_test.list[[i]] <- lman_test
  
}
head(lurf_train.list[[1]])
head(lman_train.list[[1]])


# store results
rr_rf.list <- list()
rr_train_cm.list <- list()
rr_test_cm.list <- list()
rr_train_acc.list <- list()
rr_test_acc.list <- list()

deltas_rf.list <- list()
deltas_train_cm.list <- list()
deltas_test_cm.list <- list()
deltas_train_acc.list <- list()
deltas_test_acc.list <- list()

calib_deltas_rf.list <- list()
calib_deltas_train_cm.list <- list()
calib_deltas_test_cm.list <- list()
calib_deltas_train_acc.list <- list()
calib_deltas_test_acc.list <- list()

lurf_rf.list <- list()
lurf_train_cm.list <- list()
lurf_test_cm.list <- list()
lurf_train_acc.list <- list()
lurf_test_acc.list <- list()

lman_rf.list <- list()
lman_train_cm.list <- list()
lman_test_cm.list <- list()
lman_train_acc.list <- list()
lman_test_acc.list <- list()


hyperparam_name.vec <- c()
hyperparams.list <- list()
for(i in seq(2,length(rr_train.list))){
  rm.ind <- 1 # rm Analysis num for RF
  
  ###########
  #### train and test sets
  # reduced redundancy variable space
  rr_train.dat <- rr_train.list[[i]] 
  rr_test.dat <- rr_test.list[[i]]
  #dim(rr_train.dat) # has analysis and biotic
  # [1] 140  53 
  #dim(rr_test.dat)
  # [1] 34 53
  
  # deltas space
  deltas_train.dat <- deltas_train.list[[i]]
  deltas_test.dat <- deltas_test.list[[i]]
  dim(deltas_train.dat)
  dim(deltas_test.dat)
  # [1] 34  4
  
  # calib_deltas space
  calib_deltas_train.dat <- calib_deltas_train.list[[i]]
  calib_deltas_test.dat <- calib_deltas_test.list[[i]]
  dim(calib_deltas_train.dat)
  dim(calib_deltas_test.dat)
  # [1] 34  4
  
  # lurf space
  lurf_train.dat <- lurf_train.list[[i]]
  lurf_test.dat <- lurf_test.list[[i]]
  dim(lurf_train.dat)
  dim(lurf_test.dat)
  # [1] 34  9
  
  # lman space
  lman_train.dat <- lman_train.list[[i]]
  lman_test.dat <- lman_test.list[[i]]
  dim(lman_train.dat)
  dim(lman_test.dat)
  #[1] 34  9
  
  ##########
  # tune the models
  
  ### rr 
  rr_out.ind <- which(colnames(rr_train.dat) == "biotic")
  rr_tune.list <- tuneModelHyperParams(train.df = rr_train.dat, 
                                       remove.ind = rm.ind,
                                       outcome.ind = rr_out.ind,
                                       fileName="maxTrees_out_rr.csv")
  rr_params <- rr_tune.list$tunedHyperparams
  rr_params
  #     mtry  splitrule min.node.size
  #361    6 extratrees             1
  rr_mtry <- rr_params$mtry
  rr_splitRule <- rr_params$splitrule
  rr_minNodeSize <- rr_params$min.node.size
  rr_maxTree <- rr_tune.list$numTrees
  # hyperparameter df
  rr_hyperparams.df <- as.data.frame(matrix(rep(NA,5),ncol=5))
  colnames(rr_hyperparams.df) <- c("var_space","mtry","splitrule","min_nodesize","ntrees")
  rr_hyperparams.df$var_space <- "rr"
  rr_hyperparams.df$mtry <- rr_mtry
  rr_hyperparams.df$splitrule <- rr_splitRule
  rr_hyperparams.df$min_nodesize <-rr_minNodeSize
  rr_hyperparams.df$ntrees <- rr_maxTree
  rr_hyperparams.df
  # var_space mtry  splitrule min_nodesize ntrees
  # rr    6 extratrees            1   5000
    
  ### deltas
  deltas_out.ind <- which(colnames(deltas_train.dat) == "biotic")
  deltas_tune.list <- tuneModelHyperParams(train.df = deltas_train.dat, 
                                       remove.ind = rm.ind,
                                       outcome.ind = deltas_out.ind,
                                       fileName="maxTrees_out_deltas.csv")
  deltas_params <- deltas_tune.list$tunedHyperparams
  deltas_params
  #     mtry  splitrule min.node.size
  #   1    2 extratrees             1
  deltas_mtry <- deltas_params$mtry
  deltas_splitRule <- deltas_params$splitrule
  deltas_minNodeSize <- deltas_params$min.node.size
  deltas_maxTree <- deltas_tune.list$numTrees
  # hyperparameter df
  deltas_hyperparams.df <- as.data.frame(matrix(rep(NA,5),ncol=5))
  colnames(deltas_hyperparams.df) <- c("var_space","mtry","splitrule","min_nodesize","ntrees")
  deltas_hyperparams.df$var_space <- "deltas"
  deltas_hyperparams.df$mtry <- deltas_mtry
  deltas_hyperparams.df$splitrule <- deltas_splitRule
  deltas_hyperparams.df$min_nodesize <- deltas_minNodeSize
  deltas_hyperparams.df$ntrees <- deltas_maxTree
  deltas_hyperparams.df
  #  var_space mtry  splitrule min_nodesize ntrees
  #1    deltas    2 extratrees            1   5000
  
  ### calib_deltas
  calib_deltas_out.ind <- which(colnames(calib_deltas_train.dat) == "biotic")
  calib_deltas_tune.list <- tuneModelHyperParams(train.df = calib_deltas_train.dat, 
                                           remove.ind = rm.ind,
                                           outcome.ind = calib_deltas_out.ind,
                                           fileName="maxTrees_out_calib_deltas.csv")
  calib_deltas_params <- calib_deltas_tune.list$tunedHyperparams
  calib_deltas_params
  #     mtry  splitrule min.node.size
  # 1    2 extratrees             1  
  calib_deltas_mtry <- calib_deltas_params$mtry
  calib_deltas_splitRule <- calib_deltas_params$splitrule
  calib_deltas_minNodeSize <- calib_deltas_params$min.node.size
  calib_deltas_maxTree <- calib_deltas_tune.list$numTrees
  
  # hyperparameter df
  calib_deltas_hyperparams.df <- as.data.frame(matrix(rep(NA,5),ncol=5))
  colnames(calib_deltas_hyperparams.df) <- c("var_space","mtry","splitrule","min_nodesize","ntrees")
  calib_deltas_hyperparams.df$var_space <- "calib_deltas"
  calib_deltas_hyperparams.df$mtry <- calib_deltas_mtry
  calib_deltas_hyperparams.df$splitrule <- calib_deltas_splitRule
  calib_deltas_hyperparams.df$min_nodesize <- calib_deltas_minNodeSize
  calib_deltas_hyperparams.df$ntrees <- calib_deltas_maxTree
  calib_deltas_hyperparams.df
  #     var_space mtry  splitrule min_nodesize ntrees
  #1 calib_deltas    2 extratrees            1   5000
  
  
  ### lurf
  lurf_out.ind <- which(colnames(lurf_train.dat) == "biotic")
  lurf_tune.list <- tuneModelHyperParams(train.df = lurf_train.dat, 
                                           remove.ind = rm.ind,
                                           outcome.ind = lurf_out.ind,
                                           fileName="maxTrees_out_lurf.csv")
  lurf_params <- lurf_tune.list$tunedHyperparams
  lurf_params
  #     mtry  splitrule min.node.size
  #  1    2 extratrees             1
  lurf_mtry <- lurf_params$mtry
  lurf_splitRule <- lurf_params$splitrule
  lurf_minNodeSize <- lurf_params$min.node.size
  lurf_maxTree <- lurf_tune.list$numTrees
  # hyperparameter df
  lurf_hyperparams.df <- as.data.frame(matrix(rep(NA,5),ncol=5))
  colnames(lurf_hyperparams.df) <- c("var_space","mtry","splitrule","min_nodesize","ntrees")
  lurf_hyperparams.df$var_space <- "lurf"
  lurf_hyperparams.df$mtry <- lurf_mtry
  lurf_hyperparams.df$splitrule <- lurf_splitRule
  lurf_hyperparams.df$min_nodesize <- lurf_minNodeSize
  lurf_hyperparams.df$ntrees <- lurf_maxTree
  lurf_hyperparams.df
  #  var_space mtry  splitrule min_nodesize ntrees
  #1      lurf    2 extratrees            1   5000
  
  
  ### lman
  lman_out.ind <- which(colnames(lman_train.dat) == "biotic")
  lman_tune.list <- tuneModelHyperParams(train.df = lman_train.dat, 
                                         remove.ind = rm.ind,
                                         outcome.ind = lman_out.ind,
                                         fileName="maxTrees_out_lman.csv")
  lman_params <- lman_tune.list$tunedHyperparams
  lman_params
  #     mtry  splitrule min.node.size
  #1    2 extratrees             1  
  lman_mtry <- lman_params$mtry
  lman_splitRule <- lman_params$splitrule
  lman_minNodeSize <- lman_params$min.node.size
  lman_maxTree <- lman_tune.list$numTrees
  # hyperparameter df
  lman_hyperparams.df <- as.data.frame(matrix(rep(NA,5),ncol=5))
  colnames(lman_hyperparams.df) <- c("var_space","mtry","splitrule","min_nodesize","ntrees")
  lman_hyperparams.df$var_space <- "lman"
  lman_hyperparams.df$mtry <- lman_mtry
  lman_hyperparams.df$splitrule <- lman_splitRule
  lman_hyperparams.df$min_nodesize <- lman_minNodeSize
  lman_hyperparams.df$ntrees <- lman_maxTree
  lman_hyperparams.df
  #  var_space mtry  splitrule min_nodesize ntrees
  #1      lman    2 extratrees            1   5000
  
  #####
  # create final hyperparameter df
  hyperparams.df <- as.data.frame(matrix(rep(NA,5*5),ncol=5))
  colnames(hyperparams.df) <- c("var_space","mtry","splitrule","min_nodesize","ntrees")
  hyperparams.df$var_space <- c("rr","deltas","calib_deltas","lurf","lman")
  hyperparams.df$mtry <- c(rr_mtry,deltas_mtry,calib_deltas_mtry,lurf_mtry,lman_mtry)
  hyperparams.df$splitrule <- c(rr_splitRule,deltas_splitRule,calib_deltas_splitRule,
                                lurf_splitRule,lman_splitRule)
  hyperparams.df$min_nodesize <- c(rr_minNodeSize,deltas_minNodeSize,calib_deltas_minNodeSize,
                                   lurf_minNodeSize,lMan_minNodeSize)
  hyperparams.df$ntrees <- c(rr_maxTree,deltas_maxTree,calib_deltas_maxTree,lurf_maxTree,
                             lman_maxTree)
  hyperparams.df
  # var_space mtry  splitrule min_nodesize ntrees
  # 1           rr    6 extratrees            1   5000
  # 2       deltas    2 extratrees            1   5000
  # 3 calib_deltas    2 extratrees            1   5000
  # 4         lurf    2 extratrees            1   5000
  # 5         lman    2 extratrees            5   5000     
  hyperparam_name <- paste("hyperparams_",i,sep="")
  hyperparam_name.vec <- c(hyperparam_name.vec,hyperparam_name)
  hyperparams.list[[i]] <- hyperparams.df
  
  
  ########
  # RF final fits
  
  ## change outcome col to factor (caret did this during tuning)
  # rr var RF
  rr_train.dat$biotic <- as.factor(rr_train.dat$biotic)
  rr_test.dat$biotic <- as.factor(rr_test.dat$biotic)
  # deltas features
  deltas_train.dat$biotic <- as.factor(deltas_train.dat$biotic)
  deltas_test.dat$biotic <- as.factor(deltas_test.dat$biotic)
  # calib_deltas
  calib_deltas_train.dat$biotic <- as.factor(calib_deltas_train.dat$biotic)
  calib_deltas_test.dat$biotic <- as.factor(calib_deltas_test.dat$biotic)
  # lurf
  lurf_train.dat$biotic <- as.factor(lurf_train.dat$biotic)
  lurf_test.dat$biotic <- as.factor(lurf_test.dat$biotic)
  # lman
  lman_train.dat$biotic <- as.factor(lman_train.dat$biotic)
  lman_test.dat$biotic <- as.factor(lman_test.dat$biotic)
  
  ###### final models
  ## rr RF
  rrVar.rf <- ranger(biotic~., rr_train.dat[,-rm.ind], 
                       keep.inbag = TRUE,
                       num.trees=hyperparams.df[1,]$ntrees, 
                       mtry=hyperparams.df[1,]$mtry, 
                       importance="permutation", 
                       splitrule = hyperparams.df[1,]$splitrule,
                       min.node.size=hyperparams.df[1,]$min_nodesize,
                       class.weights = as.numeric(c(1/table(rr_train.dat$biotic))),
                       scale.permutation.importance = T,
                       local.importance = T, num.threads=4)
  rr_train_acc <- 1-rrVar.rf$prediction.error
  # [1] 0.8857143
  rr_train.cm <- rrVar.rf$confusion.matrix
  #             predicted
  # true      abiotic biotic
  # abiotic      83      6
  # biotic       10     41
  rr_train.pred <- rrVar.rf$predictions
  rr_train.dat$pred <- rr_train.pred
  #head(rr_train.dat)
  rr_test.pred<-predict(rrVar.rf,data=rr_test.dat[,-rm.ind])
  rr_test.dat$pred <- rr_test.pred$predictions
  #head(rr_test.dat)
  rr_test.cm <- table(rr_test.dat$biotic, rr_test.pred$predictions)
  # abiotic biotic
  #abiotic      20      2
  #biotic        1     11
  rr_test_acc <- sum(rr_test.dat$biotic==rr_test.pred$predictions)/length(rr_test.dat$biotic)
  rr_test_acc
  # [1] 0.9117647
  
  rr_rf.list[[i]] <- rrVar.rf
  rr_train_acc.list[[i]] <- rr_train_acc
  rr_train.list[[i]] <- rr_train.dat
  rr_train_cm.list[[i]] <- rr_train.cm
  rr_test_acc.list[[i]] <-rr_test_acc
  rr_test_cm.list[[i]] <- rr_test.cm
  rr_test.list[[i]] <- rr_test.dat
  
  # deltas RF
  deltas.rf <- ranger(biotic~., deltas_train.dat[,-rm.ind], 
                    keep.inbag = TRUE,
                    num.trees=hyperparams.df[2,]$ntrees, 
                    mtry=hyperparams.df[2,]$mtry, 
                    importance="permutation", 
                    splitrule = hyperparams.df[2,]$splitrule,
                    min.node.size=hyperparams.df[2,]$min_nodesize,
                    class.weights = as.numeric(c(1/table(deltas_train.dat$biotic))),
                    scale.permutation.importance = T,
                    local.importance = T, num.threads=4)
  deltas_train.cm <- deltas.rf$confusion.matrix
  #             predicted
  # true      abiotic biotic
  # abiotic      71     18
  # biotic       23     28
  deltas_train_acc <- 1-deltas.rf$prediction.error
  # [1] 0.7071429
  deltas_train.pred <- deltas.rf$predictions
  deltas_train.dat$pred <- deltas_train.pred
  deltas_test.pred<-predict(deltas.rf,data=deltas_test.dat[,-rm.ind])
  deltas_test.dat$pred <- deltas_test.pred$predictions
  deltas_test.cm <- table(deltas_test.dat$biotic, deltas_test.pred$predictions)
  #           abiotic biotic
  # abiotic      16      6
  # biotic        5      7
  deltas_test_acc <- sum(deltas_test.dat$biotic==deltas_test.pred$predictions)/length(deltas_test.dat$biotic)
  # [1] 0.6764706
  deltas_rf.list[[i]] <- deltas.rf
  deltas_train.list[[i]] <- deltas_train.dat
  deltas_train_cm.list[[i]] <- deltas_train.cm
  deltas_train_acc.list[[i]] <- deltas_train_acc
  deltas_test_cm.list[[i]] <- deltas_test.cm
  deltas_test.list[[i]] <- deltas_test.dat
  deltas_test_acc.list[[i]] <- deltas_test_acc
  
  # calib_deltas RF
  calib_deltas.rf <- ranger(biotic~., calib_deltas_train.dat[,-rm.ind], 
                   keep.inbag = TRUE,
                   num.trees=hyperparams.df[3,]$ntrees, 
                   mtry=hyperparams.df[3,]$mtry, 
                   importance="permutation", 
                   splitrule = hyperparams.df[3,]$splitrule,
                   min.node.size=hyperparams.df[3,]$min_nodesize,
                   class.weights = as.numeric(c(1/table(calib_deltas_train.dat$biotic))),
                   scale.permutation.importance = T,
                   local.importance = T, num.threads=4)
  calib_deltas_train.cm <- calib_deltas.rf$confusion.matrix
  #              predicted
  # true      abiotic biotic
  # abiotic      73     16
  # biotic       26     25
  calib_deltas_train_acc <- 1 - calib_deltas.rf$prediction.error
  # [1] 0.7
  calib_deltas_train.pred <- calib_deltas.rf$predictions
  calib_deltas_train.dat$pred <- calib_deltas_train.pred
  calib_deltas_test.pred<-predict(calib_deltas.rf,data=calib_deltas_test.dat[,-rm.ind])
  calib_deltas_test.dat$pred <- calib_deltas_test.pred$predictions
  calib_deltas_test.cm <- table(calib_deltas_test.dat$biotic, calib_deltas_test.pred$predictions)
  #         abiotic biotic
  # abiotic      20      2
  # biotic        3      9
  calib_deltas_test_acc <- sum(calib_deltas_test.dat$biotic==calib_deltas_test.pred$predictions)/length(calib_deltas_test.dat$biotic)
  #[1] 0.8529412
  calib_deltas_rf.list[[i]] <- calib_deltas.rf
  calib_deltas_train.list[[i]] <- calib_deltas_train.dat
  calib_deltas_train_cm.list[[i]] <- calib_deltas_train.cm
  calib_deltas_train_acc.list[[i]] <- calib_deltas_train_acc
  calib_deltas_test_cm.list[[i]] <- calib_deltas_test.cm
  calib_deltas_test.list[[i]] <- calib_deltas_test.dat
  calib_deltas_test_acc.list[[i]] <- calib_deltas_test_acc
  
  # lurf RF
  lurf.rf <- ranger(biotic~., lurf_train.dat[,-rm.ind], 
                            keep.inbag = TRUE,
                            num.trees=hyperparams.df[4,]$ntrees, 
                            mtry=hyperparams.df[4,]$mtry, 
                            importance="permutation", 
                            splitrule = hyperparams.df[4,]$splitrule,
                            min.node.size=hyperparams.df[4,]$min_nodesize,
                            class.weights = as.numeric(c(1/table(lurf_train.dat$biotic))),
                            scale.permutation.importance = T,
                            local.importance = T, num.threads=4)
  lurf_train.cm <- lurf.rf$confusion.matrix
  #              predicted
  # true  abiotic biotic
  # abiotic      84      5
  # biotic        9     42    
  lurf_train_acc <- 1-lurf.rf$prediction.error
  # [1] 0.9
  lurf_train.pred <- lurf.rf$predictions
  lurf_train.dat$pred <- lurf_train.pred
  lurf_test.pred<-predict(lurf.rf,data=lurf_test.dat[,-rm.ind])
  lurf_test.dat$pred <- lurf_test.pred$predictions
  lurf_test.cm <- table(lurf_test.dat$biotic, lurf_test.pred$predictions)
  #         abiotic biotic
  # abiotic      20      2
  # biotic        2     10
  lurf_test_acc <- sum(lurf_test.dat$biotic==lurf_test.pred$predictions)/length(lurf_test.dat$biotic)
  #[1] 0.8823529
  lurf_rf.list[[i]] <- lurf.rf
  lurf_train.list[[i]] <- lurf_train.dat
  lurf_train_cm.list[[i]] <- lurf_train.cm
  lurf_train_acc.list[[i]] <- lurf_train_acc
  lurf_test_cm.list[[i]] <- lurf_test.cm
  lurf_test.list[[i]] <- lurf_test.dat
  lurf_test_acc.list[[i]] <- lurf_test_acc
  
  # lman RF
  lman.rf <- ranger(biotic~., lman_train.dat[,-rm.ind], 
                    keep.inbag = TRUE,
                    num.trees=hyperparams.df[5,]$ntrees, 
                    mtry=hyperparams.df[5,]$mtry, 
                    importance="permutation", 
                    splitrule = hyperparams.df[5,]$splitrule,
                    min.node.size=hyperparams.df[5,]$min_nodesize,
                    class.weights = as.numeric(c(1/table(lman_train.dat$biotic))),
                    scale.permutation.importance = T,
                    local.importance = T, num.threads=4)
  lman_train.cm <- lman.rf$confusion.matrix
  #              predicted
  # true    abiotic biotic
  # abiotic      77     12
  # biotic        7     44  
  lman_train_acc <- 1-lman.rf$prediction.error
  # [1] 0.8642857
  lman_train.pred <- lman.rf$predictions
  lman_train.dat$pred <- lman_train.pred
  lman_test.pred<-predict(lman.rf,data=lman_test.dat[,-rm.ind])
  lman_test.dat$pred <- lman_test.pred$predictions
  lman_test.cm <- table(lman_test.dat$biotic, lman_test.pred$predictions)
  #           abiotic biotic
  # abiotic      16      6
  # biotic        0     12
  lman_test_acc <- sum(lman_test.dat$biotic==lman_test.pred$predictions)/length(lman_test.dat$biotic)
  # [1] 0.8235294
  lman_rf.list[[i]] <- lman.rf
  lman_train.list[[i]] <- lman_train.dat
  lman_train_cm.list[[i]] <- lman_train.cm
  lman_train_acc.list[[i]] <- lman_train_acc
  lman_test_cm.list[[i]] <- lman_test.cm
  lman_test.list[[i]] <- lman_test.dat
  lman_test_acc.list[[i]] <- lman_test_acc
  
  
}
names(hyperparams.list) <- hyperparam_name.vec
#####################################
# 30. Compile list of tuned hyperparameters
#####################################
hyperparams.list
# $hyperparams_1
# var_space mtry  splitrule min_nodesize ntrees
# 1           rr    6 extratrees            1   5000
# 2       deltas    2 extratrees            1   5000
# 3 calib_deltas    2 extratrees            1   5000
# 4         lurf    2 extratrees            1   5000
# 5         lman    2 extratrees            5   5000
# 
# $hyperparams_2
# var_space mtry  splitrule min_nodesize ntrees
# 1           rr    6 extratrees            1   5000
# 2       deltas    2 extratrees            1   5000
# 3 calib_deltas    2 extratrees            1   5000
# 4         lurf    2 extratrees            1   5000
# 5         lman    2 extratrees            5   5000
# 
# $hyperparams_3
# var_space mtry  splitrule min_nodesize ntrees
# 1           rr   14 extratrees            2   5000
# 2       deltas    2 extratrees            1   5000
# 3 calib_deltas    2 extratrees            1   5000
# 4         lurf    2 extratrees            1   5000
# 5         lman    2 extratrees            5   5000
# 
# $hyperparams_4
# var_space mtry  splitrule min_nodesize ntrees
# 1           rr    6 extratrees           13   5000
# 2       deltas    2 extratrees            1   5000
# 3 calib_deltas    2 extratrees            1   5000
# 4         lurf    2 extratrees            1   5000
# 5         lman    2 extratrees            5   5000
# 
# $hyperparams_5
# var_space mtry  splitrule min_nodesize ntrees
# 1           rr   11       gini            1   5000
# 2       deltas    2 extratrees            1   5000
# 3 calib_deltas    2 extratrees            1   5000
# 4         lurf    2 extratrees            1   5000
# 5         lman    2 extratrees            5   5000

###########################
# 31. write RR results to file
############################
saveRDS(rr_rf.list[[1]],"./replicates_data/rr_rf_run1.rds")
saveRDS(rr_rf.list[[2]],"./replicates_data/rr_rf_run2.rds")
saveRDS(rr_rf.list[[3]],"./replicates_data/rr_rf_run3.rds")
saveRDS(rr_rf.list[[4]],"./replicates_data/rr_rf_run4.rds")
saveRDS(rr_rf.list[[5]],"./replicates_data/rr_rf_run5.rds")

sort(rr_rf.list[[1]]$variable.importance,decreasing=TRUE)[1:15]
# max_kl_shift  fluctanal_prop_r1   avg_rR45CO244CO2 localsimple_taures 
#   0.03794875         0.03315332         0.02174076         0.01937002 
# time_level_shift        diff2_acf10       diff2x_pacf5         diff2_acf1 
#       0.01871478         0.01555481         0.01551843         0.01425937 
# avg_d13C12C      max_var_shift         avg_pkArea    max_level_shift 
# 0.01408241         0.01318299         0.01243586         0.01219397 
#    x_pacf5              trend          sd_pkArea 
# 0.01196687         0.01185559         0.01119251 
sort(rr_rf.list[[2]]$variable.importance,decreasing=TRUE)[1:15]
# fluctanal_prop_r1       max_kl_shift localsimple_taures   time_level_shift 
# 0.04383326         0.03819295         0.02707384         0.02097244 
# avg_rR45CO244CO2       diff2x_pacf5        avg_d13C12C        diff2_acf10 
# 0.01960992         0.01573697         0.01530754         0.01391119 
# diff2_acf1       nonlinearity          sd_pkArea      max_var_shift 
# 0.01295950         0.01271060         0.01241918         0.01217357 
# max_level_shift            x_pacf5         avg_pkArea 
# 0.01144035         0.01137883         0.01121902 
sort(rr_rf.list[[3]]$variable.importance,decreasing=TRUE)[1:15]
# fluctanal_prop_r1       max_kl_shift   avg_rR45CO244CO2 localsimple_taures 
# 0.05375062         0.05262707         0.02381115         0.02291657 
# diff2x_pacf5   time_level_shift        avg_d13C12C        diff2_acf10 
# 0.01905269         0.01892637         0.01841384         0.01540968 
# diff2_acf1    max_level_shift            x_pacf5      max_var_shift 
# 0.01537497         0.01321435         0.01200842         0.01126455 
# sd_pkArea            entropy              trend 
# 0.01125103         0.01106776         0.01106538
sort(rr_rf.list[[4]]$variable.importance,decreasing=TRUE)[1:15]
#  max_kl_shift  fluctanal_prop_r1   avg_rR45CO244CO2 localsimple_taures 
# 0.030876920        0.030117278        0.017414175        0.016697537 
# time_level_shift       diff2x_pacf5        diff2_acf10        avg_d13C12C 
# 0.014308232        0.013266869        0.012391273        0.012208873 
# diff2_acf1            x_pacf5      max_var_shift    max_level_shift 
# 0.011141916        0.010161445        0.009856915        0.008745713 
# nonlinearity           arch_acf          sd_pkArea 
# 0.008640326        0.008245577        0.007932291 
sort(rr_rf.list[[5]]$variable.importance,decreasing=TRUE)[1:15]
# max_kl_shift  fluctanal_prop_r1   avg_rR45CO244CO2   time_level_shift 
# 0.075063427        0.044509707        0.024452678        0.023279243 
# max_var_shift       diff2x_pacf5    max_level_shift        avg_d13C12C 
# 0.014624872        0.014537053        0.014383762        0.012272825 
# e_acf1           arch_acf   avg_rR46CO244CO2    motiftwo_entro3 
# 0.011869769        0.009382334        0.008140659        0.008133746 
# localsimple_taures         avg_pkArea       nonlinearity 
# 0.008087153        0.007720640        0.007048020


rr_train_acc.list
# [[1]]
# 0.8857143
# [[2]]
# 0.9
# [[3]]
# 0.9
# [[4]]
# 0.8714286
# [[5]]
# 0.8785714

rr_train_cm.list 
# [[1]]
#             predicted
# true      abiotic biotic
# abiotic      83      6
# biotic       10     41
# [[2]]
#              predicted
# true      abiotic biotic
# abiotic      83      6
# biotic        8     43
# [[3]]
# rr_train    predicted
# true      abiotic biotic
# abiotic      83      6
# biotic        8     43
# [[4]]
# rr_train     predicted
# true      abiotic biotic
# abiotic      77     12
# biotic        6     45
# [[5]]
# rr_train    predicted
# true      abiotic biotic
# abiotic      83      6
# biotic       11     40

write.table(rr_train.list[[1]],"./replicates_data/rr_train_run1.csv",row.names=F,
            quote=F,sep=",")
write.table(rr_train.list[[2]],"./replicates_data/rr_train_run2.csv",row.names=F,
            quote=F,sep=",")
write.table(rr_train.list[[3]],"./replicates_data/rr_train_run3.csv",row.names=F,
            quote=F,sep=",")
write.table(rr_train.list[[4]],"./replicates_data/rr_train_run4.csv",row.names=F,
            quote=F,sep=",")
write.table(rr_train.list[[5]],"./replicates_data/rr_train_run5.csv",row.names=F,
            quote=F,sep=",")

rr_test_acc.list 
# [[1]]
# [1] 0.9117647
# [[2]]
# [1] 0.8823529
# [[3]]
# [1] 0.8529412
# [[4]]
# [1] 0.8823529
# [[5]]
# [1] 0.9117647

rr_test_cm.list
# [[1]]
# rr_test abiotic biotic
# abiotic      20      2
# biotic        1     11
# [[2]]
# rr_test abiotic biotic
# abiotic      20      2
# biotic        2     10
# [[3]]
# rr_test abiotic biotic
# abiotic      18      4
# biotic        1     11
# [[4]]
# rr_test abiotic biotic
# abiotic      21      1
# biotic        3      9
# [[5]]
# rr_test abiotic biotic
# abiotic      22      0
# biotic        3      9

write.table(rr_test.list[[1]], "./replicates_data/rr_test_run1.csv", row.names=F,quote=F,sep=",")
write.table(rr_test.list[[2]], "./replicates_data/rr_test_run2.csv", row.names=F,quote=F,sep=",")
write.table(rr_test.list[[3]], "./replicates_data/rr_test_run3.csv", row.names=F,quote=F,sep=",")
write.table(rr_test.list[[4]], "./replicates_data/rr_test_run4.csv", row.names=F,quote=F,sep=",")
write.table(rr_test.list[[5]], "./replicates_data/rr_test_run5.csv", row.names=F,quote=F,sep=",")


#######################
# 32. write deltas results to file
###########################
saveRDS(deltas_rf.list[[1]],"./replicates_data/deltas_rf_run1.rds")
saveRDS(deltas_rf.list[[2]],"./replicates_data/deltas_rf_run2.rds")
saveRDS(deltas_rf.list[[3]],"./replicates_data/deltas_rf_run3.rds")
saveRDS(deltas_rf.list[[4]],"./replicates_data/deltas_rf_run4.rds")
saveRDS(deltas_rf.list[[5]],"./replicates_data/deltas_rf_run5.rds")

sort(deltas_rf.list[[1]]$variable.importance,decreasing=TRUE)
# avg_d18O16O avg_d13C12C 
# 0.08883733  0.08345102
sort(deltas_rf.list[[2]]$variable.importance,decreasing=TRUE)
# avg_d13C12C avg_d18O16O 
# 0.11082103  0.08825764 
sort(deltas_rf.list[[3]]$variable.importance,decreasing=TRUE)
# avg_d18O16O avg_d13C12C 
# 0.09078752  0.08440575 
sort(deltas_rf.list[[4]]$variable.importance,decreasing=TRUE)
# avg_d13C12C avg_d18O16O 
# 0.09293935  0.07736480
sort(deltas_rf.list[[5]]$variable.importance,decreasing=TRUE)
# avg_d13C12C avg_d18O16O 
# 0.08533852  0.08503329 

deltas_train_acc.list
# [[1]]
# [1] 0.7071429
# [[2]]
# [1] 0.7214286
# [[3]]
# [1] 0.7071429
# [[4]]
# [1] 0.7285714
# [[5]]
# [1] 0.7142857

deltas_train_cm.list
# [[1]]
# deltas_train  predicted
# true      abiotic biotic
# abiotic      71     18
# biotic       23     28
# [[2]]
# deltas_train  predicted
# true      abiotic biotic
# abiotic      73     16
# biotic       23     28
# [[3]]
# deltas_train  predicted
# true      abiotic biotic
# abiotic      71     18
# biotic       23     28
# [[4]]
# deltas_train  predicted
# true      abiotic biotic
# abiotic      73     16
# biotic       22     29
# [[5]]
# deltas_train  predicted
# true      abiotic biotic
# abiotic      73     16
# biotic       24     27

write.table(deltas_train.list[[1]],"./replicates_data/deltas_train_run1.csv",quote=F,
            row.names=F,sep=",")
write.table(deltas_train.list[[2]],"./replicates_data/deltas_train_run2.csv",quote=F,
            row.names=F,sep=",")
write.table(deltas_train.list[[3]],"./replicates_data/deltas_train_run3.csv",quote=F,
            row.names=F,sep=",")
write.table(deltas_train.list[[4]],"./replicates_data/deltas_train_run4.csv",quote=F,
            row.names=F,sep=",")
write.table(deltas_train.list[[5]],"./replicates_data/deltas_train_run5.csv",quote=F,
            row.names=F,sep=",")

deltas_test_acc.list
# [[1]]
# [1] 0.6764706
# [[2]]
# [1] 0.7941176
# [[3]]
# [1] 0.7352941
# [[4]]
# [1] 0.7058824
# [[5]]
# [1] 0.7941176

deltas_test_cm.list
# [[1]]
# deltas_test abiotic biotic
# abiotic      16      6
# biotic        5      7
# [[2]]
# deltas_test abiotic biotic
# abiotic      19      3
# biotic        4      8
# [[3]]
# deltas_test abiotic biotic
# abiotic      16      6
# biotic        3      9
# [[4]]
# deltas_test abiotic biotic
# abiotic      18      4
# biotic        6      6
# [[5]]
# deltas_test abiotic biotic
# abiotic      20      2
# biotic        5      7

write.table(deltas_test.list[[1]],"./replicates_data/deltas_test_run1.csv",
            quote=F,row.names=F,sep=",")
write.table(deltas_test.list[[2]],"./replicates_data/deltas_test_run2.csv",
            quote=F,row.names=F,sep=",")
write.table(deltas_test.list[[3]],"./replicates_data/deltas_test_run3.csv",
            quote=F,row.names=F,sep=",")
write.table(deltas_test.list[[4]],"./replicates_data/deltas_test_run4.csv",
            quote=F,row.names=F,sep=",")
write.table(deltas_test.list[[5]],"./replicates_data/deltas_test_run5.csv",
            quote=F,row.names=F,sep=",")


################################
# 33. write calib_deltas results to file
##################################
saveRDS(calib_deltas_rf.list[[1]],"./replicates_data/calib_deltas_rf_run1.rds")
saveRDS(calib_deltas_rf.list[[2]],"./replicates_data/calib_deltas_rf_run2.rds")
saveRDS(calib_deltas_rf.list[[3]],"./replicates_data/calib_deltas_rf_run3.rds")
saveRDS(calib_deltas_rf.list[[4]],"./replicates_data/calib_deltas_rf_run4.rds")
saveRDS(calib_deltas_rf.list[[5]],"./replicates_data/calib_deltas_rf_run5.rds")

sort(calib_deltas_rf.list[[1]]$variable.importance,decreasing=TRUE)
# avg_d13C12C avg_calib_d18O16O 
# 0.08587372        0.08072759 
sort(calib_deltas_rf.list[[2]]$variable.importance,decreasing=TRUE)
# avg_d13C12C avg_calib_d18O16O 
# 0.11857866        0.09170428
sort(calib_deltas_rf.list[[3]]$variable.importance,decreasing=TRUE)
# avg_d13C12C avg_calib_d18O16O 
# 0.09606001        0.09361905
sort(calib_deltas_rf.list[[4]]$variable.importance,decreasing=TRUE)
# avg_d13C12C avg_calib_d18O16O 
# 0.09681813        0.07961831 
sort(calib_deltas_rf.list[[5]]$variable.importance,decreasing=TRUE)
# avg_d13C12C avg_calib_d18O16O 
# 0.08908048        0.08805802 

write.table(calib_deltas_train.list[[1]],"./replicates_data/calib_deltas_train_run1.csv",
            quote=F,row.names=F,sep=",")
write.table(calib_deltas_train.list[[2]],"./replicates_data/calib_deltas_train_run2.csv",
            quote=F,row.names=F,sep=",")
write.table(calib_deltas_train.list[[3]],"./replicates_data/calib_deltas_train_run3.csv",
            quote=F,row.names=F,sep=",")
write.table(calib_deltas_train.list[[4]],"./replicates_data/calib_deltas_train_run4.csv",
            quote=F,row.names=F,sep=",")
write.table(calib_deltas_train.list[[5]],"./replicates_data/calib_deltas_train_run5.csv",
            quote=F,row.names=F,sep=",")

calib_deltas_train_acc.list
# [[1]]
# [1] 0.7
# [[2]]
# [1] 0.7428571
# [[3]]
# [1] 0.7285714
# [[4]]
# [1] 0.7071429
# [[5]]
# [1] 0.6928571

calib_deltas_train_cm.list
# [[1]]
# calibDeltas_train predicted
# true      abiotic biotic
# abiotic      73     16
# biotic       26     25
# [[2]]
# calibDeltas_train predicted
# true      abiotic biotic
# abiotic      75     14
# biotic       22     29
# [[3]]
# calibDeltas_train predicted
# true      abiotic biotic
# abiotic      76     13
# biotic       25     26
# [[4]]
# calibDeltas_train predicted
# true      abiotic biotic
# abiotic      70     19
# biotic       22     29
# [[5]]
# calibDeltas_train predicted
# true      abiotic biotic
# abiotic      70     19
# biotic       24     27

calib_deltas_test_acc.list
# [[1]]
# [1] 0.8529412
# [[2]]
# [1] 0.8529412
# [[3]]
# [1] 0.7941176
# [[4]]
# [1] 0.7647059
# [[5]]
# [1] 0.7647059

calib_deltas_test_cm.list 
# [[1]]
# calibDeltas_test abiotic biotic
# abiotic      20      2
# biotic        3      9
# [[2]]
# calibDeltas_test abiotic biotic
# abiotic      20      2
# biotic        3      9
# [[3]]
# calibDeltas_test abiotic biotic
# abiotic      17      5
# biotic        2     10
# [[4]]
# calibDeltas_test abiotic biotic
# abiotic      19      3
# biotic        5      7
# [[5]]
# calibDeltas_test abiotic biotic
# abiotic      20      2
# biotic        6      6

write.table(calib_deltas_test.list[[1]],"./replicates_data/calib_deltas_test_run1.csv",
            quote=F,row.names=F,sep=",")
write.table(calib_deltas_test.list[[2]],"./replicates_data/calib_deltas_test_run2.csv",
            quote=F,row.names=F,sep=",")
write.table(calib_deltas_test.list[[3]],"./replicates_data/calib_deltas_test_run3.csv",
            quote=F,row.names=F,sep=",")
write.table(calib_deltas_test.list[[4]],"./replicates_data/calib_deltas_test_run4.csv",
            quote=F,row.names=F,sep=",")
write.table(calib_deltas_test.list[[5]],"./replicates_data/calib_deltas_test_run5.csv",
            quote=F,row.names=F,sep=",")


###############################
# 34. write LURF results to file
##############################
saveRDS(lurf_rf.list[[1]],"./replicates_data/lurf_rf_run1.rds")
saveRDS(lurf_rf.list[[2]],"./replicates_data/lurf_rf_run2.rds")
saveRDS(lurf_rf.list[[3]],"./replicates_data/lurf_rf_run3.rds")
saveRDS(lurf_rf.list[[4]],"./replicates_data/lurf_rf_run4.rds")
saveRDS(lurf_rf.list[[5]],"./replicates_data/lurf_rf_run5.rds")

sort(lurf_rf.list[[1]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1        diff2_acf1  avg_rR45CO244CO2     time_kl_shift  avg_rR46CO244CO2 
# 0.09688680        0.08779211        0.06313173        0.03979519        0.03225927 
# walker_propcross        sd_d18O13C 
#      0.02638977        0.01513562 

sort(lurf_rf.list[[2]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1 diff2_acf1  avg_rR45CO244CO2   motiftwo_entro3   avg_R45CO244CO2 
# 0.08426889        0.07605824        0.04290012        0.03699305        0.03306043 
# avg_d45CO244CO2        sd_R13C12C        sd_d13C12C  avg_rd45CO244CO2     time_kl_shift 
#      0.03276559        0.03240622        0.03214771        0.03193968        0.01962006 
# sd_d18O13C 
# 0.01432101

sort(lurf_rf.list[[3]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1        diff2_acf1  avg_rR45CO244CO2     time_kl_shift        sd_d18O13C 
#        0.13911573        0.13030811        0.07365417        0.06685651        0.03053530 

sort(lurf_rf.list[[4]]$variable.importance,decreasing=TRUE)
#     diff2_acf1  avg_R45CO244CO2 avg_rR45CO244CO2 
#     0.19466606       0.11544615       0.08780765

sort(lurf_rf.list[[5]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1        diff2_acf1  avg_rR45CO244CO2  avg_rR46CO244CO2     time_kl_shift 
#        0.12417439        0.09238166        0.06373755        0.04184923        0.03900339 
# sd_d18O13C 
# 0.02092299


lurf_train_acc.list
# [[1]]
# [1] 0.9
# [[2]]
# [1] 0.9285714
# [[3]]
# [1] 0.9285714
# [[4]]
# [1] 0.8571429
# [[5]]
# [1] 0.9071429

lurf_train_cm.list
# [[1]]
# lurf_train  predicted
# true      abiotic biotic
# abiotic      84      5
# biotic        9     42
# [[2]]
# lurf_train  predicted
# true      abiotic biotic
# abiotic      85      4
# biotic        6     45
# [[3]]
# lurf_train  predicted
# true      abiotic biotic
# abiotic      83      6
# biotic        4     47
# [[4]]
# lurf_train  predicted
# true      abiotic biotic
# abiotic      80      9
# biotic       11     40
# [[5]]
# lurf_train  predicted
# true      abiotic biotic
# abiotic      84      5
# biotic        8     43

write.table(lurf_train.list[[1]],"./replicates_data/lurf_train_run1.csv",
            quote=F,row.names=F,sep=",")
write.table(lurf_train.list[[2]],"./replicates_data/lurf_train_run2.csv",
            quote=F,row.names=F,sep=",")
write.table(lurf_train.list[[3]],"./replicates_data/lurf_train_run3.csv",
            quote=F,row.names=F,sep=",")
write.table(lurf_train.list[[4]],"./replicates_data/lurf_train_run4.csv",
            quote=F,row.names=F,sep=",")
write.table(lurf_train.list[[5]],"./replicates_data/lurf_train_run5.csv",
            quote=F,row.names=F,sep=",")

lurf_test_acc.list
# [[1]]
# [1] 0.8823529
# [[2]]
# [1] 0.8529412
# [[3]]
# [1] 0.8235294
# [[4]]
# [1] 0.8823529
# [[5]]
# [1] 0.9117647

lurf_test_cm.list
# [[1]]
# lurf_test abiotic biotic
# abiotic      20      2
# biotic        2     10
# [[2]]
# lurf_test abiotic biotic
# abiotic      20      2
# biotic        3      9
# [[3]]
# lurf_test abiotic biotic
# abiotic      19      3
# biotic        3      9
# [[4]]
# lurf_test abiotic biotic
# abiotic      21      1
# biotic        3      9
# [[5]]
# lurf_test abiotic biotic
# abiotic      22      0
# biotic        3      9

write.table(lurf_test.list[[1]] ,"./replicates_data/lurf_test_run1.csv",
            quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[2]] ,"./replicates_data/lurf_test_run2.csv",
            quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[3]] ,"./replicates_data/lurf_test_run3.csv",
            quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[4]] ,"./replicates_data/lurf_test_run4.csv",
            quote=F,row.names=F,sep=",")
write.table(lurf_test.list[[5]] ,"./replicates_data/lurf_test_run5.csv",
            quote=F,row.names=F,sep=",")
 

################################
# 35. write Lman results to file
#################################
saveRDS(lman_rf.list[[1]],"./replicates_data/lman_rf_run1.rds")
saveRDS(lman_rf.list[[2]],"./replicates_data/lman_rf_run2.rds")
saveRDS(lman_rf.list[[3]],"./replicates_data/lman_rf_run3.rds")
saveRDS(lman_rf.list[[4]],"./replicates_data/lman_rf_run4.rds")
saveRDS(lman_rf.list[[5]],"./replicates_data/lman_rf_run5.rds")

sort(lman_rf.list[[1]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1  avg_rR45CO244CO2   motiftwo_entro3  avg_rR46CO244CO2   avg_R45CO244CO2 
# 0.11245442        0.07098337        0.05442225        0.04271529        0.03457979 
# sd_R13C12C  walker_propcross 
# 0.03341098        0.03184988 

sort(lman_rf.list[[2]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1   motiftwo_entro3       avg_R13C12C       avg_d13C12C   avg_R45CO244CO2 
# 0.13308751        0.06218000        0.05029790        0.04831331        0.04486878 

sort(lman_rf.list[[3]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1 outlierinclude_mdrmd     avg_rR45CO244CO2      motiftwo_entro3 
# 0.07129895           0.03909344           0.03699646           0.03544321 
# time_kl_shift      avg_R45CO244CO2           sd_R13C12C              sampenc 
# 0.02530961           0.02405741           0.02107628           0.01935115 
# sampen_first          avg_d18O13C           sd_d18O13C 
# 0.01803013           0.01533116           0.01304541

sort(lman_rf.list[[4]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1  avg_rR45CO244CO2   motiftwo_entro3  avg_rd45CO244CO2   avg_d45CO244CO2 
# 0.09250261        0.05676524        0.04551619        0.03167181        0.03154880 
# avg_R45CO244CO2  avg_rR46CO244CO2        sd_d18O13C 
# 0.03108907        0.03071319        0.01765150

sort(lman_rf.list[[5]]$variable.importance,decreasing=TRUE)
# fluctanal_prop_r1   motiftwo_entro3  avg_rR45CO244CO2   avg_R45CO244CO2  avg_rR46CO244CO2 
# 0.12040580        0.05838576        0.05066457        0.03936839        0.03917270 
# time_kl_shift        sd_d18O13C 
# 0.03588448        0.01456150 

write.table(lman_train.list[[1]],"./replicates_data/lman_train_run1.csv",
            quote=F,row.names=F,sep=",")
write.table(lman_train.list[[2]],"./replicates_data/lman_train_run2.csv",
            quote=F,row.names=F,sep=",")
write.table(lman_train.list[[3]],"./replicates_data/lman_train_run3.csv",
            quote=F,row.names=F,sep=",")
write.table(lman_train.list[[4]],"./replicates_data/lman_train_run4.csv",
            quote=F,row.names=F,sep=",")
write.table(lman_train.list[[5]],"./replicates_data/lman_train_run5.csv",
            quote=F,row.names=F,sep=",")

lman_train_acc.list
#[[1]]
# [1] 0.8642857
# [[2]]
# [1] 0.7785714
# [[3]]
# [1] 0.8857143
# [[4]]
# [1] 0.8428571
# [[5]]
# [1] 0.8642857

lman_train_cm.list
# [[1]]
# lman_train  predicted
# true      abiotic biotic
# abiotic      77     12
# biotic        7     44
# [[2]]
# lman_train  predicted
# true      abiotic biotic
# abiotic      72     17
# biotic       14     37
# [[3]]
# lman_train  predicted
# true      abiotic biotic
# abiotic      80      9
# biotic        7     44
# [[4]]
# lman_train  predicted
# true      abiotic biotic
# abiotic      78     11
# biotic       11     40
# [[5]]
# lman_train  predicted
# true      abiotic biotic
# abiotic      78     11
# biotic        8     43

lman_test_acc.list
# [[1]]
# [1] 0.8235294
# [[2]]
# [1] 0.7941176
# [[3]]
# [1] 0.8235294
# [[4]]
# [1] 0.9117647
# [[5]]
# [1] 0.8235294

lman_test_cm.list
# [[1]]

# [[2]]
# lman_test abiotic biotic
# abiotic      18      4
# biotic        3      9
# [[3]]
# lman_test abiotic biotic
# abiotic      18      4
# biotic        2     10
# [[4]]
# lman_test abiotic biotic
# abiotic      21      1
# biotic        2     10
# [[5]]
# lman_test abiotic biotic
# abiotic      21      1
# biotic        5      7

write.table(lman_test.list[[1]],"./replicates_data/lman_test_run1.csv",
            quote=F,row.names=F,sep=",")
write.table(lman_test.list[[2]],"./replicates_data/lman_test_run2.csv",
            quote=F,row.names=F,sep=",")
write.table(lman_test.list[[3]],"./replicates_data/lman_test_run3.csv",
            quote=F,row.names=F,sep=",")
write.table(lman_test.list[[4]],"./replicates_data/lman_test_run4.csv",
            quote=F,row.names=F,sep=",")
write.table(lman_test.list[[5]],"./replicates_data/lman_test_run5.csv",
            quote=F,row.names=F,sep=",")


###################################################
# 36. Find the mean and median of train/test accuracies 
###################################################
# LURF 
lurf_train_acc.vec <- c(87.9,90.0,92.9,92.9,85.7,90.7)
lurf_test_acc.vec <- c(88.2,88.2,85.3,82.4,88.2,91.2)
mean(lurf_train_acc.vec)
# [1] 90.01667
median(lurf_train_acc.vec)
# [1] 90.35
mean(lurf_test_acc.vec)
# [1] 87.25
median(lurf_test_acc.vec)
# [1] 88.2

# RR
rr_train_acc.vec <- c(86.4,88.6,88.2,90.0,87.1,87.9)
rr_test_acc.vec <- c(91.2,91.2,88.2,85.3,88.2,91.2)
mean(rr_train_acc.vec)
# [1] 88.03333
median(rr_train_acc.vec)
# [1] 88.05
mean(rr_test_acc.vec)
# [1] 89.21667
median(rr_test_acc.vec)
# [1] 89.7


# LMan
lman_train_acc.vec <- c(87.1,86.4,77.9,88.6,84.3,86.4)
lman_test_acc.vec <- c(78.8,82.4,79.4,82.4,91.2,82.4)
mean(lman_train_acc.vec)
# [1] 85.11667
median(lman_train_acc.vec)
# [1] 86.4
mean(lman_test_acc.vec)
# [1] 82.76667
median(lman_test_acc.vec)
# [1] 82.4


# deltas
deltas_train_acc.vec <- c(72.9,70.7,72.1,70.7,72.9,71.4)
deltas_test_acc.vec <- c(61.8,67.6,79.4,73.5,70.6,79.4)
mean(deltas_train_acc.vec)
# [1] 71.78333
median(deltas_train_acc.vec)
# [1] 71.75
mean(deltas_test_acc.vec)
# [1] 72.05
median(deltas_test_acc.vec)
# [1] 72.05


# calib-deltas
cdeltas_train_acc.vec <- c(74.3,70,74.3,72.9,70.7,69.3)
cdeltas_test_acc.vec <- c(73.5,85.3,85.3,79.4,76.5,76.5)
mean(cdeltas_train_acc.vec)
# [1] 71.91667
median(cdeltas_train_acc.vec)
# [1] 71.8
mean(cdeltas_test_acc.vec)
# [1] 79.41667
median(cdeltas_test_acc.vec)
# [1] 77.95
