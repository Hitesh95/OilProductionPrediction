

#Loading library packages.
library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)

#Loading training and test files
train <-read.csv("C:/Users/h.rajesh.hinduja/Desktop/train.csv",stringsAsFactors = F)
test <-read.csv("C:/Users/h.rajesh.hinduja/Desktop/test.csv",stringsAsFactors = F)

#Checkingthe dimensions of the train file
dim(train)
str(train[,c(1:11, 12)])

#Making predicted values of Response variable as NA in the test data.
test$OR<- NA
all <-rbind(train, test)
dim(all)

#Plotting Response variable plot to see the skewness in the response variable
ggplot(data=all[!is.na(all$OR),],aes(x=OR)) +geom_histogram(fill="blue",binwidth = 1000) +scale_x_continuous(breaks=seq(0, 50000, by=1000), labels = comma)

#Checking the numeric variables and plotting the correlation plot
summary(all$OR)
numericVars<- which(sapply(all, is.numeric))
numericVarNames<- names(numericVars)
cat('There are', length(numericVars), 'numeric variables')
all_numVar<- all[, numericVars]
cor_numVar<- cor(all_numVar, use="pairwise.complete.obs")
head(cor_numVar)
cor_sorted<- as.matrix(sort(cor_numVar[,'OR'], decreasing = TRUE))
head(cor_sorted)
CorHigh<- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
head(CorHigh)
CorHigh
cor_numVar<- cor_numVar[CorHigh, CorHigh]

#checkingif any column has NA value(except the response variable) and imputing the data if there is any NA
NAcol <-which(colSums(is.na(all)) > 0)
sort(colSums(sapply(all[NAcol], is.na)), decreasing = TRUE)
cat('There are', length(NAcol), 'columns with missing values')
numericVars <- which(sapply(all, is.numeric))

factorVars <- which(sapply(all, is.factor))
cat('There are', length(numericVars), 'numeric variables, and', length(factorVars), 'categoric variables')

#Again plotting the correlation plot and removing the variables which are highlycorrelated to remove multicollinearity.
all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs")
cor_sorted<- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
cor_sorted<- as.matrix(sort(cor_numVar[,'OR'], decreasing = TRUE))
CorHigh<- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar<- cor_numVar[CorHigh, CorHigh]
cor_numVar
corrplot.mixed(cor_numVar,tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7,number.cex=.7)

#Quick randomforest to understand the variable importances.
set.seed(2018)
quick_RF<- randomForest(x=all[1:7673,-12], y=all$OR[1:7673],ntree=100,importance=TRUE)
imp_RF <-importance(quick_RF)
imp_DF <-data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <-imp_DF[order(imp_DF$MSE, decreasing = TRUE),]
imp_DF
ggplot(imp_DF[1:15,],aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + geom_bar(stat = 'identity')+ labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') +coord_flip() + theme(legend.position="none")

#Preprocessing and scaling our independent variables and removing skewness in the response variable (Centre scaling)
DFnumeric <- all[, names(all) %in% numericVarNames]
DFfactors <- all[, !(names(all) %in% numericVarNames)]
DFfactors<- DFfactors[, names(DFfactors) != 'OR']
cat('There are', length(DFnumeric), 'numeric variables, and', length(DFfactors), 'factor variables')

# removing skewness from the numeric variables
for(i in 1:ncol(DFnumeric)){
if(abs(skew(DFnumeric[,i]))>0.8){
DFnumeric[,i] <- log(DFnumeric[,i] +1)
}
}
PreNum <- preProcess(DFnumeric, method=c("center", "scale"))
print(PreNum)
head(PreNum)
DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)
DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors))

#removing skewness an plotting the QQ plot before and after skewing the response variable
skew(all$OR)
qqnorm(all$OR)
qqline(all$OR)
all$OR <- log(all$OR)
skew(all$OR)
qqnorm(all$OR)
qqline(all$OR)
head(DFnorm)

#Finally preparing training and testing dataset for model development.
train1 <- DFnorm[!is.na(all$OR),]
test1 <- DFnorm[is.na(all$OR),]

#Checked accuracy with lasso model
set.seed(27042018)
my_control<-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))
lasso_mod <- train(x=train1, y=all$OR[!is.na(all$OR)], method='glmnet', trControl= my_control, tuneGrid=lassoGrid)
lasso_mod$bestTune
min(lasso_mod$results$RMSE)

#finding variable importance using lasso
lassoVarImp <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance
varsSelected <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))
cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')

#predictions using lasso
LassoPred <- predict(lasso_mod, test1)
predictions_lasso <- exp(LassoPred)
head(predictions_lasso)
predictions_lasso
write.csv(predictions_lasso, file = "path\predictions.csv")

#Plotting the variable importance graph using xgboost to double check with random forest variable importance
library(Ckmeans.1d.dp)
mat <- xgb.importance (feature_names = colnames(train1),model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:11], rel_to_first = TRUE)

