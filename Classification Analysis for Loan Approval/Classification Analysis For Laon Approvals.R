
# Loading packages
x <-c("plyr", "wordcloud", "tm", "e1071", "RTextTools","dplyr","ggplot2","RColorBrewer","rpart","rpart.plot","randomForest","ROCR","caret")
lapply(x, require,character.only = TRUE)

# settng working directory
setwd('C:/Users/sandeep/Desktop/Quovo/Classification Analysis for Loan Approval')

# loading approved and declined csv files

Approved_data <- read.csv('LoanStats3d.csv')
Declined_data <- read.csv('RejectStatsD.csv')

# selecting the required columns
Approved_data1 <- select(Approved_data,loan_amnt,purpose,emp_length,dti)
Declined_data1 <- select(Declined_data,Amount.Requested,Loan.Title,Employment.Length,Debt.To.Income.Ratio)

colnames(Aprroved_data1) <- tolower(colnames(Aprroved_data1))
colnames(Declined_data1) <- tolower(colnames(Declined_data1))

# change column names of Declined_data1 to approved_data1
colnames(Declined_data1) = colnames(Approved_data1)

# removing rows with nas
Approved_data2 <- Approved_data1[complete.cases(Approved_data1),]
Declined_data2 <- Declined_data1[complete.cases(Declined_data1),]

# removing rows with nas in emp_length
Approved_data3 <- Approved_data2[!(Approved_data2$emp_length == 'n/a'),]
Declined_data3 <- Declined_data2[!(Declined_data2$emp_length == 'n/a'),]

# adding a columns Approval with 1 for approved data and 0 for declined data
Approved_data3$approval <- 1
Declined_data3$approval <- 0

#removing % character from dti column
Declined_data3$dti <- as.numeric(gsub("%","",as.character(Declined_data3$dti)))

t <- nrow(Approved_data3)/nrow(Declined_data3)

# random selection of rows of declined data file as the number of rows in approved data file is less than declined data file  
split0 <- sample(nrow(Declined_data3),floor(t*nrow(Declined_data3)))
Declined_data_Final <- Declined_data3[split0,]

# final data
Final_data <- rbind(Approved_data3,Declined_data_Final)

Final_data$purpose <- as.character(Final_data$purpose)
Final_data$emp_length <- as.character(Final_data$emp_length)

# seprated rows training, validation and test
split <- sample(nrow(Final_data),nrow(Final_data)*4/5)

Final_initial<- Final_data[split,]
Final_test <- Final_data[-split,]

split1 <- sample(nrow(Final_initial),nrow(Final_initial)*3/4)

Final_train <- Final_initial[split1,]
Final_validation <- Final_initial[-split1,]

Final_test

# Building model using decision trees

fit.dtree <- rpart(approval ~ .,Final_train, method = "class")
rpart.plot(fit.dtree, type = 4, extra = 101)

predict_train <- predict(fit.dtree,Final_train, type = "class")
predict_valid <- predict(fit.dtree,Final_validation, type = "class")

Training_error = 1 - mean(Final_train$approval == as.numeric(as.character(predict_train)))
print(Training_error)

Validation_error = 1 - mean(Final_validation$approval == as.numeric(as.character(predict_valid)))
print(Validation_error)

# Generate ROC Curve

pred <- prediction(as.numeric(as.character(predict_train)), Final_train$approval)

plot(performance(pred, "tpr", "fpr"))
abline(0, 1, lty = 2)

# calculation of confusion matrix, accuracy, precision, sensitivity and specificity
result <- confusionMatrix(as.numeric(as.character(predict_train)), Final_train$approval)


# state vs loan performance

detach(package:plyr)

Approved_data$check <- 1
Approved_data$addr_state <- as.character(Approved_data$addr_state)
Approved_data$loan_status <- as.character(Approved_data$loan_status)

a <- Approved_data %>% filter(loan_status == 'Fully Paid') %>%
  group_by(addr_state) %>% summarise(Full_paid_total = sum(check))

b <- Approved_data %>% filter(loan_status != 'Current') %>%
  group_by(addr_state) %>% summarise(total_excl_current = sum(check))

c <- inner_join(a,b,by = "addr_state")
c$loan_performance <- c$Full_paid_total/c$total_excl_current

write.csv(c,'analysis_states.csv')
