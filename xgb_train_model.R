# Trained XGBoost model in R

library(caret)
library(xgboost)
library(shapviz)
library(pROC)

train = read.csv("C:\\Users\\ARTY\\Desktop\\UH\\Ph.D. research\\Shiny\\xgb_training_data.csv")
train = subset(train, select = -c(id))

# Define cross-validation for the train set
trainControl <- trainControl(
    method = "cv",         # Use k-fold cross-validation
    number = 5,            # 5 folds
    search = "random"      # Random search
)

# Perform random search on the full dataset with XGBoost
set.seed(123)  # For reproducibility
xgb_model <- train(
    response ~ .,            # Formula: Outcome as the target, rest as predictors
    data = train,             # Use the train dataset
    method = "xgbTree",     # XGBoost model
    metric = "Accuracy",    # Metric to optimize
    trControl = trainControl,
    tuneLength = 10         # Try 10 random combinations of hyperparameters
)
xgb_booster_model <- xgb_model$finalModel
# saveRDS(xgb_model, "xgb_model.rds")
saveRDS(xgb_model, file = "C:\\Users\\ARTY\\Desktop\\UH\\Ph.D. research\\Shiny\\xgb_model.rds")
saveRDS(xgb_booster_model, file = "C:\\Users\\ARTY\\Desktop\\UH\\Ph.D. research\\Shiny\\xgb_model_booster.rds")


########################AUC/ACC##############################
train$response <- as.factor(train$response)  # Convert the actual response to a factor
predictions <- as.factor(predictions)        # Convert predictions to a factor
levels(predictions) <- levels(train$response)  # Make sure they have the same levels

# Step 1: Make predictions
predictions <- predict(xgb_model, train)

# Step 2: Get accuracy
conf_matrix <- confusionMatrix(predictions, train$response)
accuracy <- conf_matrix$overall['Accuracy']
print(paste("Accuracy:", accuracy))

# Step 3: Make probability predictions (for AUC)
probabilities <- predict(xgb_model, train, type = "prob")

# Step 4: Calculate AUC
# 'multiclass.roc' handles multiclass AUC computation
multiclass_auc <- multiclass.roc(train$response, probabilities)
auc_value <- auc(multiclass_auc)
print(paste("Multiclass AUC:", auc_value))

# Step 3: Accuracy for each class
class_accuracies <- conf_matrix$byClass[, "Balanced Accuracy"]
# print("Accuracy for each class:")
# print(class_accuracies)

# Step 4: Make probability predictions (for AUC calculation)
probabilities <- predict(xgb_model, train, type = "prob")

# Step 5: Calculate AUC for each class (one-vs-all)
auc_values <- sapply(levels(train$response), function(class) {
    # Create a binary response for the current class (1 for the class, 0 for the others)
    binary_response <- ifelse(train$response == class, 1, 0)
    # Compute the ROC curve and AUC for the current class
    roc_obj <- roc(binary_response, probabilities[, class])
    auc(roc_obj)
})

# print("AUC for each class:")
# print(auc_values)