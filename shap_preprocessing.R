# run the model with built-in data, these codes can run directly if package installed  
library("SHAPforxgboost")
library(shapviz)

Data = read.csv("C:\\Users\\ARTY\\Desktop\\UH\\Ph.D. research\\Shiny\\xgb_training_data.csv")
Data = subset(Data, select = -c(id))
# Data = subset(Data, select = -c(id,response))
# Data_X = subset(Data, select = -c(id,response))
Data_X = Data
# Data_X = subset(Data, select = -c(id))
xgb_model <- readRDS("C:\\Users\\ARTY\\Desktop\\UH\\Ph.D. research\\Shiny\\xgb_model.rds")
xgb_model <- readRDS("C:\\Users\\ARTY\\Desktop\\UH\\Ph.D. research\\Shiny\\xgb_model.rds")
xgb_booster_model <- readRDS("C:\\Users\\ARTY\\Desktop\\UH\\Ph.D. research\\Shiny\\xgb_model_booster.rds")

# Convert features to the xgb compatible format 
dummies_model <- dummyVars(response ~ ., data = Data)
train_numeric <- predict(dummies_model, newdata = Data_X)
train_numeric <- as.data.frame(train_numeric)
train_matrix <- as.matrix(train_numeric)  # Exclude target
model_features <- xgb_booster_model$feature_names
newdata_features <- colnames(train_matrix)
# Compare the feature names
features_to_remove <- setdiff(newdata_features, model_features)
# Remove the features identified by setdiff() from the train_matrix
train_matrix_cleaned <- train_matrix[, !(colnames(train_matrix) %in% features_to_remove)]

#calculate feature importance value
shap_values <- shapviz(xgb_booster_model, X_pred = train_matrix_cleaned, X = train_matrix_cleaned)

G <- factor(Data$response, levels = c('Minor', 'Moderate', 'Severe'), labels = c(1, 2, 3))
# Assuming 'Data$response' contains the true class labels and Data_X is the features matrix
true_labels <- G

# Create an empty matrix for storing SHAP values for the true class
shap_true_class <- matrix(0, nrow = length(true_labels), ncol = ncol(shap_values[[1]]))

# Loop through each data point, select the SHAP values for its true class
for (i in 1:length(true_labels)) {
    true_class <- true_labels[i]  # Get the true class for the ith data point
    
    # Extract the SHAP values for the true class
    shap_matrix <- shap_values[[true_class]]$S
    
    # Ensure the SHAP values are stored correctly
    shap_true_class[i, ] <- shap_matrix[i, ]  # Index the i-th row
}

# Convert the resulting matrix to a data frame
shap_df_true <- as.data.frame(shap_true_class)

# Assign column names based on the feature names from Data_X
colnames(shap_df_true) <- colnames(train_matrix_cleaned)

# Write the dataframe to a CSV file
write.csv(shap_df_true, "C:\\Users\\ARTY\\Desktop\\UH\\Ph.D. research\\Shiny\\shap_true_class.csv", row.names = FALSE)