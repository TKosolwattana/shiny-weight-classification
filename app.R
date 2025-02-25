# install.packages("shinydashboard")
# install.packages("rsconnect")
# install.packages("shiny")
# install.packages("shinythemes")
# install.packages("caTools")
# install.packages("randomForest")

library(caTools)
library(shinydashboard)
library(rsconnect)
library(shiny)
library(shinythemes)
library(readr)
library(ggplot2)
library(readxl)
library(caret)
library(dplyr)
library(randomForest)
library(xgboost)
library("SHAPforxgboost")
library(shapviz)
library(pROC)
set.seed(4569)

# Read a dataset
Data = read.csv("./xgb_training_data.csv")
SHAP = read.csv("./shap_true_class.csv")
xgb_model <- readRDS("./xgb_model.rds")
xgb_booster_model <- readRDS("./xgb_model_booster.rds")
Data = subset(Data, select = -c(id))
# Data = subset(Data, select = -c(id,response))
Data_X = Data

Data_group1 = Data[Data$response == 'Minor', ]
Data_group2 = Data[Data$response == 'Moderate', ]
Data_group3 = Data[Data$response == 'Severe', ]

SHAP$response <- Data$response
plotData1 = SHAP[SHAP$response == 'Minor', ]
plotData2 = SHAP[SHAP$response == 'Moderate', ]
plotData3 = SHAP[SHAP$response == 'Severe', ]

# Define groups
group_binary <- c('obeseObese', 'dx_MoodDisorderYes', 'dx_ADYes', 'dx_MentalDisorderYes', 'dx_ConductDisorderYes', 
                  'dx_SUDYes', 'dx_ADHDYes', 'dx_ticYes', 'antidepressionYes', 'adhd_medYes', 'anxietyYes', 
                  'weight_lossYes', 'dx_familyYes', 'dx_schizophrenia_relatedYes', 'dx_autismYes', 'aripiprazoleYes', 
                  'clozapineYes', 'ziprasidoneYes', 'paliperidoneYes', 'olanzapineYes', 'risperidoneYes', 
                  'quetiapineYes', 'sga_switchYes', 'tohighriskYes', 'tolowriskYes', 'dx_diabete_relatedYes', 
                  'ADHD_followYes', 'antidepression_followYes', 'anxiety_followYes', 'weight_loss_followYes', 
                  'insulin_followYes')

group_numeric <- c('age', 'bmiz_base', 'sga_dur', 'ADHD_dur', 'antidepression_dur', 'anxiety_dur', 
                   'weight_loss_dur', 'insulin_dur', 'freq', 'slope', 'acceleration', 'durtoevent')

group_gender <- c('genderM', 'genderU')


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

# Build up AUC/ACC score
predictions <- predict(xgb_model, Data)
Data$response <- as.factor(Data$response)  # Convert the actual response to a factor
predictions <- as.factor(predictions)        # Convert predictions to a factor
levels(predictions) <- levels(Data$response)  # Make sure they have the same levels
# Step 1: Make predictions
predictions <- predict(xgb_model, Data)
# Step 2: Get accuracy
conf_matrix <- confusionMatrix(predictions, Data$response)
accuracy <- conf_matrix$overall['Accuracy']
# Step 3: Accuracy for each class
class_accuracies <- conf_matrix$byClass[, "Balanced Accuracy"]
# Step 4: Make probability predictions (for AUC calculation)
probabilities <- predict(xgb_model, Data, type = "prob")
# Step 5: Calculate AUC for each class (one-vs-all)
auc_values <- sapply(levels(Data$response), function(class) {
    # Create a binary response for the current class (1 for the class, 0 for the others)
    binary_response <- ifelse(Data$response == class, 1, 0)
    # Compute the ROC curve and AUC for the current class
    roc_obj <- roc(binary_response, probabilities[, class])
    auc(roc_obj)
})
# print(class_accuracies)
# #class_accuracies[[1]]
# print(auc_values)

frow_hist <- fluidRow(
    # Histograms for each dataset side by side
    # column(4, plotOutput("hist1")),
    # column(4, plotOutput("hist2")),
    # column(4, plotOutput("hist3")),
    plotOutput("hist1"),
    plotOutput("hist2"),
    plotOutput("hist3"),
    selectInput("selected_column", "Select a variable:",
                choices = colnames(plotData1),
                selected = colnames(plotData1)[1])
)

frow_con <- fluidRow(
    # Histograms for each dataset side by side
    # column(4, plotOutput("hist1")),
    # column(4, plotOutput("hist2")),
    # column(4, plotOutput("hist3")),
    plotOutput("plot1"),
    plotOutput("plot2"),
    plotOutput("plot3"),
    selectInput("selected_c", "Select a variable:",
                choices = c(group_binary, group_numeric, "gender", "race", "region", "provider_specialty"),
                selected = group_binary[1])
)

# UI build up
frow9 <- fluidRow(
    box(
        plotOutput("xgb_imp_plot"),
        numericInput("Imp", "Enter the number of variables:",  10)
    ))

frow_pred <- fluidRow(
    radioButtons(inputId = "obese", 
                 label = "Obesity status",choices = list("Nonobese" = 'Nonobese', "Obese" = 'Obese'
                 ),selected = 'Nonobese'),
    radioButtons(inputId = "gender", 
                 label = "Select your gender",choices = list("Male" = 'M', "Female" = 'F', "Not specified" = 'U'
                 ),selected = 'M'),
    radioButtons(inputId = "race", 
                 label = "Select your race",choices = list("Caucasian" = 'CAUCASIAN', "African" = 'AFRICAN A', "Asian" = 'ASIAN', 'Hispanic' = 'HISPANIC', 'Others' = 'OTHER', 'Unknown' = 'UNKNOWN'
                 ),selected = 'CAUCASIAN'),
    numericInput(inputId = "age", 
                 label = "Enter your age",
                 value = 18
    ),
    radioButtons(inputId = "region1", 
                 label = "Select your region",choices = list("South" = 'South',"Northeast" = 'Northeast', 
                                                             "West" = 'West', "Midwest" = 'Midwest'),
                 selected = 'South'
    ),
    #dx_MoodDisorder
    radioButtons(inputId = "dx_MoodDisorder", 
                 label = "Mood Disorder",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #dx_AD
    radioButtons(inputId = "dx_AD", 
                 label = "AD",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #dx_MentalDisorder
    radioButtons(inputId = "dx_MentalDisorder", 
                 label = "Mental Disorder",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #dx_ConductDisorder
    radioButtons(inputId = "dx_ConductDisorder", 
                 label = "Conduct Disorder",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #dx_SUD
    radioButtons(inputId = "dx_SUD", 
                 label = "SUD",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #dx_ADHD
    radioButtons(inputId = "dx_ADHD", 
                 label = "ADHD",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'Yes'),
    #dx_tic
    radioButtons(inputId = "tic", 
                 label = "tic",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #antidepression
    radioButtons(inputId = "antidepression", 
                 label = "Antidepression",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #adhd_med
    radioButtons(inputId = "adhd_med", 
                 label = "ADHD med",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #anxiety
    radioButtons(inputId = "anxiety", 
                 label = "Anxiety",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #weight_loss
    radioButtons(inputId = "weight_loss", 
                 label = "weight loss",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #provider_specialty
    radioButtons(inputId = "provider_specialty", 
                 label = "Select your provider",choices = list("Men" = 'Men', "Other" = 'Oth', "PCP" = 'PCP', "Unknown" = 'Unk'
                 ),selected = 'Oth'),
    #dx_family
    radioButtons(inputId = "dx_family", 
                 label = "Family",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #dx_schizophrenia_related
    radioButtons(inputId = "dx_schizophrenia_related", 
                 label = "Schizophrenia Related",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #dx_autism
    radioButtons(inputId = "dx_autism", 
                 label = "Autism",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #aripiprazole
    radioButtons(inputId = "aripiprazole", 
                 label = "Aripiprazole",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #clozapine
    radioButtons(inputId = "clozapine", 
                 label = "Clozapine",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #ziprasidone
    radioButtons(inputId = "ziprasidone", 
                 label = "Ziprasidone",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #paliperidone
    radioButtons(inputId = "paliperidone", 
                 label = "Paliperidone",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #olanzapine
    radioButtons(inputId = "olanzapine", 
                 label = "Olanzapine",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #risperidone
    radioButtons(inputId = "risperidone", 
                 label = "Risperidone",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #quetiapine
    radioButtons(inputId = "quetiapine", 
                 label = "Quetiapine",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #sga_switch
    radioButtons(inputId = "sga_switch", 
                 label = "sga_switch",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #tohighrisk
    radioButtons(inputId = "tohighrisk", 
                 label = "tohighrisk",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #tolowrisk
    radioButtons(inputId = "tolowrisk", 
                 label = "tolowrisk",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #dx_diabete_related
    radioButtons(inputId = "dx_diabete_related", 
                 label = "Diabete Related",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #bmiz_base
    numericInput(inputId = "bmiz_base", 
                 label = "Enter your bmiz_base",
                 value = 0
    ),
    #ADHD_follow
    radioButtons(inputId = "ADHD_follow", 
                 label = "ADHD follow",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #antidepression_follow
    radioButtons(inputId = "antidepression_follow", 
                 label = "Antidepression follow",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #anxiety_follow
    radioButtons(inputId = "anxiety_follow", 
                 label = "Anxiety follow",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #weight_loss_follow
    radioButtons(inputId = "weight_loss_follow", 
                 label = "Weight loss follow",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #insulin_follow
    radioButtons(inputId = "insulin_follow", 
                 label = "Insulin follow",choices = list("Yes" = 'Yes', "No" = 'No'
                 ),selected = 'No'),
    #sga_dur
    numericInput(inputId = "sga_dur", 
                 label = "Enter sga duration (days)",
                 value = 0
    ),
    #ADHD_dur
    numericInput(inputId = "ADHD_dur", 
                 label = "Enter ADHD duration (days)",
                 value = 0
    ),
    #antidepression_dur
    numericInput(inputId = "antidepression_dur", 
                 label = "Enter antidepression duration (days)",
                 value = 0
    ),
    #anxiety_dur
    numericInput(inputId = "anxiety_dur", 
                 label = "Enter anxiety duration (days)",
                 value = 0
    ),
    #weight_loss_dur
    numericInput(inputId = "weight_loss_dur", 
                 label = "Enter weight loss duration (days)",
                 value = 0
    ),
    #insulin_dur
    numericInput(inputId = "insulin_dur", 
                 label = "Enter insulin duration (days)",
                 value = 0
    ),
    #freq
    numericInput(inputId = "freq", 
                 label = "Enter frequency",
                 value = 0.5
    ),
    #slope
    numericInput(inputId = "slope", 
                 label = "Enter slope",
                 value = 0.5
    ),
    #acceleration
    numericInput(inputId = "acceleration", 
                 label = "Enter acceleration",
                 value = 0.5
    ),
    #durtoevent
    numericInput(inputId = "durtoevent", 
                 label = "Enter event duration (days)",
                 value = 0
    ),
    actionButton(inputId = "go", 
                 label = "Predict", class = "btn btn-primary"),
    # htmlOutput ("pred"),
    # htmlOutput ("feedback")
    tableOutput('tabledata')
)


header <- dashboardHeader(title = "Prediction model")  
sidebar <- dashboardSidebar(
    sidebarMenu(
        # menuItem("EDA-1", tabName = "EDA-1", icon = icon("chart-bar")),
        # menuItem("EDA-2", tabName = "EDA-2", icon = icon("chart-line")),
        # menuItem("EDA-3", tabName = "EDA-3", icon = icon("columns")),
        menuItem("Overview Feature Contributions", tabName = "Overview", icon = icon("columns")),
        menuItem("Internal Criteria", tabName = "Criteria", icon = icon("chart-bar")),
        menuItem("Variable Consistency", tabName = "Consistency", icon = icon("chart-line")),
        menuItem("Patient Prediction", tabName = "Prediction", icon = icon("bullseye"))
    )
)


body <- dashboardBody(tabItems(
    # tabItem(tabName = "EDA-1",frow1,frow2),
    #                            tabItem(tabName = "EDA-2",frow3,frow4),
    #                            tabItem(tabName = "EDA-3",frow5,frow6,frow7),
    tabItem(tabName = "Overview",frow9),
    tabItem(tabName = "Criteria",frow_hist),
    tabItem(tabName = "Consistency",frow_con),
    tabItem(tabName = "Prediction",frow_pred)))


ui <- dashboardPage(title = 'Application for Heart Disease Classification',header, sidebar, body, skin='red')


# server

server = function(input, output) {
    
    # Check if column is numeric
    is_numeric_column <- function(data, column) {
        is.numeric(data[[column]])
    }
    # Calculate common limits for numeric data across datasets
    get_common_limits <- function(data_list, selected_col) {
        v_min <- Inf
        v_max <- -Inf
        for (data in data_list) {
            if (is_numeric_column(data, selected_col)) {
                col_data <- data[[selected_col]]
                v_min <- min(v_min, min(col_data, na.rm = TRUE))
                v_max <- max(v_max, max(col_data, na.rm = TRUE))
            } 
            # else {
            #     # For non-numeric columns, consider all values from the column to calculate limits
            #     v_min <- min(v_min, min(data[[selected_col]], na.rm = TRUE))
            #     v_max <- max(v_max, max(data[[selected_col]], na.rm = TRUE))
            # }
        }
        return(list(vlim = c(v_min, v_max)))
    }
    # Render histogram for the first dataset
    output$hist1 <- renderPlot({
        selected_col <- input$selected_column
        if (is_numeric_column(plotData1, selected_col)) {
            hist(plotData1[[selected_col]], main = paste("Minor"),
                 xlab = selected_col, col = "lightblue", border = "black",
                 xlim = c(-1, 1))
        } else {
            barplot(table(plotData1[[selected_col]]), main = paste("Minor"),
                    xlab = "Categories", ylab = "Frequency", col = "lightblue", border = "black")
        }
    })
    
    # Render histogram for the second dataset
    output$hist2 <- renderPlot({
        selected_col <- input$selected_column
        if (is_numeric_column(plotData2, selected_col)) {
            hist(plotData2[[selected_col]], main = paste("Moderate"),
                 xlab = selected_col, col = "lightgreen", border = "black",
                 xlim = c(-1, 1))
        } else {
            barplot(table(plotData2[[selected_col]]), main = paste("Moderate"),
                    xlab = "Categories", ylab = "Frequency", col = "lightgreen", border = "black")
        }
    })
    
    # Render histogram for the third dataset
    output$hist3 <- renderPlot({
        selected_col <- input$selected_column
        if (is_numeric_column(plotData3, selected_col)) {
            hist(plotData3[[selected_col]], main = paste("Severe"),
                 xlab = selected_col, col = "lightcoral", border = "black",
                 xlim = c(-1, 1))
        } else {
            barplot(table(plotData3[[selected_col]]), main = paste("Severe"),
                    xlab = "Categories", ylab = "Frequency", col = "lightcoral", border = "black")
        }
    })
    
    output$xgb_imp_plot <- renderPlot({
        sv_importance(shap_values, show_numbers = TRUE, max_display = input$Imp)
        # sv_importance(shap_values, kind = 'no', show_numbers = TRUE, max_display = Inf)
    })
    
    # Function to render appropriate plot for a dataset
    render_plot <- function(data, Data, selected_col, group_name) {
        # Gender group: Show boxplot for genderM and genderU
        if (selected_col == "gender") {
            df <- data.frame(Gender = c(rep("Male", length(data$genderM)), rep("Unknown", length(data$genderU))),
                             Value = c(data$genderM, data$genderU))
            limits_y_n <- get_common_limits(list(plotData1, plotData2, plotData3), 'genderM')
            limits_y_s <- get_common_limits(list(plotData1, plotData2, plotData3), 'genderU')
            ylim_vals1 <- limits_y_n$vlim
            ylim_vals2 <- limits_y_s$vlim
            global_y_min <- min(ylim_vals1[1], ylim_vals2[1])
            global_y_max <- max(ylim_vals1[2], ylim_vals2[2])
            ggplot(df, aes(x = Gender, y = Value, fill = Gender)) +
                geom_boxplot() +
                theme_minimal() +
                labs(title = paste("Box Plots for", selected_col ,"on", group_name ,"group"), y = "Value", x = "Gender") +
                scale_fill_manual(values = c("Male" = "lightblue", "Unknown" = "lightblue")) + scale_y_continuous(limits = c(global_y_min, global_y_max))
        }
        
        # Race group: Show boxplot for race categories
        else if (selected_col == "race") {
            df <- data.frame(Race = c(rep("Asian", length(data$raceASIAN)),
                                      rep("Caucasian", length(data$raceCAUCASIAN)),
                                      rep("Hispanic", length(data$raceHISPANIC)),
                                      rep("Other", length(data$raceOTHER)),
                                      rep("Unknown", length(data$raceUNKNOWN))),
                             Value = c(data$raceASIAN, data$raceCAUCASIAN, data$raceHISPANIC, data$raceOTHER, data$raceUNKNOWN))
            limits_y_as <- get_common_limits(list(plotData1, plotData2, plotData3), 'raceASIAN')
            limits_y_ca <- get_common_limits(list(plotData1, plotData2, plotData3), 'raceCAUCASIAN')
            limits_y_hi <- get_common_limits(list(plotData1, plotData2, plotData3), 'raceHISPANIC')
            limits_y_oth <- get_common_limits(list(plotData1, plotData2, plotData3), 'raceOTHER')
            limits_y_unk <- get_common_limits(list(plotData1, plotData2, plotData3), 'raceUNKNOWN')
            ylim_vals1 <- limits_y_as$vlim
            ylim_vals2 <- limits_y_ca$vlim
            ylim_vals3 <- limits_y_hi$vlim
            ylim_vals4 <- limits_y_oth$vlim
            ylim_vals5 <- limits_y_unk$vlim
            global_y_min <- min(ylim_vals1[1], ylim_vals2[1], ylim_vals3[1], ylim_vals4[1], ylim_vals5[1])
            global_y_max <- max(ylim_vals1[2], ylim_vals2[2], ylim_vals3[2], ylim_vals4[2], ylim_vals5[2])
            ggplot(df, aes(x = Race, y = Value, fill = Race)) +
                geom_boxplot() +
                theme_minimal() +
                labs(title = paste("Box Plots for", selected_col ,"on",group_name ,"group"), y = "Value", x = "Race") +
                scale_fill_brewer(palette = "Pastel1") + scale_y_continuous(limits = c(global_y_min, global_y_max))
        }
        
        # Region group: Show boxplot for region categories
        else if (selected_col == "region") {
            df <- data.frame(Region = c(rep("Northeast", length(data$region1Northeast)), 
                                        rep("South", length(data$region1South)), 
                                        rep("West", length(data$region1West))),
                             Value = c(data$region1Northeast, data$region1South, data$region1West))
            limits_y_n <- get_common_limits(list(plotData1, plotData2, plotData3), 'region1Northeast')
            limits_y_s <- get_common_limits(list(plotData1, plotData2, plotData3), 'region1South')
            limits_y_w <- get_common_limits(list(plotData1, plotData2, plotData3), 'region1West')
            ylim_vals1 <- limits_y_n$vlim
            ylim_vals2 <- limits_y_s$vlim
            ylim_vals3 <- limits_y_w$vlim
            global_y_min <- min(ylim_vals1[1], ylim_vals2[1], ylim_vals3[1])
            global_y_max <- max(ylim_vals1[2], ylim_vals2[2], ylim_vals3[2])
            ggplot(df, aes(x = Region, y = Value, fill = Region)) +
                geom_boxplot() +
                theme_minimal() +
                labs(title = paste("Box Plots for", selected_col ,"on", group_name ,"group"), y = "Value", x = "Region") +
                scale_fill_brewer(palette = "Pastel2") + scale_y_continuous(limits = c(global_y_min, global_y_max))
        }
        
        # Provider specialty group: Show boxplot for provider specialty categories
        else if (selected_col == "provider_specialty") {
            df <- data.frame(Provider = c(rep("Other", length(data$provider_specialtyOth)), 
                                          rep("PCP", length(data$provider_specialtyPCP)), 
                                          rep("Unknown", length(data$provider_specialtyUnk))),
                             Value = c(data$provider_specialtyOth, data$provider_specialtyPCP, data$provider_specialtyUnk))
            limits_y_oth <- get_common_limits(list(plotData1, plotData2, plotData3), 'provider_specialtyOth')
            limits_y_pcp <- get_common_limits(list(plotData1, plotData2, plotData3), 'provider_specialtyPCP')
            limits_y_unk <- get_common_limits(list(plotData1, plotData2, plotData3), 'provider_specialtyUnk')
            ylim_vals1 <- limits_y_oth$vlim
            ylim_vals2 <- limits_y_pcp$vlim
            ylim_vals3 <- limits_y_unk$vlim
            global_y_min <- min(ylim_vals1[1], ylim_vals2[1], ylim_vals3[1])
            global_y_max <- max(ylim_vals1[2], ylim_vals2[2], ylim_vals3[2])
            ggplot(df, aes(x = Provider, y = Value, fill = Provider)) +
                geom_boxplot() +
                theme_minimal() +
                labs(title = paste("Box Plots for", selected_col ,"on", group_name ,"group"), y = "Value", x = "Provider Specialty") +
                scale_fill_brewer(palette = "Set3") + scale_y_continuous(limits = c(global_y_min, global_y_max))
        }
        
        # Binary group: Show box plot
        else if (selected_col %in% group_binary) {
            df <- data.frame(Category = selected_col, Value = data[[selected_col]])
            limits_y <- get_common_limits(list(plotData1, plotData2, plotData3), selected_col)
            ylim_vals <- limits_y$vlim
            ggplot(df, aes(x = Category, y = Value, fill = Category)) +
                geom_boxplot() +
                theme_minimal() +
                labs(title = paste("Box Plot for", selected_col ,"on", group_name ,"group"), y = "Value", x = "Category") +
                scale_fill_manual(values = "lightblue") + scale_y_continuous(limits = ylim_vals)
        }
        
        # Numeric group: Show scatter plot with corresponding column from plotData4 on the x-axis
        else if (selected_col %in% group_numeric) {
            limits_x <- get_common_limits(list(Data_group1, Data_group2, Data_group3), selected_col)
            limits_y <- get_common_limits(list(plotData1, plotData2, plotData3), selected_col)
            xlim_vals <- limits_x$vlim
            ylim_vals <- limits_y$vlim
            df <- data.frame(X = Data[[selected_col]], Y = data[[selected_col]])
            ggplot(df, aes(x = X, y = Y)) +
                geom_point(color = "darkblue", size = 2) +
                geom_smooth(method = "loess", color = "red", size = 1) +  # LOESS regression line
                theme_minimal() +
                labs(title = paste("Box Plot for", selected_col ,"on",   group_name ,"group"), y = "Feature Contributions (SHAP values)", x = selected_col) + scale_x_continuous(limits = limits_x$vlim)+   # Set xlim using limits_x
                scale_y_continuous(limits = ylim_vals)    # Set ylim using limits_y
        }
    }
    
    # Render plot for the first dataset
    output$plot1 <- renderPlot({
        selected_col <- input$selected_c
        limits_y <- get_common_limits(list(plotData1, plotData2, plotData3), selected_col)
        render_plot(plotData1, Data_group1, selected_col, "Minor")  # Use plotData1 for x-axis and y-axis
    })
    
    # Render plot for the second dataset
    output$plot2 <- renderPlot({
        selected_col <- input$selected_c
        render_plot(plotData2, Data_group2, selected_col, "Moderate")  # Use plotData2 for x-axis and y-axis
    })
    
    # Render plot for the third dataset
    output$plot3 <- renderPlot({
        selected_col <- input$selected_c
        render_plot(plotData3, Data_group3, selected_col, "Severe")  # Use plotData3 for x-axis and y-axis
    })
    
    
    ee <- eventReactive(input$go, {
        sample.obs <- data.frame(
            obese = as.factor(input$obese),
            gender = as.factor(input$gender),
            race = as.factor(input$race),
            age = as.integer(input$age),
            region1 = as.factor(input$region1),
            dx_MoodDisorder = as.factor(input$dx_MoodDisorder),
            dx_AD = as.factor(input$dx_AD),
            dx_MentalDisorder = as.factor(input$dx_MentalDisorder),
            dx_ConductDisorder = as.factor(input$dx_ConductDisorder),
            dx_SUD = as.factor(input$dx_SUD),
            dx_ADHD = as.factor(input$dx_ADHD),
            dx_tic = as.factor(input$tic),
            antidepression = as.factor(input$antidepression),
            adhd_med = as.factor(input$adhd_med),
            anxiety = as.factor(input$anxiety),
            weight_loss = as.factor(input$weight_loss),
            provider_specialty = as.factor(input$provider_specialty),
            dx_family = as.factor(input$dx_family),
            dx_schizophrenia_related = as.factor(input$dx_schizophrenia_related),
            dx_autism = as.factor(input$dx_autism),
            aripiprazole = as.factor(input$aripiprazole),
            clozapine = as.factor(input$clozapine),
            ziprasidone = as.factor(input$ziprasidone),
            paliperidone = as.factor(input$paliperidone),
            olanzapine = as.factor(input$olanzapine),
            risperidone = as.factor(input$risperidone),
            quetiapine = as.factor(input$quetiapine),
            sga_switch = as.factor(input$sga_switch),
            tohighrisk = as.factor(input$tohighrisk),
            tolowrisk = as.factor(input$tolowrisk),
            dx_diabete_related = as.factor(input$dx_diabete_related),
            bmiz_base = as.numeric(input$bmiz_base),
            ADHD_follow = as.factor(input$ADHD_follow),
            antidepression_follow = as.factor(input$antidepression_follow),
            anxiety_follow = as.factor(input$anxiety_follow),
            weight_loss_follow = as.factor(input$weight_loss_follow),
            insulin_follow = as.factor(input$insulin_follow),
            sga_dur = as.integer(input$sga_dur),
            ADHD_dur = as.integer(input$ADHD_dur),
            antidepression_dur = as.integer(input$antidepression_dur),
            anxiety_dur = as.integer(input$anxiety_dur),
            weight_loss_dur = as.integer(input$weight_loss_dur),
            insulin_dur = as.integer(input$insulin_dur),
            freq = as.numeric(input$freq),
            slope = as.numeric(input$slope),
            acceleration = as.numeric(input$acceleration),
            durtoevent = as.integer(input$durtoevent)
        )
        Output <- data.frame(Prediction=predict(xgb_model, newdata = sample.obs), round(predict(xgb_model, newdata = sample.obs, type = "prob"), 2))
        # pred_class <- apply(as.matrix(predict(xgb_model, sample.obs)), 1, which.max)
        print(Output)
        # print(str(sample.obs))
        # print(str(d))
        # print(predict(xgb_model, newdata = sample.obs))
        # print(pred_class)
    })
    
    output$tabledata <- renderTable({
        if (input$go>0) { 
            isolate(ee()) 
        } 
    })
}

shinyApp(ui, server)
# rsconnect::deployApp(server='shinyapps.io')
# col.names = c('id','obese',	'gender',	'race',	'age',	'region1',	'dx_MoodDisorder',	'dx_AD',	'dx_MentalDisorder',	'dx_ConductDisorder',	'dx_SUD',	'dx_ADHD',	'dx_tic',	'antidepression',	'adhd_med',	'anxiety',	'weight_loss',	'provider_specialty',	'dx_family',	'dx_schizophrenia_related',	'dx_autism',	'aripiprazole',	'clozapine',	'ziprasidone',	'paliperidone',	'olanzapine',	'risperidone',	'quetiapine',	'sga_switch',	'tohighrisk',	'tolowrisk',	'dx_diabete_related',	'bmiz_base',	'ADHD_follow',	'antidepression_follow',	'anxiety_follow',	'weight_loss_follow',	'insulin_follow','response',	'sga_dur',	'ADHD_dur',	'antidepression_dur',	'anxiety_dur',	'weight_loss_dur',	'insulin_dur',	'freq',	'slope',	'acceleration',	'durtoevent'))
# print(str(d))
# d <- transform(
#     d,
#     obese=factor(obese),
#     gender=factor(gender),
#     race=factor(race),
#     age=as.integer(age),
#     region1=factor(region1),
#     dx_MoodDisorder=factor(dx_MoodDisorder),
#     dx_AD=factor(dx_AD),
#     dx_MentalDisorder=factor(dx_MentalDisorder),
#     dx_ConductDisorder=factor(dx_ConductDisorder),
#     dx_SUD=factor(dx_SUD),
#     dx_ADHD=factor(dx_ADHD),
#     dx_tic=factor(dx_tic),
#     antidepression=factor(antidepression),
#     adhd_med=factor(adhd_med),
#     anxiety=factor(anxiety),
#     weight_loss=factor(weight_loss),
#     provider_specialty=factor(provider_specialty),
#     dx_family=factor(dx_family),
#     dx_schizophrenia_related=factor(dx_schizophrenia_related),
#     dx_autism=factor(dx_autism),
#     aripiprazole=factor(aripiprazole),
#     clozapine=factor(clozapine),
#     ziprasidone=factor(ziprasidone),
#     paliperidone=factor(paliperidone),
#     olanzapine=factor(olanzapine),
#     risperidone=factor(risperidone),
#     quetiapine=factor(quetiapine),
#     sga_switch=factor(sga_switch),
#     tohighrisk=factor(tohighrisk),
#     tolowrisk=factor(tolowrisk),
#     dx_diabete_related=factor(dx_diabete_related),
#     bmiz_base=as.numeric(bmiz_base),
#     ADHD_follow=factor(ADHD_follow),
#     antidepression_follow=factor(antidepression_follow),
#     anxiety_follow=factor(anxiety_follow),
#     weight_loss_follow=factor(weight_loss_follow),
#     insulin_follow=factor(insulin_follow),
#     sga_dur=as.integer(sga_dur),
#     ADHD_dur=as.integer(ADHD_dur),
#     antidepression_dur=as.integer(antidepression_dur),
#     anxiety_dur=as.integer(anxiety_dur),
#     weight_loss_dur=as.integer(weight_loss_dur),
#     insulin_dur=as.integer(insulin_dur),
#     freq=as.numeric(freq),
#     slope=as.numeric(slope),
#     acceleration=as.numeric(acceleration),
#     durtoevent=as.integer(durtoevent)
# )