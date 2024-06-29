# Step 1: Prepare the Data
#Click "File" at the top left corner => import data set => from excel 
#=> Browse (select the data set) => Import 

# Step 2: Install and load necessary libraries
install.packages("caret")
install.packages("neuralnet")
library(caret)
library(neuralnet)

#Step 3: Split the dataset into training and testing sets
set.seed(123)
split <- createDataPartition(diabetes_dataset$diabetes, p = 0.7, list = FALSE)
train_data <- diabetes_dataset[split, ]
test_data <- diabetes_dataset[-split, ]

#Step 4: Define the formula for the neural network
formula <- diabetes ~ gender + age + hypertension + heart_disease + smoking_history + bmi + HbA1_level + blood_glucose_level

#Step 5: Train the neural network model
nn_model <- neuralnet(formula, data = train_data, hidden = 2, linear.output = FALSE, act.fct = "logistic")

#Step 6: Plot the Artificial Neural Network model
plot(nn_model)

#Step 7: Make predictions on test set
predictions <- predict(nn_model, newdata = test_data)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

#Step 7.1: Evaluate model performance
confusion_matrix <- table(predicted_classes, test_data$diabetes)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

#Step 7.2: Print the accuracy and confusion matrix
print(paste("Accuracy:", accuracy))
print("Confusion Matrix:")
print(confusion_matrix)

#Step 8: Create a sample test set with 2 individuals
test <- data.frame(
  gender = c(0, 0),  # Gender of the individuals
  age = c(56, 18),  # Age of the individuals
  hypertension = c(0, 0),  # Hypertension status (1: Yes, 0: No)
  heart_disease = c(0, 0),  # Heart disease status (1: Yes, 0: No)
  smoking_history = c(3, 0),  # Smoking history ('never': 0 | 'No Info': 1 | 'not current': 2 | 'current': 3 | 'former': 4)
  bmi = c(20.70, 23),  # Body Mass Index (BMI)
  HbA1_level = c(5.7, 5.0),  # HbA1c level
  blood_glucose_level = c(100, 95)  # Blood glucose level
)

#Step 9: Make predictions on sample test set
new_predictions <- predict(nn_model, test)
new_predicted_classes <- ifelse(new_predictions > 0.5, 'Diabetes', 'Non-diabetes')
print(new_predicted_classes)
