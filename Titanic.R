# ---- Import_libraries ----

library(tidyverse)
library(caret)
library(Cairo)
library(randomForest)
library(gam)
library(Rborist)
library(timeR)
library(ggthemes) # visualization
library(ggridges) # visualization
library(corrplot) # correlation visualisation
library(VIM) # missing values

#CairoWin()



#rm(list = ls())


# ---- Import_Datas ----
df <-
  read_csv("C:/Users/Mathieu/Documents/Dropbox/DataAnalysis/R/titanic/datas/train.csv")

df <- df %>% mutate(
  Survived = as.factor(Survived),
  Pclass = as.factor(Pclass),
  Embarked = as.factor(Embarked),
  Sex = as.factor(Sex)
)

y <- df$Survived



# ---- Exploratory_Data_Analysis ----


# ---- First_Check_NA ----
summary(df)
#str(df)
glimpse(df)


#let's look at the differents NA values
aggr(
  df,
  prop = FALSE,
  combined = TRUE,
  numbers = TRUE,
  sortVars = TRUE,
  sortCombs = TRUE
)

#There's 529 observations where the only missing value is the cabin, 158 obsevations where it is the cabin and the age, 19
#obsevartions where only the age is missing and finally, 2 observations where the place of embarkation is missing.

# ---- Correlation_matrix ----
df %>%
  select(-PassengerId, -Name, -Cabin, -Ticket) %>%
  mutate(Sex = fct_recode(Sex,
                          "0" = "male",
                          "1" = "female")) %>%
  
  mutate(
    Sex = as.integer(Sex),
    Pclass = as.integer(Pclass),
    Survived = as.integer(Survived),
    Embarked = as.integer(Embarked)
  ) %>%
  cor(use = "complete.obs", method = "pearson") %>%
  corrplot(type = "lower", diag = FALSE)



# ---- Age ----

#let's look at the age repartition with respect to the class
df %>% ggplot(aes(x = Age, y = ..density.., fill = Pclass)) +
  geom_histogram(binwidth = 4) +
  geom_density(alpha = 0.3) +
  facet_grid(. ~ Pclass)

#Although the ship is mostly filled with adults around 30-40 years old, young adults and children seem more common in the 2nd and 3rd class,
#whereas older people, ie > 40 years old are more common in the first class

#class and Sex
df %>% group_by(Pclass) %>%
  summarize(mean(Survived == 1))


#The class with the highest rate of survival if the first one. In the third class, nearly 75% died.

df %>% group_by(Sex) %>%
  summarize(mean(Survived == 1))

#Nearly 75% of the women survived

# ---- NP_Cabin ----

# let's look at the cabin, the first letter corresponds to the deck. Let's see what we can deduce from it

df <- df %>% mutate(Deck = substring(Cabin, 1, 1))

#What's the repartition of the class with respect to the deck

df %>% filter(!is.na(Deck)) %>%
  ggplot(aes(x = Pclass, fill = Pclass)) +
  geom_bar() +
  facet_wrap(. ~ Deck)

# Decks A,B,C, and T are exclusively filled with 1st class, G with 3rd class, 2nd class is only found on decks D,E,F.
#What's the rate of Survival for each deck


df %>% filter(!is.na(Deck)) %>%
  group_by(Deck) %>%
  summarize(Surv_rate = mean(Survived == 1)) %>%
  arrange(desc(Surv_rate))

df %>% filter(!is.na(Deck)) %>%
  ggplot(aes(x = Pclass, fill = Survived)) +
  geom_bar(position = 'fill') +
  facet_wrap(. ~ Deck) +
  ylab('Survival rate')



#Everybody died od Deck T, decks D, E, B, F have the highest survival rate.

# ---- NP_Title ----

# The full name won't be usefull as it is, however, we can extract the title of the passenger
df <- df %>%
  mutate(Title = str_extract(Name, "[A-Z][a-z]*\\.")) %>%
  mutate(Title = str_replace_all(Title, "\\.", "")) %>%
  mutate(Title = str_trim(Title))

#crosstable
table(df$Title, df$Sex)

df %>% group_by(Title) %>%
  summarize(rate = mean(Survived == 1)) %>%
  arrange(desc(rate))

#Master accounts for young children, apart from female and male title, one can also noticed some "rare titles" like "Sir", "Don" etc

df <- df %>%
  mutate(Title = factor(Title)) %>%
  mutate(Title = fct_collapse(
    Title,
    "Miss" = c("Mlle", "Ms"),
    "Mrs" = "Mme",
    "Ranked" = c("Major", "Dr", "Capt", "Col", "Rev"),
    "Royalty" = c("Lady", "Countess", "Don", "Sir", "Jonkheer")
  ))


#
# female_title <- c("Miss", "Mlle", "Mme", "Mrs", "Ms")
#
# df <-
#   df %>% mutate(
#     Title_type = case_when(
#       Title %in% female_title ~ "female",
#       Title == "Mr" ~ "male",
#       Title == "Master" ~ "child",
#       TRUE ~ "rare"
#     )
#   )

df %>% group_by(Title) %>%
  summarize(rate = mean(Survived == 1)) %>%
  arrange((desc(rate)))

# ---- NP_Family ----

# Family : chldren, siblings, spouses, etc

df <- df %>% mutate(family = Parch + SibSp + 1)


df %>% ggplot(aes(x = family, fill = Survived)) +
  geom_bar(position = 'fill') +
  ylab('Survival rate')


df %>% group_by(family) %>%
  summarize(rate = mean(Survived == 1)) %>%
  arrange(desc(rate))


#small families, ie between 2-4 people, were much more likely to survive. Let's create a predictor for that

df <-
  df %>% mutate(family_size = factor(ifelse(
    family > 4, "large", ifelse(2 < family, "medium", "small")
  )))

df %>% ggplot(aes(x = family_size, fill = Survived)) +
  geom_bar(position = 'fill') +
  ylab('Survival rate')


# ---- NP_Fare1 ----

df %>% ggplot(aes(x = Fare, y = Pclass, fill = Pclass)) +
  geom_density_ridges()

#As it is, the Fare doesn't seem to be usefull. The strange thing here is that the class here overlap a lot, which shouldn't be. The
#price of the travel in the 3rd class shouldn't overlap as much the price of the 1rst class, which is the most expensive one.

df %>% group_by(Ticket, Pclass, Fare) %>%
  summarize(n = n()) %>%
  arrange(desc(n))

#As we can see here, the Fare predictor does not seem to correspond to the price of one person, but to the whole price for a group,
#i.e. the ticket 1601 seems to account for 7 people traveling together. The real price should then be given by the following code.

# ---- NP_Fare2 ----


df <- df %>% group_by(Ticket) %>%
  mutate(real_fare = Fare / n(),
         shared_ticket = ifelse(n() > 1, 1, 0)) %>%
  ungroup()


df %>%
  filter(shared_ticket == 1) %>%
  ggplot(aes(Pclass, fill = Survived)) +
  geom_bar(position = 'fill')

df %>%
  filter(real_fare > 0) %>%
  ggplot(aes(real_fare, Pclass, fill = Pclass)) +
  geom_density_ridges() +
  scale_x_log10(lim = c(3, 1000)) +
  labs(x = "real_fare")

#We now have a much clearer separation between the different classes.

# NP_tticket
#
# df <- df %>%
#   mutate(tticket = str_extract(Ticket, "A*\\.*\\/*5*[A-Z]*\\.*[A-Z]+\\.*[A-Z]*\\/*[A-Z]*\\.*[A-Z]*\\.*[A-Z]*")) %>%
#   mutate(tticket = str_replace_all(tticket, "\\.", "")) %>%
#   mutate(tticket = str_replace_all(tticket, "STON", "SOTON")) %>%
#   mutate(tticket = str_replace_all(tticket, "\\/", ""))
#
# #crosstable
# table(df$Pclass, df$tticket)
#
#
# df %>% filter(!is.na(tticket) &  Pclass ==3) %>%
#   ggplot((aes(x = Deck, fill = tticket)))+
#   geom_bar(position ='dodge')+
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
#   coord_flip()
#

# too much NA's, doesnt seem to be worth the effort


#	---- Embarkation ----

df %>%
  ggplot(aes(x = Embarked, fill = Survived)) +
  geom_bar(position = 'fill') +
  ylab('Survival rate')

df %>% filter(!is.na(Embarked)) %>%
  group_by(Embarked) %>%
  summarize(n = n(),
            Survival_rate = mean(Survived == 1) * 100)

df %>% filter(!is.na(Embarked)) %>%
  group_by(Embarked, Pclass) %>%
  summarize(n = n(),
            Survival_rate = mean(Survived == 1) * 100)

# ---- Save_State_1 ----

df <- df %>% ungroup()


#Saved Step with all the new predictors

saveRDS(df, file = "Titanic/datas/my_data_NP.rds")


df <- readRDS(file = "Titanic/datas/my_data_NP.rds")
y <- df$Survived


# ---- Prepocessing ----

# ---- Missing_Embarkation ----

df %>% filter(is.na(Embarked))

df %>% filter(!is.na(Embarked) & Pclass == 1 & Deck == 'B') %>%
  group_by(Embarked) %>%
  summarize(n = n())

df <- df %>%
  mutate(Embarked = as.character(Embarked)) %>%
  mutate(Embarked = case_when(is.na(Embarked) ~ "S",
                              TRUE ~ Embarked)) %>%
  mutate(Embarked = as.factor(Embarked))



# ---- Missing_Deck1 ----



df %>%
  group_by(Pclass) %>%
  summarize(
    n = n(),
    missing = sum(is.na(Deck)),
    ratio_missing_deck = sum(is.na(Deck)) * 100 / n()
  )

df <- df %>% mutate(known_deck = ifelse(!is.na(Deck), 1, 0))

# ---- Missing_Age ----


df <-
  select(df,
         -c(Cabin,
            PassengerId,
            Name,
            Ticket,
            Fare,
            Deck,
            SibSp,
            Parch,
            family))

df <- df %>% mutate(
  Sex = factor(Sex),
  Pclass = factor(Pclass),
  Survived = factor(Survived),
  Embarked = factor(Embarked),
  shared_ticket = factor(shared_ticket),
  known_deck = factor(known_deck)
)



str(df)


df_preProcess <- preProcess(df, method = 'bagImpute')
df_preProcess
df <- predict(df_preProcess, newdata = df)
anyNA(df)



# ---- Save_State_2 ----

df <- df %>% ungroup()


#Once you've finished tweaking your predictors and just want to get your hands on ML algorithms,
#Saving your file so you don't have to run all the previous script

saveRDS(df, file = "Titanic/datas/my_data.rds")


df <- readRDS(file = "Titanic/datas/my_data.rds")
y <- df$Survived



# ---- New_Correlation ----
df %>%
  mutate(Sex = fct_recode(Sex,
                          "0" = "male",
                          "1" = "female")) %>%
  
  mutate(
    Sex = as.integer(Sex),
    Pclass = as.integer(Pclass),
    Survived = as.integer(Survived),
    Embarked = as.integer(Embarked),
    Title = as.integer(Title),
    family_size = as.integer(family_size),
    shared_ticket = as.integer(shared_ticket),
    known_deck = as.integer(known_deck)
  ) %>%
  cor(use = "complete.obs", method = "pearson") %>%
  corrplot(type = "lower", diag = FALSE)

# ---- Machine_learning ----

# ---- Partition_creation ----
set.seed(1)
val_index <-
  createDataPartition(y, times = 1, p = 0.8, list = FALSE)



df_train <- df[val_index, ]
y_train <- as.factor(y[val_index])
df_validation <- df[-val_index, ]
y_validation <- as.factor(y[-val_index])



# ---- Ensembling_Training1 ----

#Use the training set to build a model with several of the models available from the caret package.
#We will test out all of the following models:

models <- c("rf",
            "ranger",
            "RRF",
            "wsrf",
            "Rborist",
            "avNNet",
            "mlp",
            "monmlp",
            "adaboost",
            "gbm")


#Run the following code to train the various models:



# ---- Timer ----

set.seed(1)

# Create a timer object
train_timer <- createTimer()

#no need to run this part if you don't want timer, as timer isn't compatible with predict.
fits_timer <- lapply(models, function(model) {
  train_timer$start(model)
  print(model)
  train(Survived ~ ., method = model, data = df_train)
  train_timer$stop(model, comment = 'train')
})

names(fits_timer) <- models


# ---- Timer2 ----

getTimer(train_timer) %>% knitr::kable()


# ---- Ensembling_Training2 ----

set.seed(1)


fits <- lapply(models, function(model) {
  print(model)
  train(Survived ~ ., method = model, data = df_train)
})

names(fits) <- models





# ---- Ensembling_Predicting ----

#Now that you have all the trained models in a list, use sapply or map to create a matrix of predictions
#for the validation set.

fits_predicts <- sapply(fits, function(fits) {
  predict(fits, newdata = df_validation)
})

dim(fits_predicts)

# ---- First_Acc_report ----


#Accuracy for each model in the validation set and the mean accuracy across all models can be
#computed using the following code:

acc <- colMeans(fits_predicts == y_validation)
acc %>% knitr::kable()
mean(acc)

votes <- rowMeans(fits_predicts == "1")
y_hat <- ifelse(votes > 0.5, "1", "0")
mean(y_hat == y_validation)


results <-
  tibble(
    method = "All the models",
    Sensitivity = confusionMatrix(reference = y_validation, data = factor(y_hat))$byClass[1],
    Specificity = confusionMatrix(reference = y_validation, data = factor(y_hat))$byClass[2]
  )

results %>% knitr::kable()


# ---- Acc_via_CV ----

#Use the accuracy estimates obtained from cross validation with the training data.
#Obtain these estimates and save them in an object. Report the mean accuracy of the new estimates.
#What is the mean accuracy of the new estimates?

# accuracy_cv <-
#   sapply(models, function(m) {
#     min(fits[[m]][['results']][['Accuracy']])
#   })
#
# accuracy_cv %>% knitr::kable()
# mean(accuracy_cv)

#You can calculate the mean accuracy of the new estimates using the following code:

acc_hat <- sapply(fits, function(fit)
  min(fit$results$Accuracy))

acc_hat %>% knitr::kable()

mean(acc_hat)


# ---- Model_Selection_Training ----

#Now let's only consider the methods with an estimated accuracy of greater than or equal to 0.8
#when constructing the ensemble.
#What is the accuracy of the ensemble now?

names(which(acc_hat >= 0.8))

new_models <- names(which(acc_hat >= 0.8))

new_fits <- lapply(new_models, function(model) {
  print(model)
  train(Survived ~ ., method = model, data = df_train)
})

names(new_fits) <- new_models

# ---- Model_Selection_Predicting ----


new_fits_predicts <- sapply(new_fits, function(fits) {
  predict(fits, newdata = df_validation)
})

votes <- rowMeans(new_fits_predicts == "1")
y_hat <- ifelse(votes > 0.5, "1", "0")
mean(y_hat == y_validation)


results <- bind_rows(
  results,
  tibble(
    method = "The best models",
    Sensitivity = confusionMatrix(reference = y_validation, data = factor(y_hat))$byClass[1],
    Specificity = confusionMatrix(reference = y_validation, data = factor(y_hat))$byClass[2]
  )
)
results %>% knitr::kable()
