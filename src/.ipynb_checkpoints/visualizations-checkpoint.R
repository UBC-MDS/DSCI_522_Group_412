# author: Alistair Clark
# date: 2020-01-21

"This script creates exploratory data visualizations.

Usage: visualizations.R --input_file<input_file> --output_file<output_file>

Options:
--input_file<input_file>    A path/filename for the input data.
--output_file<output_file>  A path/filename for the created figure.

" -> doc

### Ideas and to-dos
# add inputs of categorical column names and numerical column names?
# Approved as factor or continuous? I have both plots below
# main function calls visualize_categorical once for eery feature
# main function calls ggpairs once
# main function MIGHT need to split data sets? Or select based on list of numerical/categorical?


library(docopt)
library(GGally)
library(tidyverse)

# Manual data cleaning that will be replaced with other script

credit_data <- read_csv("data/raw.csv",
                        col_types = "icddccccdllilccdc",
                        col_names = c('X1',
                                      'Sex',
                                      'Age',
                                      'Debt',
                                      'Married',
                                      'BankCustomer',
                                      'EducationLevel',
                                      'Ethnicity',
                                      'YearsEmployed',
                                      'PriorDefault',
                                      'Employed',
                                      'CreditScore',
                                      'DriversLicense',
                                      'Citizen',
                                      'ZipCode',
                                      'Income',
                                      'Approved'),
                        skip = 1
                        )

credit_data <- 
  credit_data %>% 
  select(-X1) %>% 
  mutate(Approved = if_else(Approved == '+', 1, 0))

# Categorical Features

visualize_categorical <- function(data, response, predictor) {
  p <- 
    ggplot(data, aes(x = {{predictor}})) +
    geom_bar() +
    labs(
      y = "Frequency",
      title = paste0("Categorical variable: ", deparse(substitute(predictor)))) +
    theme_bw() +
    facet_grid(rows = vars({{response}}))
  ggsave(plot = p, filename = paste0("img/", deparse(substitute(predictor)), ".png"))
}

# Numerical Features

num_data1 <- 
  credit_data %>% 
  select(c(Age, Debt, YearsEmployed, CreditScore, Income, Approved)) %>% 
  mutate(Approved = as.factor(Approved))

ggpairs(num_data)

num_data2 <- 
  credit_data %>% 
  select(c(Age, Debt, YearsEmployed, CreditScore, Income, Approved))

ggpairs(num_data)