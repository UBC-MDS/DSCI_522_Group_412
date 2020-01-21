# author: Alistair Clark
# date: 2020-01-21

"This script creates exploratory data visualizations.

Usage: visualizations.R --input_file<input_file> --output_file<output_file>

Options:
--input_file<input_file>    A path/filename for the input data.
--output_file<output_file>  A path/filename for the created figure.

" -> doc

library(docopt)
library(GGally)
library(tidyverse)

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
  select(-X1)

visualize_categorical <- function(data, response, target) {
  p <- 
    ggplot(data, aes(x = {{target}})) +
    geom_bar() +
    labs(
      y = "Frequency",
      title = paste0("Categorical variable: ", deparse(substitute(target)))) +
    theme_bw() +
    facet_grid(rows = vars({{response}}))
  ggsave(plot = p, filename = paste0("img/", deparse(substitute(target)), ".png"))
}

visualize_categorical(credit_data, Approved, Sex)


paste0(deparse(substitute(target)), ".png")