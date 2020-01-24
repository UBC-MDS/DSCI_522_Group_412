# author: Alistair Clark
# date: 2020-01-22

"Creates eda plots for the pre-processed training data from the credit approval data.
Saves the plots as png files.

Usage: src/visualizations.R --train=<train> --out_dir=<out_dir>

Options:
--train=<train>      Path (including filename) to training data (which needs to be saved as a csv file).
--out_dir=<out_dir> Path to directory where the figures will be saved.
" -> doc

library(docopt)
library(GGally)
library(tidyverse)
library(cowplot)

opt <- docopt(doc)

main <- function(train, out_dir) {
  # load data
  df <- read_csv("data/clean-train.csv")
  categorical_cols = c('Sex',
                       'Married',
                       'BankCustomer',
                       'EducationLevel',
                       'Ethnicity',
                       'DriversLicense',
                       'Citizen',
                       'Employed',
                       'PriorDefault')
  numerical_cols = c('Age',
                     'Debt',
                     'YearsEmployed',
                     'CreditScore',
                     'Income',
                     'Approved')
  
  # Plot categorical features
  plot_lst <- list()
  for (col in categorical_cols) {
    g <- visualize_categorical(df, Approved, col)
    plot_lst[[col]] <- g
  }
  p1 <- plot_grid(plotlist = plot_lst)
  
  # Plot numerical features
  num_data <- 
    df %>% 
    select(c(numerical_cols)) %>% 
    mutate(Approved = as.factor(Approved))
  p2 <- ggpairs(num_data)
  
  # Save figures
  ggsave(plot = p1,
         filename = paste0(out_dir,"categorical.png"),
         width = 10,
         height = 10)
  ggsave(plot = p2,
         filename = paste0(out_dir,"numerical.png"))
  
}


visualize_categorical <- function(data, response, predictor) {
  ggplot(data, aes(x = !!sym(predictor))) +
  geom_bar() +
  labs(
    y = "Frequency",
    title = predictor) +
  theme_bw() +
  facet_grid(cols = vars({{response}}))
}

main(opt[["--train"]], opt[["--out_dir"]])