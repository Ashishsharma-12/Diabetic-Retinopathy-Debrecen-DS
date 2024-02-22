setwd(dirname(file.choose()))
getwd()

pacman::p_load(  # Use p_load function from pacman
  caret,         # Train/test functions
  e1071,         # Machine learning functions
  magrittr,      # Pipes
  pacman,        # Load/unload packages
  rattle,        # Pretty plot for decision trees
  rio,           # Import/export data
  tidyverse,      # So many reasons
  cluster,
  car,
  ggplot2,
  GGally
)

# Load Dataset from UCI

# Borzooei,Shiva and Tarokhian,Aidin. (2023). 
# Differentiated Thyroid Cancer Recurrence. 
# UCI Machine Learning Repository. 


df <-  import("./datasets/Diabetic Retinopathy Debrecen.csv") %>% as_tibble() 

summary(df)

# Factor count in numbers

df %>% 
  pull(Target) %>%  
  as_factor() %>% 
  fct_count() 

# Alternative method in percentage

round(prop.table(table(df$Target)) * 100, digits = 1)

# Changing 0,1 into Labels, and Rename class variable as `y` and changing it into a factor variable

df %<>% 
  rename(y = Target) %>%   
  mutate(
    y = ifelse(
      y == 0, 
      "Healthy", 
      "PD"
    )
  ) %>% 
  mutate(y = as_factor(y))

glimpse(df)

# Check if any na values

apply(df, MARGIN = 2, FUN = function(x) sum(is.na(x)))


library(janitor) #to check if there are any constants
df %<>% remove_constant()

na.omit(df)

# ``````````````````````````````````````````````
# Visualization
# ``````````````````````````````````````````````

attach(df)

boxplot(df)
boxplot(Attr12)
boxdata <- boxplot(Attr12)
print(boxdata)

for(i in 1:length(boxdata$group)){
  #add text to the boxplot
  text(boxdata$group[i], boxdata$out[i],
       which(Attr12==boxdata$out[i]),pos=4, cex=1)}

# feature extraction

df %>%  
  gather(var, val,-y) %>%  # Gather key value pairs
  ggplot(aes(x = val, group = y, fill = y)) +
  geom_histogram(binwidth = 1) +
  facet_wrap(~var, ncol = 3) +
  theme(legend.position = "bottom")

# Good Features

df %>%  
  select(Attr3,Attr4, Attr5, Attr6, Attr7, Attr8, Attr9,Attr10,Attr11,y) %>% 
  gather(var, val, -y) %>%  # Gather key value pairs
  ggplot(aes(x = val, group = y, fill = y)) +
  geom_histogram(binwidth = 1) +
  facet_wrap(~var, ncol = 3) +
  theme(legend.position = "bottom")

# Not well dispersed

df %>%  
  select(Attr1,Attr13, Attr14, Attr15, Attr16, Attr17, Attr19, Attr2, y) %>% 
  gather(var, val, -y) %>%  # Gather key value pairs
  ggplot(aes(x = val, group = y, fill = y)) +
  geom_histogram(binwidth = 1) +
  facet_wrap(~var, ncol = 3) +
  theme(legend.position = "bottom")


# Randomize the data

df <- df[sample(1:nrow(df)), ] 

# ``````````````````````````````
# Normalization
# ``````````````````````````````

# Using log mthod
log <- df %>% select(-y) %>% log()
boxplot(log)

# Using scale mthod
sc <- df %>% select(-y) %>% scale()
boxplot(sc)

# After scalling following features are not good

df %<>%
  select(-c('Attr13','Attr14','Attr15','Attr16'))


# min-max scaling 

mm <- df %>% select(-y) %>% apply( MARGIN = 2, FUN = function(x) (x - min(x))/diff(range(x)))
boxplot(mm)


# z-score

z1 <- df %>% select(-y) %>% apply( MARGIN = 2, FUN = function(x) (x - mean(x))/sd(x))
z2 <- df %>% select(-y) %>% apply( MARGIN = 2, FUN = function(x) (x - mean(x))/(2*sd(x)))

boxplot (z1, main = "Z-score, 1 sd")
boxplot (z2, main = "Z-score, 2 sd")

# Softmax

library(DMwR2)
# help(SoftMax)

sm <- df %>% select(-y) %>% apply( MARGIN = 2, FUN = function(x) (SoftMax(x,lambda = 1, mean(x), sd(x))))
boxplot (sm)


nrow(df)

set.seed(1)  # You can use any number here

# # Split data into training (trn) and testing (tst) sets 80-20 split

trn <- sm[1:920, ]
tst <- sm[921:1151, ]

# create labels (from first column) for training and test data
trn_y <- df[1:920, c('y')] 
tst_y <- df[921:1151, c('y')] 

trn_y <- trn_y %>% pull(y) %>% as_factor()
tst_y <- tst_y %>% pull(y) %>% as_factor()

# round prop to check the percentile distribution

round(prop.table(table(trn_y)) * 100, digits = 1)
round(prop.table(table(tst_y)) * 100, digits = 1)

# Alternative for train test split

# df %<>% mutate(ID = row_number())  # Add row ID
# trn <- df %>%                      # Create trn
#   slice_sample(prop = .70)         # 70% in trn
# tst <- df %>%                      # Create tst
#   anti_join(trn, by = "ID") %>%    # Remaining data in tst
#   select(-ID)                      # Remove id from tst
# trn %<>% select(-ID)               # Remove id from trn
# df %<>% select(-ID)                # Remove id from df

# trny <- trn %>% pull(y) %>% as.factor()
# trn %<>% select(-Attr1,-Attr13, -Attr14, -Attr15, -Attr16, -Attr17, -Attr19,-Attr2, -y)
# 
# tsty <- tst %>% pull(y) %>% as.factor()
# tst %<>% select(-Attr1,-Attr13, -Attr14, -Attr15, -Attr16, -Attr17, -Attr19,-Attr2,-y)

# ````````````````````````````
# Saving files in RDS format
# ````````````````````````````

df  %>% saveRDS("datasets/data.rds")
trn %>% saveRDS("datasets/data_trn.rds")
tst %>% saveRDS("datasets/data_tst.rds")
trn_y %>% saveRDS("datasets/data_trn_y.rds")
tst_y %>% saveRDS("datasets/data_tst_y.rds")
