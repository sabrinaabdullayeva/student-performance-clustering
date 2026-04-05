# =====================================================
# Student ID:97633
# Name: Sabrina Abdullayeva
# Assignment: Clustering Analysis (Student Performance Dataset)
# Dataset: muhammadkhubaibahmad/student-performance-and-clustering-dataset
# =====================================================

# =====================================================
# STEP 0 – LOAD REQUIRED PACKAGES
# =====================================================
# List of required packages for data import, manipulation, visualization, and clustering
packages <- c(
  "RKaggle", "ggplot2", "factoextra", "dplyr", "corrplot",
  "PerformanceAnalytics", "cluster", "tidyr", "scales",
  "caret", "mclust"
)

# Install missing packages automatically
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) install.packages(new_packages, dependencies = TRUE)

# Load all packages
invisible(lapply(packages, library, character.only = TRUE))

# Set seed for reproducibility
set.seed(123)

# =====================================================
# STEP 1 – DATA IMPORT
# =====================================================
# Import dataset from Kaggle using RKaggle
dataset_id <- "muhammadkhubaibahmad/student-performance-and-clustering-dataset"
student_df <- RKaggle::get_dataset(dataset_id)

# Quick overview of dataset structure and first 10 rows
print(dim(student_df))   # Number of rows and columns
summary(student_df)      # Summary statistics
str(student_df)          # Data types of each column
head(student_df, 10)     # First 10 observations

# =====================================================
# STEP 2 – DATA CLEANING AND PREPARATION
# =====================================================
# Remove columns with all missing values
all_na_cols <- names(student_df)[colSums(is.na(student_df)) == nrow(student_df)]
if(length(all_na_cols) > 0) student_df <- student_df %>% select(-all_of(all_na_cols))

# Keep only numeric columns (required for clustering)
student_df <- student_df %>% select(where(is.numeric))

# Remove columns with zero variance (no information)
zero_var_cols <- names(student_df)[sapply(student_df, function(x) sd(x, na.rm = TRUE) == 0)]
if(length(zero_var_cols) > 0) student_df <- student_df %>% select(-all_of(zero_var_cols))

# Impute remaining missing values using the median
na_counts <- colSums(is.na(student_df))
if(any(na_counts > 0)) {
  student_df <- student_df %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
}

# Optional: Boxplot to visually check for outliers in numeric variables
boxplot(student_df, main = "Boxplot of Student Performance Variables",
        col = "lightblue", las = 2, cex.axis = 0.7)

# =====================================================
# STEP 3 – CORRELATION ANALYSIS
# =====================================================
# Compute correlation matrix to identify highly correlated variables
cor_matrix <- cor(student_df, use = "pairwise.complete.obs")
cor_matrix[is.na(cor_matrix)] <- 0  # Replace any NA correlations with 0

# Visualize correlations using a hierarchical ordering
corrplot(cor_matrix, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix")

# Pairwise correlation plot with histograms
suppressWarnings(chart.Correlation(student_df, histogram = TRUE, pch = 19))

# Optional: Identify variables with very high correlation (>0.9)
high_corr <- findCorrelation(cor_matrix, cutoff = 0.9, names = TRUE)
if(length(high_corr) > 0) print(paste("Highly correlated variables:", paste(high_corr, collapse = ", ")))

# =====================================================
# STEP 4 – SCALING AND PCA
# =====================================================
# Scale numeric variables (mean=0, sd=1) for clustering
student_scaled <- scale(student_df)

# Perform Principal Component Analysis (PCA) to visualize variance and cluster separation
pca_res <- prcomp(student_scaled)  # Already scaled, so no need to center/scale again

# Scree plot showing proportion of variance explained by each principal component
fviz_eig(pca_res, addlabels = TRUE, barfill = "steelblue") +
  ggtitle("Variance Explained by Principal Components")

# PCA biplot: shows samples in PC space and variable contributions
fviz_pca_biplot(pca_res, geom = "point", repel = TRUE,
                title = "PCA Biplot - Student Performance")

# =====================================================
# STEP 5 – DETERMINE OPTIMAL NUMBER OF CLUSTERS
# =====================================================
set.seed(67)

# 5.1 Elbow Method: Plot WSS (Within-Cluster Sum of Squares) for k = 1 to 10
wss <- numeric(10)
for(k in 1:10) {
  km_model <- kmeans(student_scaled, centers = k, nstart = 25)
  wss[k] <- sum(km_model$withinss)
}

plot(1:10, wss, type = "b", pch = 12, frame = FALSE,
     xlab = "Number of Clusters (k)", ylab = "Within-Cluster Sum of Squares",
     main = "Elbow Method", col = "steelblue")

# 5.2 Silhouette Method: Evaluate cluster quality (average silhouette width)
sil_width <- numeric(10)
for(k in 2:10) {
  km_model <- kmeans(student_scaled, centers = k, nstart = 25)
  ss <- silhouette(km_model$cluster, dist(student_scaled))
  sil_width[k] <- mean(ss[, 3])
}

plot(2:10, sil_width[2:10], type = "b", pch = 12, frame = FALSE,
     xlab = "Number of Clusters (k)", ylab = "Average Silhouette Width",
     main = "Silhouette Method", col = "darkorange")

# Optional: Use factoextra functions to visualize WSS and silhouette methods
fviz_nbclust(student_scaled, kmeans, method = "wss") + labs(title = "Elbow Method")
fviz_nbclust(student_scaled, kmeans, method = "silhouette") + labs(title = "Silhouette Method")

# Choose optimal number of clusters based on plots
k_opt <- 3
cat("Optimal number of clusters selected:", k_opt, "\n")

# =====================================================
# STEP 6 – HIERARCHICAL CLUSTERING
# =====================================================
# Compute Euclidean distance matrix
dist_mat <- dist(student_scaled, method = "euclidean")

# Hierarchical clustering using Ward's method
hc_model <- hclust(dist_mat, method = "ward.D2")

# Plot dendrogram to visualize cluster hierarchy
plot(hc_model, labels = FALSE, main = "Hierarchical Clustering Dendrogram")
rect.hclust(hc_model, k = k_opt, border = 2:5)  # Highlight clusters

# Assign cluster labels
hc_clusters <- cutree(hc_model, k = k_opt)
table(hc_clusters)

# Visualize hierarchical clusters on PCA space
fviz_pca_biplot(pca_res, geom = "point", habillage = as.factor(hc_clusters),
                addEllipses = TRUE, ellipse.level = 0.95,
                title = "PCA - Hierarchical Clusters")

# =====================================================
# STEP 7 – K-MEANS CLUSTERING
# =====================================================
set.seed(123)
km_model <- kmeans(student_scaled, centers = k_opt, nstart = 25)
km_clusters <- km_model$cluster
print(km_model$size)  # Number of samples in each cluster

# Visualize K-means clusters on PCA space
fviz_pca_biplot(pca_res, geom = "point", habillage = as.factor(km_clusters),
                addEllipses = TRUE, ellipse.level = 0.95,
                title = "PCA - K-means Clusters")

# =====================================================
# STEP 8 – COMPARE CLUSTERING RESULTS
# =====================================================
# Contingency table to compare hierarchical vs K-means assignments
contingency_table <- table(Hierarchical = hc_clusters, Kmeans = km_clusters)
print("Contingency Table:")
print(contingency_table)

# Adjusted Rand Index (ARI) measures similarity between two clustering solutions
ari_value <- adjustedRandIndex(hc_clusters, km_clusters)
cat("Adjusted Rand Index:", round(ari_value, 3), "\n")

# =====================================================
# STEP 9 – CLUSTER PROFILING
# =====================================================
# Add cluster labels to dataset
student_final <- as.data.frame(student_scaled)
student_final$Cluster_KM <- as.factor(km_clusters)
student_final$Cluster_HC <- as.factor(hc_clusters)

# Calculate mean values per K-means cluster to profile clusters
cluster_summary <- student_final %>%
  group_by(Cluster_KM) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))

print("Cluster Summary (K-means):")
print(cluster_summary)

# Simple interpretation of clusters
# Cluster 1: Generally high-performing students
# Cluster 2: Medium performance; some weaker subjects
# Cluster 3: Lower-performing students overall

# Visualization of mean values per cluster
cluster_summary_long <- cluster_summary %>%
  pivot_longer(cols = -Cluster_KM, names_to = "Variable", values_to = "Mean_Value")

ggplot(cluster_summary_long, aes(x = Variable, y = Mean_Value, fill = Cluster_KM)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Average Scaled Values per K-means Cluster",
       x = "", y = "Mean (Scaled Value)") +
  theme(legend.position = "bottom")

# =====================================================
# STEP 10 – SAVE FINAL OUTPUT
# =====================================================
write.csv(student_final, "97633_Abdullayeva_clustered_data.csv", row.names = FALSE)

# --- END OF SCRIPT ---

