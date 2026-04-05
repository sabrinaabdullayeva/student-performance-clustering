# Student Performance Clustering Analysis 

## Project Overview
This project applies unsupervised machine learning techniques to analyze and segment student performance data. The goal is to identify distinct student profiles based on their academic scores to help tailor educational support programs. 

## Dataset
* **Source:** Kaggle (`muhammadkhubaibahmad/student-performance-and-clustering-dataset`)
* **Size:** 300 rows × 16 features
* **Variables:** Academic scores (quizzes, midterms), attendance, and demographics.

## Methodology
1. **Data Preprocessing:** Handled missing values (median imputation), removed zero-variance columns, and applied feature scaling to prepare data for distance-based algorithms.
2. **Clustering Algorithms:** * **K-Means Clustering:** Optimal number of clusters (k=3) was determined using the Elbow and Silhouette methods.
   * **Hierarchical Clustering:** Used as a secondary method for robust profiling.
3. **Dimensionality Reduction:** Applied **Principal Component Analysis (PCA)** to visualize the multi-dimensional clusters in a 2D space.
4. **Evaluation:** Compared the two clustering algorithms using the **Adjusted Rand Index (ARI = 0.872)**, indicating a high consistency between the models.

## Key Findings
The analysis successfully divided students into 3 actionable profiles:
* **Cluster 1:** High-performing students across most subjects.
* **Cluster 2:** Medium-performing students (struggling with specific subjects).
* **Cluster 3:** Lower-performing students requiring overall academic support.

## Files in this Repository
* `97633_Abdullayeva.R`: Complete R script with the end-to-end data pipeline (importing, cleaning, modeling, and visualization).
* `97633_Abdullayeva.pdf`: Presentation slides summarizing the methodology, PCA visualizations, and business interpretations.

---
*Author: Sabrina Abdullayeva* Unicas 2026 | *Tools: R, ggplot2, factoextra, dplyr*
