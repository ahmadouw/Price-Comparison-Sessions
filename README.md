# What to Compare? Towards Understanding User Sessions on Price Comparison Platforms

This repository accompanies the paper titled "What to compare? Towards understanding user sessions on price comparison platforms." The repository contains the necessary notebooks, scripts and documentation needed to comprehend the setup and analysis of the clustering experiments discussed in the paper. Please note that all values are reported either relative or standardized since the dataset used in this research is not publicly available.

## Repository Structure
- **requirements.txt**:
  
- **clustering_main.ipynb**: This notebook contains the setup and execution of the clustering experiments. It provides steps for preprocessing the data, running clustering algorithms, and evaluating the resulting clusters. 

- **cluster_analysis.ipynb**: This notebook provides additional analysis of the clusters generated from the experiments in `clustering_main.ipynb`. It includes the verification questions to validate the clusters used internally and the selection of questions reported in the paper.

- **libs/**
  - **utils_clustering.py**: This script contains helper functions used in the clustering experiments.
  - **utils_analysis.py**: This script includes helper functions for analysis of the clusters.

## Expert Workshop Questions

As part of the research, an expert workshop was conducted to gather insights on user behavior and platform interaction and interpret the clusters. The questions posed in the first part of the workshop are listed below:

### General Perception
- How would you describe the current user base in general terms?
- What are the primary goals or needs you believe users have when visiting the platform?

### User Classification
- What kinds of users/sessions do you currently classify?
- What characteristics/behaviour is considered typical for each type?
- Do you notice different types of sessions from specific user groups over time?

### Engagement and Interaction
- How do you perceive user engagement with various features of the platform (e.g., search, product comparison, filtering)?
- What features do you believe are most critical for user retention?

### Challenges
- What are the biggest challenges you face in understanding and addressing user needs?
- Are there any user groups that you find particularly challenging to engage or satisfy?
