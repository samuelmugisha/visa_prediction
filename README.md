# Visa Approval Prediction: A classification model to facilitate the Visa approval process for foreign workers in the United States.

## Overview
This project focuses on building a machine learning classification model to facilitate the visa approval process for foreign workers in the United States. The goal is to assist the Office of Foreign Labor Certification (OFLC) in shortlisting candidates with higher chances of visa approval, thereby streamlining the review process which has become increasingly tedious due to a rising number of applications.

## Business Problem
Business communities in the U.S. face high demand for skilled human resources. The OFLC processes hundreds of thousands of labor certification applications annually, a number that continues to grow. Manually reviewing every case is inefficient. EasyVisa, an employer-hired firm, seeks a data-driven solution to predict visa certification status and recommend suitable applicant profiles, ensuring compliance with the Immigration and Nationality Act (INA) while addressing workforce shortages.

## Data
The dataset contains attributes of both the employee and the employer for visa applications. It comprises 25,480 rows and 12 columns. Key features include:

- `case_id`: ID of each visa application (dropped as unique identifier).
- `continent`: Employee's continent of origin.
- `education_of_employee`: Employee's education level (e.g., High School, Bachelor's, Master's, Doctorate).
- `has_job_experience`: Whether the employee has job experience (Y/N).
- `requires_job_training`: Whether the employee requires job training (Y/N).
- `no_of_employees`: Number of employees in the employer's company.
- `yr_of_estab`: Year the employer's company was established.
- `region_of_employment`: Intended region of employment in the US.
- `prevailing_wage`: Average wage for similar workers in the area.
- `unit_of_wage`: Unit of prevailing wage (Hourly, Weekly, Monthly, Yearly).
- `full_time_position`: Whether the position is full-time (Y/N).
- `case_status`: Target variable; indicates if the visa was Certified or Denied.

**Data Observations:**
- `no_of_employees` had negative values, which were corrected by taking absolute values.
- The target variable `case_status` is imbalanced, with 'Certified' being the majority class (approx. 66.8%).
- No missing values were found.

## Approach
1.  **Data Loading and Initial Inspection**: Loaded the `EasyVisa.csv` dataset and performed initial checks (`.head()`, `.tail()`, `.shape`, `.info()`, `.duplicated()`).
2.  **Data Cleaning**: Corrected negative values in `no_of_employees` by taking their absolute value. Dropped the `case_id` column as it was unique for each record.
3.  **Exploratory Data Analysis (EDA)**:
    *   **Univariate Analysis**: Explored distributions of individual features like `continent`, `education_of_employee`, `has_job_experience`, `unit_of_wage`, and `case_status` using bar plots.
    *   **Bivariate Analysis**: Investigated relationships between features and the target variable (`case_status`), as well as inter-feature relationships. This included analyzing `education_of_employee` vs. `case_status`, `continent` vs. `case_status`, `has_job_experience` vs. `case_status`, `prevailing_wage` vs. `case_status`, and `unit_of_wage` vs. `case_status`.
4.  **Data Pre-processing for Modeling**:
    *   Converted the `case_status` target variable into numerical format (1 for 'Certified', 0 for 'Denied').
    *   Applied one-hot encoding (`pd.get_dummies`) to all categorical features, dropping the first category to avoid multicollinearity.
    *   Split the data into training (70%), validation (27%), and test (3%) sets, ensuring stratification to maintain class distribution.
5.  **Model Building (Baseline)**:
    *   Trained several ensemble models (Bagging, Random Forest, Gradient Boosting, AdaBoost, XGBoost, Decision Tree) on the original (imbalanced) training data.
    *   Evaluated models using 5-fold stratified cross-validation and F1-score on the validation set.
6.  **Model Building with Resampling**:
    *   **Oversampling**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to balance the classes.
    *   **Undersampling**: Applied Random Under-sampling to the training data to balance the classes.
    *   Retrained and evaluated the same ensemble models on both oversampled and undersampled datasets.
7.  **Hyperparameter Tuning**: Performed RandomizedSearchCV with 5-fold cross-validation and F1-score as the metric for the best performing models (AdaBoost, Random Forest, Gradient Boosting, XGBoost) using the resampled data.
    *   Tuned AdaBoost using oversampled data.
    *   Tuned Random Forest using undersampled data.
    *   Tuned Gradient Boosting using oversampled data.
    *   Tuned XGBoost using oversampled data.

## Results
After comprehensive model training and hyperparameter tuning, the following performance was observed on the validation set:

| Model                                         | Accuracy | Recall   | Precision | F1 Score |
| :-------------------------------------------- | :------- | :------- | :-------- | :------- |
| Gradient Boosting (tuned, oversampled)        | 0.7072   | 0.7991   | 0.7709    | 0.7847   |
| XGBoost (tuned, oversampled)                  | 0.6972   | 0.9565   | 0.7000    | 0.8084   |
| AdaBoost (tuned, oversampled)                 | 0.7212   | 0.8067   | 0.7825    | 0.7944   |
| Random Forest (tuned, undersampled)           | 0.6995   | 0.7096   | 0.8164    | 0.7593   |

**Final Model Selection**: While XGBoost showed the highest validation F1 score (0.8084), its precision was notably low. **AdaBoost tuned with oversampled data** was selected as the final model due to its balanced performance, achieving an F1 score of 0.7944 on the validation set and an F1 score of **0.8153** on the unseen test data.

**Feature Importance (from Tuned AdaBoost Model):**
- `prevailing_wage` is the most important feature.
- `no_of_employees` and `yr_of_estab` are also highly influential.

## Tools & Technologies
-   **Programming Language**: Python
-   **Data Manipulation**: Pandas, NumPy
-   **Machine Learning**: Scikit-learn (for models, train-test split, metrics), XGBoost
-   **Imbalanced Data Handling**: imbalanced-learn (SMOTE, RandomUnderSampler)
-   **Data Visualization**: Matplotlib, Seaborn

## Key Learnings
**Profile for Visa Approval (Certified cases):**
-   **Education Level**: Higher education (Master's and Doctorate) correlates strongly with certification. Bachelor's degree holders also have good chances.
-   **Job Experience**: Applicants with job experience have significantly higher chances of visa certification (around 80%).
-   **Prevailing Wage**: The median prevailing wage for certified applications is slightly higher than for denied applications (around $72,000).
-   **Unit of Wage**: A 'Yearly' unit of wage is a strong indicator for certification (approx. 75% certification rate).
-   **Continent**: Applicants from Europe and Africa show higher certification rates (around 80% and 70% respectively), followed by Asia (around 60%).
-   **Region of Employment**: The Midwest region has the highest certification rate (around 75%), followed by the South region (around 70%). Specific educational demands vary by region (e.g., High School in South, Master's in Northeast, Doctorate in West).

**Profile for Visa Denial (Denied cases):**
-   **Education Level**: Applicants with only a High School education or no degree have a higher likelihood of denial.
-   **Job Experience**: Lack of job experience increases the chances of visa denial.
-   **Prevailing Wage**: Applications with lower prevailing wages (median around $65,000) are more likely to be denied.
-   **Unit of Wage**: An 'Hourly' unit of wage is associated with a lower certification rate (only 35%).
-   **Continent**: Applicants from South America, North America, and Oceania have relatively higher denial rates.

**Recommendations:**
*   Focus on applicants with higher education levels and prior job experience.
*   Pay attention to the proposed prevailing wage and its unit, favoring yearly wages.
*   While not discriminatory, historical data suggests higher approval rates for applicants from certain continents and regions, which could be considered for initial shortlisting within legal frameworks.
*   Collecting additional employer and employee information (e.g., sector, specialization, exact years of experience) could further enhance model accuracy and provide deeper insights.
