# Hotel Booking Cancellation Prediction

## Project Overview
This repository contains a machine learning project that aims to predict whether a hotel customer will cancel their reservation. By analyzing historical booking data, we build classification models that can predict cancellations based on new data.
## Business Problem
Hotel cancellations severely impact revenue and operational planning. If a hotel knows in advance that a guest is highly likely to cancel, they can take proactive measures, such as:

* Optimizing Overbooking: Safely accepting more reservations than physical rooms available.
* Targeted Communication: Sending automated reminders or requiring a deposit from high-risk guests.

## Dataset
* **Source:** [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
* **Description:** The dataset contains historical data for a City Hotel and a Resort Hotel, including details like booking lead time, deposit type, market segment, and special requests.
* **Target Variable:** `is_canceled` (1 = Canceled, 0 = Not Canceled)

## Tech Stack & Methods
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Yellowbrick, imblearn
* **Dimensionality Reduction:** PCA, t-SNE, RFE 
* **Machine Learning Algorithms:** Logistic Regression, k-Nearest Neighbors (k-NN), Support Vector Machines (SVM)
* **Oversampling:** SMOTE algorithms

## Key EDA Insights
* Market Segment matters: Bookings made through Online Travel Agencies (OTAs) have much higher cancellation rates compared to direct bookings.
<img width="863" height="448" alt="image" src="https://github.com/user-attachments/assets/d1f1ffee-e8e9-4381-930b-52e243793755" />
 
* The dataset is unbalanced (there are more class 0 than class 1 observations) which may cause problems in estimating models using accuracy
<img width="708" height="486" alt="image" src="https://github.com/user-attachments/assets/a5daffc2-e2c3-4279-9935-70a60ddb696f" />


## Modeling & Results
### Full Feature Set
| Model | Accuracy | F1-Score (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 79% | 66% | 56% |
| k-NN (k=3, weighted) | 83% | 78% | 79% |
| SVM (poly) | 83% | 74% | 66% |

### PCA Reduced Dataset (20 components)
| Model | Accuracy | F1-Score (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 77% | 63% | 53% |
| k-NN (k=3, weighted) | 83% | 78% | 78% |
| SVM (poly) | 82% | 71% | 61% |

### RFE Selected Features
| Model | Accuracy | F1-Score (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 79% | 66% | 57% |
|**k-NN (k=3, weighted)**| **85%** | **80%**| **80%** |
| SVM (poly) | 83% | 74% | 66% |


### Full Feature Set after SMOTE
| Model | Accuracy | F1-Score (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: |
|k-NN (k=9, weighted)| 82% | 78%| 84% |
| SVM (poly) | 84% | 79% | 82% |


**Key Takeaways:**
* The optimized k-NN model (trained on RFE selected columns) provided the best balance, successfully catching 80% of all cancellations.

* ROC-AUC: The k-NN model (trained on RFE selected columns) achieved an AUC of 0.90, proving it is highly capable of separating the two classes.

* Precision-Recall Trade-off: The PR curve (Average Precision = 0.84) shows that if the hotel wants to aggressively catch 85% of cancellations, they must accept a drop in precision to around 70%.
