# Hotel Booking Cancellation Prediction

## Project Overview
This repository contains a machine learning project that aims to predict whether a hotel customer will cancel their reservation. By analyzing historical booking data, we build classification models that can predict cancellations based on new data.
## Business Problem
Hotel cancellations severely impact revenue and operational planning. If a hotel knows in advance that a guest is highly likely to cancel, they can take proactive measures. Because the cost of being wrong varies depending on the action taken, we evaluated models based on two distinct business strategies:

* Optimizing Overbooking: Safely accepting more reservations than physical rooms available. (Requires high Precision to avoid walking a guest if the model is wrong).

* Targeted Communication: Sending automated reminders or requiring a non-refundable deposit from high-risk guests. (Requires high Recall to catch as many potential cancellations as possible).

## Dataset
* **Source:** [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
* **Description:** The dataset contains historical data for a City Hotel and a Resort Hotel, including details like booking lead time, deposit type, market segment, and special requests.
* **Target Variable:** `is_canceled` (1 = Canceled, 0 = Not Canceled)

## Tech Stack & Methods
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, imblearn
* **Preprocessing:** MinMaxScaler, One-Hot Encoding
* **Dimensionality Reduction:** PCA, t-SNE, Recursive Feature Elimination with Cross-Validation (RFECV)
* **Machine Learning Algorithms:** Logistic Regression, k-Nearest Neighbors (k-NN), Support Vector Machines (SVM), Decision Tree
* **Oversampling:** SMOTE

## Key EDA Insights
* Market Segment matters: Bookings made through Online Travel Agencies (OTAs) have much higher cancellation rates compared to direct bookings.
<img width="600" height="311" alt="image" src="https://github.com/user-attachments/assets/d1f1ffee-e8e9-4381-930b-52e243793755" />
 
* Class Imbalance: The dataset is imbalanced (significantly more Class 0 than Class 1 observations). Because of this, standard Accuracy is a misleading metric. Models were evaluated primarily on F1-Score, Precision, and Recall for Class 1.

* Scaling & PCA: PCA struggled to compress the dataset efficiently. Because the data contains over 50 binary features (from one-hot encoding), standardizing the variance artificially inflated the importance of rare categorical variables. RFECV proved to be a much more effective dimensionality reduction technique for this specific feature space.****


## Modeling & Results
### Full Feature Set
| Model | Accuracy | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 81% | 72% | 80% | 65% |
| k-NN | 85% | 79% | 81% | 77% |
| SVM | 86% | 80% | 84% | 76% |
| Decision Tree | 85% | 80% | 77% | 84% |

### Full Feature Set after SMOTE
| Model | Accuracy | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
|k-NN | 83% | 78% | 73% | 84% |
| SVM | 85% | 81% | 77% | 85% |

### PCA Reduced Dataset (40 components)
*Note: PCA generally degraded performance because it is an unsupervised technique that discarded variance vital for predicting the target variable.*
| Model | Accuracy | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 78% | 67% | 79% | 58% |
| k-NN | 85% | 79% | 81% | 77% |
| SVM | 83% | 77% | 83% | 73% |


### PCA Reduced Dataset after SMOTE
| Model | Accuracy | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| k-NN | 82% | 78% | 74% | 82% |
| SVM | 83% | 78% | 74% | 82% |

### RFE Selected Features
*Note: RFECV successfully maintained the predictive power of the full dataset while dropping uninformative features, resulting in a more efficient, lighter model.*
| Model | Accuracy | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 81% | 72% | 81% | 65% |
| k-NN | 86% | 81% | 82% | 80% |
| SVM (poly) | 86% | 80% | 84% | 76% |
| Decision Tree | 85% | 81% | 77% | 84% |

### RFE Selected Features after SMOTE
| Model | Accuracy | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| k-NN | 84% | 80% | 75% | 85% |
| SVM | 85% | 80% | 77% | 85% |

## Business Solutions & Model Selection

There is no single "best" model; the choice depends entirely on the specific financial risk the hotel is trying to mitigate.

### Strategy 1: Optimizing Overbooking
* **The Risk:** A False Positive (predicting a cancellation, giving the room away, and the original guest shows up).
* **The Champion Model:** **RFECV + SVM (Without SMOTE)**
* **Why:** This model prioritizes playing it safe. It achieves a high Precision of **84%**, meaning that when it flags a guest as a cancellation, it is highly likely to be correct, making overbooking much safer.

### Strategy 2: Implementing Deposits & Reminders
* **The Risk:** A False Negative (missing a cancellation). This results in an empty room and lost revenue. A False Positive here is low-risk, as asking a guest who intends to stay for a deposit causes minimal friction.
* **The Champion Model:** **RFECV + SVM (With SMOTE)**
* **Why:** This model prioritizes catching as many cancellations as possible. By training on SMOTE-balanced data, it achieves a Recall of **85%**, ensuring the hotel identifies the vast majority of flight-risk guests so they can secure revenue via deposits.

### Flexibility & Robustness (ROC/AUC)
The Precision-Recall and ROC curves for the SVC model on RFE Selected Features demonstrate its high robustness (**ROC-AUC = 0.93**).

<p align="center">
  <img width="45%" alt="Precision-Recall Curve" src="https://github.com/user-attachments/assets/ef8d357d-be7f-4436-83a3-5ab1b771a05f" />
  <img width="45%" alt="ROC Curve" src="https://github.com/user-attachments/assets/249ae245-1b31-4199-88fb-a90b97a20834" />
</p>

This allows the hotel to flexibly adjust its decision threshold based on current needs:
* **Aggressive approach:** Catch **90%** of all cancellations to enforce deposits (while maintaining a ~70% precision rate).
* **Conservative approach:** Safely overbook by catching **70%** of cancellations with a **90%** precision rate, minimizing the risk of walking a guest.

---
