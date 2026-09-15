# Hotel_Booking_Cancellation_prediction
Machine learning project that predicts hotel booking cancellations using Python, Pandas, Scikit-learn, Random Forest, and Streamlit.
#  Hotel Booking Cancellation Prediction

A machine learning project that predicts whether a hotel booking is likely to be **cancelled or not cancelled** based on reservation details such as lead time, deposit type, customer type, market segment, previous cancellations, booking changes, and special requests.

The project uses a **Random Forest Classifier** with a Scikit-learn preprocessing pipeline and provides both a **command-line prediction script** and an interactive **Streamlit web application**.

##  Project Overview

Hotel booking cancellations can affect room occupancy and revenue planning. This project applies machine learning to identify bookings that have a higher probability of cancellation.

The trained model takes booking information as input and predicts:

* `1` → Booking likely to be cancelled
* `0` → Booking likely to be not cancelled

The Streamlit application also displays the cancellation probability and provides basic insights based on the booking information.

##  Dataset

The project uses a dataset containing **700 hotel booking records** and **12 columns**.

### Features

| Feature                  | Description                                |
| ------------------------ | ------------------------------------------ |
| `lead_time`              | Number of days between booking and arrival |
| `adults`                 | Number of adults                           |
| `children`               | Number of children                         |
| `weekend_nights`         | Number of weekend nights                   |
| `weekday_nights`         | Number of weekday nights                   |
| `previous_cancellations` | Number of previous cancellations           |
| `booking_changes`        | Number of changes made to the booking      |
| `deposit_type`           | Type of deposit                            |
| `customer_type`          | Type of customer                           |
| `market_segment`         | Booking market segment                     |
| `special_requests`       | Number of special requests                 |
| `cancelled`              | Target variable                            |

##  Machine Learning Approach

The project uses a **Random Forest Classifier**.

### Workflow

1. Load the hotel booking dataset using Pandas
2. Separate features and target variable
3. Identify numerical and categorical features
4. Apply One-Hot Encoding to categorical features
5. Split the data into training and testing sets
6. Train a Random Forest classification model
7. Evaluate the model using accuracy and classification metrics
8. Save the trained model using Joblib
9. Generate a feature importance visualization
10. Use the trained model to predict new booking cancellations

### Model Configuration

```text
Algorithm: Random Forest Classifier
Number of Trees: 200
Test Size: 20%
Random State: 42
Class Weight: Balanced
```

##  Application Features

The Streamlit application provides three main sections:

###  Reservation Cancellation Predictor

Users can enter booking details such as:

* Lead time
* Number of adults and children
* Weekend and weekday nights
* Deposit type
* Customer type
* Market segment
* Previous cancellations
* Booking changes
* Special requests

The application then displays:

* Cancellation prediction
* Cancellation probability
* Booking fulfillment likelihood
* Basic revenue-management insights

### Feature Importance

Displays the most important features used by the Random Forest model when making predictions.

###  Historical Bookings Dataset

Displays the booking dataset along with basic statistics such as:

* Total bookings
* Overall cancellation rate
* Average lead time

##  Project Structure

```text
Hotel_Booking_Cancellation_Prediction_Sklearn/
│
├── data/
│   └── hotel_bookings.csv
│
├── app.py
├── train_model.py
├── predict.py
├── hotel_booking_cancellation_model.pkl
├── feature_importance.png
├── requirements.txt
├── HOW_TO_RUN.md
└── README.md
```

##  Technologies Used

* **Python**
* **Pandas** – Data loading and manipulation
* **Scikit-learn** – Machine learning and preprocessing
* **Random Forest** – Classification algorithm
* **Joblib** – Model serialization
* **Matplotlib** – Feature importance visualization
* **Streamlit** – Interactive web application

##  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Hotel_Booking_Cancellation_Prediction_Sklearn.git
```

Navigate to the project directory:

```bash
cd Hotel_Booking_Cancellation_Prediction_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

##  Train the Model

Run:

```bash
python train_model.py
```

This will:

* Train the Random Forest model
* Display the model accuracy
* Display the classification report
* Save the trained model
* Generate the feature importance chart

##  Make a Prediction

Run:

```bash
python predict.py
```

The program will ask for the booking details and return the predicted cancellation status and confidence.

##  Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

##  Output

The project produces:

* Trained machine learning model
* Cancellation prediction
* Prediction probability
* Classification report
* Feature importance visualization
* Interactive Streamlit dashboard

##  Project Objective

The main objective of this project is to demonstrate how machine learning can be applied to hotel reservation data to predict potential booking cancellations and provide useful information for better reservation and revenue management.

##  Disclaimer

This project is developed for **educational and demonstration purposes**. The dataset included with the project is a limited dataset and predictions should not be treated as a substitute for real-world hotel business decisions.

##  Author

**Vandana**


