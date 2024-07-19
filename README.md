# Predicting Yelp Ratings

Welcome to the Predicting Yelp Ratings project! This project focuses on predicting the star ratings of restaurants in Las Vegas, Nevada, using various regression and classification models. The goal is to help businesses understand which factors are most important in attaining high star ratings and gaining popularity on Yelp.

## Project Overview

Yelp is a popular platform that publishes information and reviews of local businesses. Each review includes a star rating between 1 and 5, in addition to written comments. In this project, we use data from Yelp to build models that predict the star ratings of restaurants based on their attributes.

## Dataset
The dataset for this project is contained in two files:

- `yelp242a_train.csv` (Training Set: 6272 observations)
- `yelp242a_test.csv` (Test Set: 2688 observations)

Each observation contains the average star rating, number of reviews, and a list of attributes collected from the Yelp page of a particular restaurant in the Las Vegas area.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yelp-ratings-prediction.git
    ```
2. Change into the directory:
    ```bash
    cd yelp-ratings-prediction
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    ```bash
    # On Windows
    .\venv\Scripts\activate

    # On macOS and Linux
    source venv/bin/activate
    ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Load the dataset and preprocess it.
2. Define and fit the linear regression model using the provided code.
3. Analyze the results and visualize the findings.

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **Statsmodels**: Statistical modeling
- **Matplotlib**: Data visualization
- **Jupyter Notebook**: Interactive computing environment

## Modeling Approach

### Regression Models

1. **Linear Regression**: Built using all provided independent variables with missing values treated as explicit categories.
2. **Regression Tree**: Constructed using the CART algorithm, with complexity parameter selected through cross-validation.

### Classification Models

1. **Thresholding Regression Models**: Linear and regression tree models thresholded at a value of 4.
2. **Logistic Regression**: Built using all independent variables.
3. **Classification Tree**: Constructed using the CART algorithm, with hyperparameters selected through cross-validation.

### Evaluation Metrics

- **R-squared (R²)**
- **Mean Absolute Error (MAE)**
- **Accuracy**
- **True Positive Rate (TPR)**
- **False Positive Rate (FPR)**
- **ROC Curve and AUC**

### Results

The results indicate that more sophisticated models like Logistic Regression and Classification Trees perform better than simpler models like Baseline, Linear Regression, and Regression Trees in terms of accuracy, TPR, and FPR. Logistic Regression strikes a good balance between performance and complexity, making it the recommended model for this problem.

### Tips for High Yelp Ratings

Based on the analysis, here are three actionable tips for Las Vegas restaurants to achieve a high star rating on Yelp:

1. **Ensure Accessibility**: Being wheelchair accessible significantly increases the likelihood of receiving a star rating of four or above.
2. **Provide Reservation Options**: Offering reservation options can enhance the dining experience and potentially result in higher ratings.
3. **Emphasize Outdoor Seating**: Providing outdoor seating options can positively influence 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was developed as part of a course at the IEOR Department in the University od California, Berkeley. Special thanks to Prof. Paul Grigas and his team for their guidance and support.
