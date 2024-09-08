Overview

The machine learning application designed to estimate the price of a car and ipl team winning based on its features such as year, kilometers driven, company, and fuel type. This project leverages a RandomForestRegressor model and provides an interactive user interface using Streamlit.
Features

    Data Preprocessing: Cleans and prepares the dataset, handles non-numeric values, and converts categorical features to numerical representations.
    Machine Learning Model: Utilizes RandomForestRegressor for price prediction.
    Interactive Web Interface: Built with Streamlit, allowing users to input car details and receive price predictions.
    Performance Metrics: Evaluates model accuracy with Mean Absolute Error (MAE).

Demo

You can try out the Car Price Predictor application by accessing the live demo.
Installation

To run this project locally, follow these steps:

    Clone the repository:

    bash

git clone https://github.com/yourusername/car-price-predictor.git

Navigate to the project directory:

bash

cd car-price-predictor

Create a virtual environment (optional but recommended):

bash

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required dependencies:

bash

pip install -r requirements.txt

Run the Streamlit app:

bash

    streamlit run car_price_predictor.py
    stremalit run ipwinning.py
Usage

    Open the Streamlit app in your browser.
    Input the car details such as year, kilometers driven, company, and fuel type.
    Click the "Predict" button to get an estimated price for the car.

Project Structure

    car_price_predictor.py: Main script for training the model and creating the Streamlit application.
    requirements.txt: Lists the Python packages required to run the project.
    data/: Contains the dataset used for model training and evaluation.
    README.md: This file.

License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

    Internpe for providing the opportunity to work on this project.
    Streamlit for the interactive web framework.
    Scikit-learn for the machine learning tools.

Contact

For any questions or feedback, please reach out to satvikdubey268@gmail.com
