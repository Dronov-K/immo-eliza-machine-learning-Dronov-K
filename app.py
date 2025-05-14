import streamlit as st
import joblib
from src.input_form import PropertyInput
from src.app_styles import AppStyle


@st.cache_resource
def load_model(path: str):
    """
    Loads a trained machine learning model from the specified file path.
    This function uses Streamlit's caching mechanism to avoid reloading the model on every run.

    :param path: Path to the saved model file.
    :return: Loaded model object.
    """
    return joblib.load(path)


class PricePredictor:
    """
    A Streamlit-based interface for predicting property prices using a trained model.

    This class loads the model, collects user input through a form, and displays
    the predicted price based on the input features.
    """

    def __init__(self, model_path: str):
        """
        Initializes the PricePredictor with a model loaded from the given path.

        :param model_path: Path to the trained machine learning model.
        """
        self.model = load_model(model_path)

    def run(self) -> None:
        """
        Runs the Streamlit app interface.
        Renders the input form, waits for user interaction, and displays
        the predicted price when the user clicks the button.

        :return: None
        """
        AppStyle.apply_background_color('#87CEEB')
        AppStyle.center_title('Property Price Prediction')

        property_input = PropertyInput()
        property_input.render()
        input_data = property_input.get_input_data()

        col1, col2, col3 = st.columns([2, 1, 2])  # for centered the button
        with col2:
            if st.button('Predict the price'):
                prediction = self.model.predict(input_data)[0]
                AppStyle.show_prediction_block(prediction)

        AppStyle.add_footer('Konstantin Dronov')
