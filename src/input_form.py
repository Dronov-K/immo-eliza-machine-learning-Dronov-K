import streamlit as st
import pandas as pd
import numpy as np


class PropertyInput:
    """
    Streamlit form for collecting user input related to a property's characteristics.

    This class provides a graphical interface for users to input detailed property
    information, which can then be used for price prediction or other analysis.
    """

    def __init__(self):
        """
        Initializes an empty dictionary to store input data from the user.
        """
        self.input_data = {}

    def render(self):
        """
        Renders a multi-section input form in the Streamlit app.

        The form includes:
        - General property information (type, subtype, province, locality, post code)
        - Living space details (surface, bedrooms, bathrooms, toilets)
        - Building and energy characteristics (condition, construction year, heating type, EPC score)
        - Boolean features (e.g., hasAttic, hasLift, hasGarden, etc.)
        - Outdoor and structural features (land surface, terrace surface, parking counts, flood zone type)

        All input is saved to `self.input_data`.

        :return: None
        """

        col1, col2 = st.columns(2)
        with col1:
            self.input_data['type'] = st.selectbox(label='Type:',
                                                   options=['House', 'Apartment'],
                                                   index=None,
                                                   placeholder='--Select--')

            self.input_data['province'] = st.selectbox(label='Province:',
                                                       options=['West Flanders', 'Antwerp', 'East Flanders', 'Brussels',
                                                                'Hainaut',
                                                                'Liege',
                                                                'Flemish Brabant', 'Limburg', 'Walloon Brabant',
                                                                'Namur',
                                                                'Luxembourg'],
                                                       index=None,
                                                       placeholder='--Select--'
                                                       )
            self.input_data['postCode'] = self.number_input_nan(label='Post Code:')
        with col2:
            self.input_data['subtype'] = st.selectbox(label='Subtype:',
                                                      options=['House', 'Apartment', 'Villa', 'Apartment block',
                                                               'Mixed use building', 'Ground floor', 'Duplex',
                                                               'Flat studio', 'Penthouse', 'Exceptional property',
                                                               'Mansion',
                                                               'Town house', 'Service flat', 'Bungalow', 'Kot',
                                                               'Country cottage',
                                                               'Farmhouse', 'Loft', 'Chalet', 'Triplex', 'Castle',
                                                               'Other property',
                                                               'Manor house', 'Pavilion'],
                                                      index=None,
                                                      placeholder='--Select--'
                                                      )
            self.input_data['locality'] = st.text_input(label='Locality:')

        st.markdown('---')

        col3, col4 = st.columns(2)
        with col3:
            self.input_data['habitableSurface'] = self.number_input_nan(label='Habitable Surface (m²):')
            self.input_data['bathroomCount'] = self.number_input_nan(label='Bathrooms:')
        with col4:
            self.input_data['bedroomCount'] = self.number_input_nan(label='Bedrooms:')
            self.input_data['toiletCount'] = self.number_input_nan(label='Toilets:')

        st.markdown('---')

        col5, col6 = st.columns(2)
        with col5:
            self.input_data['buildingCondition'] = st.selectbox(label='Building Condition:',
                                                                options=['Good', 'As new', 'To renovate',
                                                                         'To be done up',
                                                                         'Just renovated',
                                                                         'To restore'],
                                                                index=None,
                                                                placeholder='--Select--'
                                                                )
            self.input_data['kitchenType'] = st.selectbox(label='Kitchen Type:',
                                                          options=['Installed', 'Hyper equipped', 'Semi equipped',
                                                                   'Not installed',
                                                                   'Usa hyper equipped', 'Usa installed',
                                                                   'Usa semi equipped', 'Usa uninstalled'],
                                                          index=None,
                                                          placeholder='--Select--'
                                                          )
            self.input_data['epcScore'] = st.selectbox(label='EPC Score:',
                                                       options=['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G'],
                                                       index=None,
                                                       placeholder='--Select--'
                                                       )

        with col6:
            self.input_data['buildingConstructionYear'] = self.number_input_nan(label='Construction Year:')
            self.input_data['heatingType'] = st.selectbox(label='Heating Type:',
                                                          options=['Gas', 'Fueloil', 'Electric', 'Pellet', 'Wood',
                                                                   'Solar', 'Carbon'],
                                                          index=None,
                                                          placeholder='--Select--'
                                                          )
        st.markdown('---')

        cols = st.columns(3)
        features = [
            ('Attic', 'hasAttic'), ('Basement', 'hasBasement'), ('Lift', 'hasLift'),
            ('Heat Pump', 'hasHeatPump'), ('Photovoltaic Panels', 'hasPhotovoltaicPanels'),
            ('Thermic Panels', 'hasThermicPanels'), ('Garden', 'hasGarden'),
            ('Air Conditioning', 'hasAirConditioning'), ('Armored Door', 'hasArmoredDoor'),
            ('Visiophone', 'hasVisiophone'), ('Office', 'hasOffice'),
            ('Swimming Pool', 'hasSwimmingPool'), ('Fireplace', 'hasFireplace'),
            ('Terrace', 'hasTerrace')
        ]
        # evenly distributed across columns from top to bottom
        for i, (label, key) in enumerate(features):
            # cols[i % 3] defines the column by the remainder of the division
            self.input_data[key] = cols[i % 3].checkbox(label)

        st.markdown('---')

        col7, col8 = st.columns(2)
        with col7:
            self.input_data['landSurface'] = self.number_input_nan(label='Land Surface (m²):')
            self.input_data['terraceSurface'] = self.number_input_nan(label='Terrace Surface (m²):')
            self.input_data['parkingCountIndoor'] = self.number_input_nan(label='Indoor Parking:')
            self.input_data['facedeCount'] = self.number_input_nan(label='Facade Count:')

        with col8:
            self.input_data['gardenSurface'] = self.number_input_nan(label='Garden Surface (m²):')
            self.input_data['floodZoneType'] = st.selectbox(label='Flood Zone Type:',
                                                            options=['Non flood zone', 'Possible flood zone',
                                                                     'Recognized flood zone',
                                                                     'Recognized n circumscribed flood zone',
                                                                     'Circumscribed waterside zone',
                                                                     'Circumscribed flood zone',
                                                                     'Possible n circumscribed flood zone',
                                                                     'Possible n circumscribed waterside zone',
                                                                     'Recognized n circumscribed waterside flood zone'],
                                                            index=None,
                                                            placeholder='--Select--'
                                                            )
            self.input_data['parkingCountOutdoor'] = self.number_input_nan(label='Outdoor Parking:')

        st.markdown('---')

    def get_input_data(self) -> pd.DataFrame:
        """
        Returns the user input as a single-row pandas DataFrame.

        :return: DataFrame containing the collected property input.
        """
        return pd.DataFrame([self.input_data])

    @staticmethod
    def number_input_nan(label: str, min_value: int = 0, step: int = 1, format: str = "%d",
                         zero_as_nan: bool = True) -> float | int:
        """
        Streamlit number_input wrapper that returns np.nan instead of 0 if zero_as_nan is True.

        :param label: Label for the input
        :param min_value: Minimum value allowed (default: 0)
        :param step: Step size (default: 1)
        :param format: Number display format (default: '%d')
        :param zero_as_nan: If True, 0 will be converted to np.nan
        :return: int or np.nan
        """
        value = st.number_input(label=label,
                                min_value=min_value,
                                step=step,
                                format=format,
                                )
        return np.nan if zero_as_nan and value == 0 else value
