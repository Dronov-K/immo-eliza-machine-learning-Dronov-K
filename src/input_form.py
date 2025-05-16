import streamlit as st
import pandas as pd
import numpy as np
from src.location_utils import build_location_structure


class PropertyInput:
    """
    Streamlit form for collecting user input related to a property's characteristics.

    This class provides a graphical interface for users to input detailed property
    information, which can then be used for price prediction or other analysis.
    """

    def __init__(self):
        """
        Initializes an empty dictionary to store input data from the user.

        The location_map is a nested dictionary structure loaded from CSV,
        mapping provinces to localities and localities to postal codes.
        """
        self.location_map = build_location_structure('data/updated_Kangaroo.csv')
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
        # Create two columns for better layout of inputs
        col1, col2 = st.columns(2)
        with col1:
            # Select property type (House or Apartment)
            self.input_data['type'] = st.selectbox(label='Type:',
                                                   options=['House', 'Apartment'],
                                                   index=None,
                                                   placeholder='--Select--')
            # Retrieve sorted list of provinces from location map
            provinces = sorted(self.location_map.keys())
            # Select province from the list
            selected_province = st.selectbox(
                'Province:',
                options=provinces,
                index=None,
                placeholder='--Select--'
            )
            self.input_data['province'] = selected_province

            # Determine available localities based on selected province
            if selected_province:
                # Localities for selected province
                localities = sorted(self.location_map[selected_province].keys())
            else:
                # If no province selected, gather all localities from all provinces
                all_localities = set()
                for loc in self.location_map.values():
                    all_localities.update(loc.keys())
                localities = sorted(all_localities)

            # Select locality if any are available
            selected_locality = st.selectbox(
                'Locality:',
                options=localities,
                index=None,
                placeholder='--Select--'
            ) if localities else None

            # Store selected locality
            self.input_data['locality'] = selected_locality

        with col2:
            # Define subtype options separately for houses and apartments
            house_subtypes = [
                'House', 'Villa', 'Bungalow', 'Farmhouse', 'Chalet', 'Country cottage',
                'Town house', 'Mansion', 'Manor house', 'Castle', 'Other property', 'Pavilion'
            ]

            apartment_subtypes = [
                'Apartment', 'Apartment block', 'Ground floor', 'Duplex', 'Flat studio',
                'Penthouse', 'Service flat', 'Loft', 'Triplex', 'Kot', 'Mixed use building'
            ]

            # Choose subtype options based on property type selected
            if self.input_data['type'] == 'House':
                subtype_options = sorted(house_subtypes)
            elif self.input_data['type'] == 'Apartment':
                subtype_options = sorted(apartment_subtypes)
            else:
                # If no type selected yet, show all subtypes
                subtype_options = sorted(house_subtypes + apartment_subtypes)
            # Select property subtype
            self.input_data['subtype'] = st.selectbox(label='Subtype:',
                                                      options=subtype_options,
                                                      index=None,
                                                      placeholder='--Select--'
                                                   )
            # Determine postcodes available based on selected province or locality
            if selected_province and selected_locality:
                # Get postcodes for exact province + locality
                postcodes = sorted(
                    self.location_map.get(selected_province, {}).get(selected_locality, [])
                )

            elif selected_province:
                # All postcodes in the selected province
                all_codes = set()
                for codes in self.location_map[selected_province].values():
                    all_codes.update(codes)
                postcodes = sorted(all_codes)

            elif selected_locality:
                # If province not selected, find postcodes for selected locality across provinces
                for province_data in self.location_map.values():
                    if selected_locality in province_data:
                        postcodes = sorted(province_data[selected_locality])
                        break
                else:
                    postcodes = []  # No postcodes found for locality

            else:
                # If neither province nor locality selected, show all postcodes
                all_codes = set()
                for province_data in self.location_map.values():
                    for codes in province_data.values():
                        all_codes.update(codes)
                postcodes = sorted(all_codes)

            # Select postcode if available
            self.input_data['postCode'] = st.selectbox(
                'Post Code:',
                options=postcodes,
                index=None,
                placeholder='--Select--'
            ) if postcodes else None

        st.markdown('---')

        # Living space details in two columns
        col3, col4 = st.columns(2)
        with col3:
            self.input_data['habitableSurface'] = self.number_input_nan(label='Habitable Surface (m²):')
            self.input_data['bathroomCount'] = self.number_input_nan(label='Bathrooms:')
        with col4:
            self.input_data['bedroomCount'] = self.number_input_nan(label='Bedrooms:')
            self.input_data['toiletCount'] = self.number_input_nan(label='Toilets:')

        st.markdown('---')

        # Building and energy characteristics in two columns
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

        # Boolean features (checkboxes), grouped separately for houses and apartments
        house_features = [
            ('Attic', 'hasAttic'), ('Basement', 'hasBasement'), ('Heat Pump', 'hasHeatPump'),
            ('Photovoltaic Panels', 'hasPhotovoltaicPanels'), ('Thermic Panels', 'hasThermicPanels'),
            ('Garden', 'hasGarden'), ('Swimming Pool', 'hasSwimmingPool'), ('Fireplace', 'hasFireplace'),
            ('Terrace', 'hasTerrace'), ('Office', 'hasOffice')
        ]
        apartment_features = [
            ('Lift', 'hasLift'), ('Air Conditioning', 'hasAirConditioning'), ('Armored Door', 'hasArmoredDoor'),
            ('Visiophone', 'hasVisiophone'), ('Terrace', 'hasTerrace'), ('Office', 'hasOffice')
        ]

        # Select appropriate feature list depending on property type
        if self.input_data.get('type') == 'House':
            features = house_features
        elif self.input_data.get('type') == 'Apartment':
            features = apartment_features
        else:
            # If type not selected, combine all features avoiding duplicates
            features = list(set(house_features + apartment_features))

        # Distribute checkboxes evenly across 3 columns
        cols = st.columns(3)
        for i, (label, key) in enumerate(features):
            # cols[i % 3] defines the column by the remainder of the division
            self.input_data[key] = cols[i % 3].checkbox(label)

        # Ensure all feature keys exist in input_data, default to False
        all_feature_keys = {key for _, key in house_features + apartment_features}
        for key in all_feature_keys:
            if key not in self.input_data:
                self.input_data[key] = False

        st.markdown('---')

        # Outdoor and structural features in two columns
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
        Returns the collected input data as a single-row pandas DataFrame.

        This method converts the dictionary of input values into a DataFrame
        suitable for further processing or feeding into ML models.

        :return: pandas DataFrame containing the user's input.
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
