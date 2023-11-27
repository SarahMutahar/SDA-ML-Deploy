import streamlit as st
import pandas as pd
import pickle

# Load the machine learning model from the pickle file
with open('pickled_model.pkl', 'rb') as file:
    model = pickle.load(file)

# List of available make and model values
make_model_values = ['Mercedes-Benz A 160', 'Mercedes-Benz EQE 350',
       'Mercedes-Benz A 45 AMG', 'Mercedes-Benz A 35 AMG',
       'Mercedes-Benz C 160', 'Mercedes-Benz CLA 180',
       'Mercedes-Benz A 220', 'Mercedes-Benz A 200',
       'Mercedes-Benz A 250', 'Mercedes-Benz A 180',
       'Mercedes-Benz GLC 220', 'Mercedes-Benz E 220',
       'Mercedes-Benz B 200', 'Mercedes-Benz A 150',
       'Mercedes-Benz V 220', 'Mercedes-Benz AMG GT', 'Mercedes-Benz EQS',
       'Mercedes-Benz A 140', 'Mercedes-Benz CL', 'Mercedes-Benz B 180',
       'Mercedes-Benz GLB 200', 'Mercedes-Benz E 350',
       'Mercedes-Benz CLA 200', 'Mercedes-Benz GLA 180',
       'Mercedes-Benz Viano', 'Mercedes-Benz E 53 AMG',
       'Mercedes-Benz SLK 200', 'Mercedes-Benz GLE 350',
       'Mercedes-Benz 220', 'Mercedes-Benz E 400',
       'Mercedes-Benz SLC 43 AMG', 'Mercedes-Benz SL 350',
       'Mercedes-Benz SLC 250', 'Mercedes-Benz SLK 350',
       'Mercedes-Benz SLS', 'Mercedes-Benz CLK 200',
       'Mercedes-Benz C 400', 'Mercedes-Benz SL 500',
       'Mercedes-Benz SL 65 AMG', 'Mercedes-Benz C 43 AMG',
       'Mercedes-Benz C 63 AMG', 'Mercedes-Benz SLK 250',
       'Mercedes-Benz SLK 300', 'Mercedes-Benz SLC 200',
       'Mercedes-Benz SLK 55 AMG', 'Mercedes-Benz S 63 AMG',
       'Mercedes-Benz E 500', 'Mercedes-Benz SLK',
       'Mercedes-Benz SL 55 AMG', 'Mercedes-Benz CLK 55 AMG',
       'Mercedes-Benz C 300', 'Mercedes-Benz SLK 280',
       'Mercedes-Benz E 450', 'Mercedes-Benz SLC 180',
       'Mercedes-Benz E 300', 'Mercedes-Benz C 200',
       'Mercedes-Benz E 200', 'Mercedes-Benz SL 63 AMG',
       'Mercedes-Benz S 560', 'Mercedes-Benz C 180',
       'Mercedes-Benz C 250', 'Mercedes-Benz C 220',
       'Mercedes-Benz E 250', 'Mercedes-Benz SL 600', 'Mercedes-Benz 250',
       'Mercedes-Benz CLK 350', 'Mercedes-Benz SLC 300',
       'Mercedes-Benz S 500', 'Mercedes-Benz SLR', 'Mercedes-Benz SL 400',
       'Mercedes-Benz CLK 280', 'Mercedes-Benz CLK 240',
       'Mercedes-Benz 200', 'Mercedes-Benz SLK 230',
       'Mercedes-Benz CL 500', 'Mercedes-Benz CLA 220',
       'Mercedes-Benz CLA 45 AMG', 'Mercedes-Benz CLS 500',
       'Mercedes-Benz S 400', 'Mercedes-Benz CL 63 AMG',
       'Mercedes-Benz CLS 350', 'Mercedes-Benz GLC 43 AMG',
       'Mercedes-Benz CLS 400', 'Mercedes-Benz CLK 63 AMG',
       'Mercedes-Benz CLS 250', 'Mercedes-Benz CLK 500',
       'Mercedes-Benz CLA 35 AMG', 'Mercedes-Benz GLC 63 AMG',
       'Mercedes-Benz CLS 320', 'Mercedes-Benz CLS 450',
       'Mercedes-Benz CLS 53 AMG', 'Mercedes-Benz CL 600',
       'Mercedes-Benz GLE 43 AMG', 'Mercedes-Benz CLA 250',
       'Mercedes-Benz GLC 200', 'Mercedes-Benz GLE 63 AMG',
       'Mercedes-Benz GLC 400', 'Mercedes-Benz CLS',
       'Mercedes-Benz GLE 400', 'Mercedes-Benz CLK',
       'Mercedes-Benz GLC 250', 'Mercedes-Benz CLK 320',
       'Mercedes-Benz S 65 AMG', 'Mercedes-Benz GLC 350',
       'Mercedes-Benz CLS 300', 'Mercedes-Benz GLE 300',
       'Mercedes-Benz E 50 AMG', 'Mercedes-Benz S 450',
       'Mercedes-Benz GLC 300', 'Mercedes-Benz GLE 450',
       'Mercedes-Benz ML 320', 'Mercedes-Benz ML 63 AMG',
       'Mercedes-Benz ML 500', 'Mercedes-Benz GL 500',
       'Mercedes-Benz GLE 580', 'Mercedes-Benz GLE 53 AMG',
       'Mercedes-Benz EQA', 'Mercedes-Benz G 400',
       'Mercedes-Benz GL 63 AMG', 'Mercedes-Benz ML 450',
       'Mercedes-Benz ML 300', 'Mercedes-Benz GLS 400',
       'Mercedes-Benz ML 350', 'Mercedes-Benz EQC 400',
       'Mercedes-Benz R 300', 'Mercedes-Benz GLA 45 AMG',
       'Mercedes-Benz GLK 350', 'Mercedes-Benz GLK 220',
       'Mercedes-Benz GL 350', 'Mercedes-Benz GLB 220',
       'Mercedes-Benz G 63 AMG', 'Mercedes-Benz GLA 220',
       'Mercedes-Benz GLK 250', 'Mercedes-Benz G 350',
       'Mercedes-Benz GLA 250', 'Mercedes-Benz G 500',
       'Mercedes-Benz GLE 250', 'Mercedes-Benz ML 400',
       'Mercedes-Benz EQB 350', 'Mercedes-Benz ML 250',
       'Mercedes-Benz GLA 200', 'Mercedes-Benz GLA 35 AMG',
       'Mercedes-Benz GLK 200', 'Mercedes-Benz G',
       'Mercedes-Benz G 55 AMG', 'Mercedes-Benz GLS 350',
       'Mercedes-Benz GL 420', 'Mercedes-Benz GLE 500',
       'Mercedes-Benz EQB 300', 'Mercedes-Benz GLB 250',
       'Mercedes-Benz GLB 180', 'Mercedes-Benz EQA 250',
       'Mercedes-Benz GLB 35 AMG', 'Mercedes-Benz X 250',
       'Mercedes-Benz GL 320', 'Mercedes-Benz GLS 63 AMG',
       'Mercedes-Benz 170', 'Mercedes-Benz ML 280',
       'Mercedes-Benz G 65 AMG', 'Mercedes-Benz Sprinter',
       'Mercedes-Benz C 350', 'Mercedes-Benz E 43 AMG',
       'Mercedes-Benz CLS 63 AMG', 'Mercedes-Benz C 32 AMG',
       'Mercedes-Benz E 63 AMG', 'Mercedes-Benz V 300',
       'Mercedes-Benz E 280', 'Mercedes-Benz EQE 43',
       'Mercedes-Benz V 250', 'Mercedes-Benz Vito',
       'Mercedes-Benz EQV 300', 'Mercedes-Benz B 220',
       'Mercedes-Benz B 250', 'Mercedes-Benz S 300',
       'Mercedes-Benz S 580', 'Mercedes-Benz S 600',
       'Mercedes-Benz C 280', 'Mercedes-Benz S 350',
       'Mercedes-Benz CLS 280', 'Mercedes-Benz S 55 AMG',
       'Mercedes-Benz E 240', 'Mercedes-Benz C 320',
       'Mercedes-Benz E 320', 'Mercedes-Benz E 230',
       'Mercedes-Benz EQE 500', 'Opel Corsa', 'Opel Astra', 'Opel Adam',
       'Opel Corsa-e', 'Opel Meriva', 'Opel Karl', 'Opel Agila',
       'Opel Insignia', 'Opel Vectra', 'Opel Antara', 'Opel Ampera',
       'Opel GT', 'Opel Cascada', 'Opel Tigra', 'Opel Speedster',
       'Opel Crossland X', 'Opel Grandland', 'Opel Grandland X',
       'Opel Mokka', 'Opel Mokka X', 'Opel Crossland', 'Opel Mokka-E',
       'Opel Vivaro', 'Opel Zafira Tourer', 'Opel Zafira Life',
       'Opel Combo Life', 'Opel Combo', 'Opel Zafira', 'Opel Movano',
       'Renault Megane', 'Renault Clio', 'Renault Laguna',
       'Renault Twingo', 'Renault ZOE', 'Renault Captur', 'Renault Twizy',
       'Renault Fluence', 'Renault Grand Scenic', 'Renault Avantime',
       'Renault Megane E-Tech', 'Renault Espace', 'Renault Kangoo Z.E.',
       'Renault Alpine A110', 'Renault Wind', 'Renault Coupe',
       'Renault Talisman', 'Renault Kangoo', 'Renault Arkana',
       'Renault Kadjar', 'Renault Alaskan', 'Renault Koleos',
       'Renault Scenic', 'Renault Trafic', 'Renault Master',
       'Renault Express', 'Renault Grand Espace', 'Renault P 1400',
       'Renault Latitude', 'Renault R 9', 'Renault Grand Modus',
       'Renault R 11', 'Peugeot 308', 'Peugeot 206', 'Peugeot 208',
       'Peugeot 207', 'Peugeot 1007', 'Peugeot 307', 'Peugeot 108',
       'Peugeot Rifter', 'Peugeot e-208', 'Peugeot 3008', 'Peugeot 508',
       'Peugeot Partner', 'Peugeot 107', 'Peugeot 106', 'Peugeot iOn',
       'Peugeot Expert', 'Peugeot 407', 'Peugeot RCZ', 'Peugeot 406',
       'Peugeot 2008', 'Peugeot 4007', 'Peugeot 4008', 'Peugeot 5008',
       'Peugeot e-2008', 'Peugeot Traveller', 'Peugeot Bipper',
       'Peugeot Ranch', 'Peugeot 607', 'Peugeot 301', 'Peugeot Boxer',
       'Fiat 500 Abarth', 'Fiat 595 Abarth', 'Fiat 500', 'Fiat Tipo',
       'Fiat Stilo', 'Fiat 500X', 'Fiat 500e', 'Fiat Punto', 'Fiat Panda',
       'Fiat Fiorino', 'Fiat 500C', 'Fiat New Panda', 'Fiat Grande Punto',
       'Fiat Punto Evo', 'Fiat Seicento', 'Fiat 124 Spider',
       'Fiat Barchetta', 'Fiat Spider Europa', 'Fiat Fullback',
       'Fiat Freemont', 'Fiat 500L', 'Fiat Strada', 'Fiat Sedici',
       'Fiat Talento', 'Fiat Croma', 'Fiat Qubo', 'Fiat Doblo',
       'Fiat Bravo', 'Fiat Multipla', 'SEAT Leon', 'SEAT Ibiza',
       'SEAT Toledo', 'SEAT Cordoba', 'SEAT Arona', 'SEAT Mii',
       'SEAT Altea XL', 'SEAT Tarraco', 'SEAT Arosa', 'SEAT Ateca',
       'SEAT Altea', 'SEAT Alhambra', 'SEAT Exeo', 'SEAT Leon e-Hybrid',
       'Skoda Octavia', 'Skoda Scala', 'Skoda Rapid/Spaceback',
       'Skoda Fabia', 'Skoda Kamiq', 'Skoda Citigo', 'Skoda Superb',
       'Skoda Kodiaq', 'Skoda 105', 'Skoda Karoq', 'Skoda Enyaq',
       'Skoda Yeti', 'Dacia Spring', 'Dacia Sandero', 'Dacia Logan',
       'Dacia Lodgy', 'Dacia Dokker', 'Dacia Duster', 'Dacia Jogger',
       'Dacia Break', 'Toyota Prius', 'Toyota Aygo X', 'Toyota Yaris',
       'Toyota Aygo', 'Toyota Corolla', 'Toyota Auris',
       'Toyota Urban Cruiser', 'Toyota iQ', 'Toyota Verso',
       'Toyota Avensis', 'Toyota Yaris Cross', 'Toyota Verso-S',
       'Toyota MR 2', 'Toyota C-HR', 'Toyota Supra', 'Toyota GT86',
       'Toyota Coaster', 'Toyota Celica', 'Toyota GR86',
       'Toyota Land Cruiser', 'Toyota RAV 4', 'Toyota Tacoma',
       'Toyota FJ Cruiser', 'Toyota Hilux', 'Toyota Highlander',
       'Toyota Proace', 'Toyota Tundra', 'Toyota Corolla Verso',
       'Toyota Sienna', 'Toyota Mirai', 'Toyota Camry', 'Toyota Prius+',
       'Nissan Leaf', 'Nissan Micra', 'Nissan Qashqai', 'Nissan Juke',
       'Nissan Pulsar', 'Nissan Note', 'Nissan Almera', 'Nissan Tiida',
       'Nissan Primera', 'Nissan Titan', 'Nissan 370Z', 'Nissan 350Z',
       'Nissan GT-R', 'Nissan Navara', 'Nissan Skyline', 'Nissan Ariya',
       'Nissan X-Trail', 'Nissan Murano', 'Nissan Terrano',
       'Nissan Pathfinder', 'Nissan Qashqai+2', 'Nissan King Cab',
       'Nissan Townstar', 'Nissan NV300', 'Nissan Primastar',
       'Nissan NV200', 'Nissan NV250', 'Nissan Almera Tino',
       'Nissan Evalia', 'Nissan Rogue', 'Nissan Cube', 'Nissan NV400',
       'Nissan E-NV200', 'Nissan Pixo', 'Nissan Gloria', 'Ford Fiesta',
       'Ford Focus', 'Ford Mondeo', 'Ford F 250', 'Ford Kuga',
       'Ford Ka/Ka+', 'Ford B-Max', 'Ford EcoSport', 'Ford Mustang',
       'Ford Bronco', 'Ford Puma', 'Ford Thunderbird', 'Ford Focus CC',
       'Ford Streetka', 'Ford GT', 'Ford Mustang Mach-E', 'Ford Explorer',
       'Ford Edge', 'Ford Ranger', 'Ford Maverick', 'Ford F 150',
       'Ford Ranger Raptor', 'Ford Tourneo Custom', 'Ford Transit Custom',
       'Ford S-Max', 'Ford Transit Connect', 'Ford Tourneo Connect',
       'Ford Galaxy', 'Ford Transit', 'Ford Tourneo Courier',
       'Ford Grand Tourneo', 'Ford Grand C-Max', 'Ford C-Max',
       'Ford Transit Courier', 'Hyundai i30', 'Hyundai i10',
       'Hyundai i20', 'Hyundai ELANTRA', 'Hyundai BAYON', 'Hyundai IONIQ',
       'Hyundai iX20', 'Hyundai Getz', 'Hyundai Terracan', 'Hyundai KONA',
       'Hyundai ACCENT', 'Hyundai Atos', 'Hyundai VELOSTER',
       'Hyundai TUCSON', 'Hyundai Coupe', 'Hyundai Genesis',
       'Hyundai Genesis Coupe', 'Hyundai Tiburon', 'Hyundai SANTA FE',
       'Hyundai iX35', 'Hyundai H-1', 'Hyundai IONIQ 5', 'Hyundai NEXO',
       'Hyundai Grand Santa Fe', 'Hyundai i40', 'Hyundai STARIA',
       'Hyundai Trajet', 'Hyundai SONATA', 'Hyundai H 350',
       'Hyundai KONA Elektro', 'Hyundai Matrix', 'Volvo V40', 'Volvo S80',
       'Volvo S60', 'Volvo S40', 'Volvo XC60', 'Volvo C30',
       'Volvo V40 Cross Country', 'Volvo C70', 'Volvo XC90', 'Volvo S90',
       'Volvo C40', 'Volvo V90 Cross Country', 'Volvo XC40',
       'Volvo V60 Cross Country', 'Volvo XC70', 'Volvo V90', 'Volvo V50',
       'Volvo V60', 'Volvo V70']

power_kw_values = sorted([
    75.0, 215.0, 310.0, 225.0, 100.0, 90.0, 140.0, 102.0, 66.0,
    280.0, 155.0, 85.0, 80.0, 118.0, 120.0, 143.0, 265.0, 160.0,
    125.0, 110.0, 173.0, 70.0, 410.0, 270.0, 320.0, 202.0, 375.0,
    165.0, 132.0, 130.0, 121.0, 115.0, 87.0, 228.0, 67.0, 79.0,
    101.0, 142.0, 112.0, 122.0, 81.0, 195.0, 103.0, 135.0, 235.0,
    350.0, 250.0, 245.0, 150.0, 200.0, 285.0, 390.0, 450.0, 287.0,
    232.0, 180.0, 170.0, 185.0, 430.0, 243.0, 335.0, 190.0, 145.0,
    210.0, 386.0, 336.0, 212.5, 220.0, 300.0, 136.0, 105.0, 184.0,
    330.0, 99.0, 168.0, 440.0, 139.0, 230.0, 176.0, 154.0, 95.0,
    147.0, 205.0, 55.0, 177.0, 51.0, 152.0, 74.0, 64.0, 63.0,
    73.0, 92.0, 96.0, 54.0, 77.0, 59.0, 43.0, 44.0, 206.0,
    104.0, 162.0, 88.0, 65.0, 114.0, 82.0, 194.0, 224.0, 108.0,
    231.0, 76.0, 294.0, 141.0, 191.0, 62.0, 221.0, 134.0, 133.0,
    60.0, 61.0, 107.0, 169.0, 128.0, 97.0, 91.0, 126.5, 129.0,
    83.0, 151.0, 56.0, 68.0, 117.0, 52.0, 53.0, 48.0, 50.0,
    187.0, 72.0, 57.0, 86.0, 119.0, 71.0, 78.0, 98.0, 201.0,
    192.0, 223.0, 175.0, 213.0, 116.0, 84.0, 163.0, 93.0, 49.0,
    199.0, 182.0, 98.5, 81.5, 113.0, 158.0, 33.0, 156.0, 137.0,
    172.0, 251.0, 100.5, 138.0, 241.0, 253.0, 419.0, 404.0
])

mileage_values = list(range(0, 100001, 5000))  # Values from 0 to 100,000 with a step of 10,000
mileage_values.append('Other')  # Add 'Other' option

# Function to predict price based on input features
def predict_price(make_model, power_kW, mileage, age, engine_size, car_type):
    input_data = pd.DataFrame({
        'make_model': [make_model],
        'power_kW': [power_kW],
        'mileage': [mileage],
        'age': [age],
        'engine_size': [engine_size],
        'type': [car_type]
    })

    prediction = model.predict(input_data)
    return prediction[0]  # Assuming the model returns a single prediction

# Streamlit app
def main():
    st.title('Car Price Prediction Application')

    # Sidebar with input features
    st.sidebar.header('Input Features')

    # Dropdown for make and model
    selected_make_model = st.sidebar.selectbox('Make and Model:', make_model_values + ['Other'])

    # Text input for custom make and model if "Other" is selected
    if selected_make_model == 'Other':
        make_model = st.sidebar.text_input('Enter Make and Model:')
    else:
        make_model = selected_make_model

    # Dropdown for power_kW
    selected_power_kw = st.sidebar.selectbox('Power (kW):', power_kw_values + ['Other'])
    if selected_power_kw == 'Other':
        power_kW = st.sidebar.number_input('Enter Power (kW):')
    else:
        power_kW = selected_power_kw

    # Slider for mileage
    selected_mileage = st.sidebar.select_slider('Mileage:', options=mileage_values, value=10000)

    # Text input for custom mileage if "Other" is selected
    if selected_mileage == 'Other':
        mileage = st.sidebar.number_input('Enter Mileage:')
    else:
        mileage = selected_mileage


    # Remaining input features
    #mileage = st.sidebar.number_input('Mileage:')
    age = st.sidebar.slider('Age:', 0, 200, 10)
    engine_size = st.sidebar.number_input('Engine Size:', value=1000)
    car_type = st.sidebar.selectbox('Car Type', ['Used', 'Pre-registered', 'Demonstration', "Employee's car"])

    # Predict button
    if st.sidebar.button('Predict Price'):
        # Perform prediction
        price_prediction = predict_price(make_model, power_kW, mileage, age, engine_size, car_type)
        st.sidebar.success(f'Predicted Price: {price_prediction:.2f} USD')

    # Display image at the center
    st.image('car_image.jpg', use_column_width=True, caption='Car Image')

if __name__ == '__main__':
    main()