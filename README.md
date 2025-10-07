# Damage Detection API
This API detects various damages on a car from images or in real-time video streams. It can be implemented using either FASTAPI or Websocket.

## Token Creation and Database Access
To use the API, you need to generate a token. Each project directory contains a database_management folder that provides access to the PostgreSQL database hosted on the OVH cloud. The relevant table, `api_damagedetetion_user`, is located within the models_api schema.

Within the database_management folder, you will find the add_new_user_postgres.py file. This script requires the following information to create a new user:

```` 
    'first_name': 'Reza',
    'last_name': 'Reza',
    'phone': '1234567890',
    'email': 'reza@gmail.fr',
    'address': '123 Example St',
    'code_postal': '111111',
    'ville': 'ExampleCity',
    'pays': 'France',
````

## Token Generation and Validation
We use the `JWT` (JSON Web Token) method for token creation, with each token being valid for one month.

### Token Validation
The procedure for validating the token in both FASTAPI and Websocket is not similar.
- **FASTAPI**: The token is validated for each frame sent to the API.
- **Websocket**: The token is validated with the initial request. If the token is valid, the connection is established, and frames are accepted for processing.