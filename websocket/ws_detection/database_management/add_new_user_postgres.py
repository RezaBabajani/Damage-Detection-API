import psycopg2
from config import connect
import uuid
import time
import jwt
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = str(os.getenv('SECRET_KEY'))
# Expire the token for one month
expire_time = time.time() + 60 * 60 * 24 * 30
payload = {
        'exp': expire_time,
        'SECRET_KEY': SECRET_KEY,
        'uuid': str(uuid.uuid4()),
        'email': 'r.babajani@omia.fr'
}

token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

new_user_info = {
    'first_name': 'Reza',
    'last_name': 'BABAJANIVALASHEDI',
    'phone': '1234567890',
    'email': 'r.babajani@omia.fr',
    'address': '4, Rue Gilbert Defer',
    'code_postal': '31300',
    'ville': 'Toulouse',
    'pays': 'France',
    'token': token,
    'date_join': time.strftime('%Y-%m-%d %H:%M:%S')
}


def add_new_user(database, user_info):
    """ Add a new user to the models_api schema and the api_plaque table """
    try:
        cursor = database.cursor()
        # insert a new user into the users table
        cursor.execute("INSERT INTO models_api.api_damagedetection_user "
                       "(first_name, last_name, phone, email, "
                       "address, code_postal, ville, pays, token, date_join) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                       (user_info['first_name'], user_info['last_name'],
                        user_info['phone'], user_info['email'],
                        user_info['address'], user_info['code_postal'],
                        user_info['ville'], user_info['pays'],
                        user_info['token'], user_info['date_join']))
        database.commit()
        cursor.close()
        print(f"User {user_info['first_name']} {user_info['last_name']} added to the api_licence_user table.")
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


if __name__ == '__main__':
    conn = connect()
    add_new_user(conn, new_user_info)
