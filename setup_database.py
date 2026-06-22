import sqlite3
import csv
from pathlib import Path
import sys

def create_database(db_path: Path):
    """Create SQLite database and tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create hotel_reservations table with cleaned column names
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hotel_reservations (
            Reservation_ID INTEGER PRIMARY KEY,
            Guest_ID INTEGER,
            First_Name TEXT,
            Last_Name TEXT,
            Gender TEXT,
            Email TEXT,
            Phone TEXT,
            Nationality TEXT,
            Birthdate TEXT,
            Address TEXT,
            City TEXT,
            Postal_Code TEXT,
            Country TEXT,
            Check_in_Date TEXT,
            Check_out_Date TEXT,
            Room_Number INTEGER,
            Floor_Number INTEGER,
            Room_Type TEXT,
            Adults INTEGER,
            Children INTEGER,
            Total_Nights INTEGER,
            Total_Amount REAL,
            Payment_Status TEXT,
            Booking_Date TEXT,
            Check_in_Time TEXT,
            Check_out_Time TEXT,
            Stay_Duration INTEGER
        )
    ''')

    # Create pir_sensor_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pir_sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            room_number INTEGER,
            pir_motion INTEGER,
            room_state TEXT,
            persona TEXT,
            adults INTEGER,
            children INTEGER,
            guest_id INTEGER
        )
    ''')

    # Create temperature_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS temperature_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            room_number INTEGER,
            floor INTEGER,
            facade TEXT,
            room_type TEXT,
            size_m2 REAL,
            outside_temp REAL,
            room_temp REAL,
            setpoint REAL,
            ideal_temp REAL,
            hvac_mode TEXT,
            ac_persona TEXT,
            occupant_state TEXT,
            pir_persona TEXT,
            room_state TEXT,
            pir_motion INTEGER,
            guest_id TEXT
        )
    ''')

    # Create weather_antalya table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_antalya (
            date TEXT PRIMARY KEY,
            max_temp REAL,
            min_temp REAL
        )
    ''')

    # Create lightning_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lightning_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            room_number INTEGER,
            floor INTEGER,
            Addresss TEXT,
            lamp_location TEXT,
            Value REAL,
            room_state TEXT,
            reservation_active TEXT,
            pir_motion INTEGER,
            pir_persona TEXT,
            lightning_persona TEXT,
            n_occupants INTEGER,
            active_actors INTEGER,
            hurry_morning TEXT,
            lazy_day TEXT,
            forgetful TEXT
        )
    ''')

    conn.commit()
    return conn

def table_has_data(conn, table_name):
    """Check if a table already has data"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    return count > 0

def import_csv_to_table(conn, csv_path: Path, table_name: str):
    """Import CSV data into database table"""
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = reader.fieldnames

        # Clean column names for SQLite (replace spaces with underscores)
        cleaned_columns = [col.replace(' ', '_').replace('-', '_') for col in columns]

        # Prepare INSERT statement
        placeholders = ', '.join(['?' for _ in cleaned_columns])
        column_names = ', '.join([f'"{col}"' for col in cleaned_columns])

        insert_sql = f'INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})'

        # Insert data
        cursor = conn.cursor()
        row_count = 0
        for row in reader:
            values = [row[col] for col in columns]  # Use original column names from CSV
            cursor.execute(insert_sql, values)
            row_count += 1

        conn.commit()
        print(f"Imported {row_count} rows into {table_name}")

def setup_database(force_reimport=False):
    """Main function to setup database and import data"""
    base_dir = Path(__file__).resolve().parent  # setup_database.py is in the project root
    data_dir = base_dir / 'Data'
    db_path = base_dir / 'data.db'

    print(f"Base dir: {base_dir}")
    print(f"Data dir: {data_dir}")
    print(f"DB path: {db_path}")

    print("Creating database...")
    conn = create_database(db_path)

    # Import hotel reservation data
    hotel_csv = data_dir / 'hotelReservationData.csv'
    print(f"Hotel CSV path: {hotel_csv}")
    print(f"Hotel CSV exists: {hotel_csv.exists()}")
    if hotel_csv.exists():
        if table_has_data(conn, 'hotel_reservations') and not force_reimport:
            print("Hotel reservations table already has data. Skipping import. Use --force to reimport.")
        else:
            if force_reimport and table_has_data(conn, 'hotel_reservations'):
                print("Clearing existing hotel reservation data...")
                conn.execute("DELETE FROM hotel_reservations")
                conn.commit()
            print("Importing hotel reservation data...")
            import_csv_to_table(conn, hotel_csv, 'hotel_reservations')
    else:
        print("Warning: hotelReservationData.csv not found")

    # Import PIR sensor data
    pir_csv = data_dir / 'PIRSensorData.csv'
    print(f"PIR CSV path: {pir_csv}")
    print(f"PIR CSV exists: {pir_csv.exists()}")
    if pir_csv.exists():
        if not force_reimport and table_has_data(conn, 'pir_sensor_data'):
            print("PIR sensor data table already has data. Skipping import. Use --force to reimport.")
        else:
            if force_reimport and table_has_data(conn, 'pir_sensor_data'):
                print("Dropping and recreating PIR sensor data table...")
                conn.execute("DROP TABLE IF EXISTS pir_sensor_data")
                conn.commit()
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE pir_sensor_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        room_number INTEGER,
                        pir_motion INTEGER,
                        room_state TEXT,
                        persona TEXT,
                        adults INTEGER,
                        children INTEGER,
                        guest_id INTEGER
                    )
                ''')
                conn.commit()
            print("Importing PIR sensor data...")
            import_csv_to_table(conn, pir_csv, 'pir_sensor_data')
    else:
        print("Warning: PIRSensorData.csv not found")

    # Import lightning data
    lightning_csv = data_dir / 'lightningData.csv'
    print(f"Lightning CSV path: {lightning_csv}")
    print(f"Lightning CSV exists: {lightning_csv.exists()}")
    if lightning_csv.exists():
        if not force_reimport and table_has_data(conn, 'lightning_data'):
            print("Lightning data table already has data. Skipping import. Use --force to reimport.")
        else:
            if force_reimport and table_has_data(conn, 'lightning_data'):
                print("Dropping and recreating lightning data table...")
                conn.execute("DROP TABLE IF EXISTS lightning_data")
                conn.commit()
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE lightning_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        room_number INTEGER,
                        floor INTEGER,
                        Addresss TEXT,
                        lamp_location TEXT,
                        Value REAL,
                        room_state TEXT,
                        reservation_active TEXT,
                        pir_motion INTEGER,
                        pir_persona TEXT,
                        lightning_persona TEXT,
                        n_occupants INTEGER,
                        active_actors INTEGER,
                        hurry_morning TEXT,
                        lazy_day TEXT,
                        forgetful TEXT
                    )
                ''')
                conn.commit()
            print("Importing lightning data...")
            import_csv_to_table(conn, lightning_csv, 'lightning_data')
    else:
        print("Warning: lightningData.csv not found")

    # Import temperature data
    temp_csv = data_dir / 'temperatureData.csv'
    print(f"Temperature CSV path: {temp_csv}")
    print(f"Temperature CSV exists: {temp_csv.exists()}")
    if temp_csv.exists():
        if not force_reimport and table_has_data(conn, 'temperature_data'):
            print("Temperature data table already has data. Skipping import. Use --force to reimport.")
        else:
            if force_reimport and table_has_data(conn, 'temperature_data'):
                print("Dropping and recreating temperature data table...")
                conn.execute("DROP TABLE IF EXISTS temperature_data")
                conn.commit()
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE temperature_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        room_number INTEGER,
                        floor INTEGER,
                        facade TEXT,
                        room_type TEXT,
                        size_m2 REAL,
                        outside_temp REAL,
                        room_temp REAL,
                        setpoint REAL,
                        ideal_temp REAL,
                        hvac_mode TEXT,
                        ac_persona TEXT,
                        occupant_state TEXT,
                        pir_persona TEXT,
                        room_state TEXT,
                        pir_motion INTEGER,
                        guest_id TEXT
                    )
                ''')
                conn.commit()
            print("Importing temperature data (this may take a moment)...")
            import_csv_to_table(conn, temp_csv, 'temperature_data')
    else:
        print("Warning: temperatureData.csv not found")

    # Import weather Antalya data
    weather_csv = data_dir / 'WheatherDataAntalya.csv'
    print(f"Weather CSV path: {weather_csv}")
    print(f"Weather CSV exists: {weather_csv.exists()}")
    if weather_csv.exists():
        if not force_reimport and table_has_data(conn, 'weather_antalya'):
            print("Weather Antalya table already has data. Skipping import. Use --force to reimport.")
        else:
            if force_reimport and table_has_data(conn, 'weather_antalya'):
                print("Clearing existing weather Antalya data...")
                conn.execute("DELETE FROM weather_antalya")
                conn.commit()
            print("Importing weather Antalya data...")
            import_csv_to_table(conn, weather_csv, 'weather_antalya')
    else:
        print("Warning: WheatherDataAntalya.csv not found")

    conn.close()
    print(f"Database setup complete! Database saved at: {db_path}")

if __name__ == '__main__':
    force_reimport = '--force' in sys.argv
    setup_database(force_reimport=force_reimport)