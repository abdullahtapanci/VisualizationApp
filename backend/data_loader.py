import sqlite3
from pathlib import Path


def get_db_connection():
    """Get database connection"""
    base_dir = Path(__file__).resolve().parent.parent
    db_path = base_dir / 'data.db'
    return sqlite3.connect(db_path)


def get_table_names():
    """Get list of available tables in database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_table_columns(table_name: str):
    """Get column names for a table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

    # Map cleaned column names back to original names for display
    column_mapping = {
        'reservation_id': 'Reservation ID',
        'guest_id': 'Guest ID',
        'first_name': 'First Name',
        'last_name': 'Last Name',
        'check_in_date': 'Check-in Date',
        'check_out_date': 'Check-out Date',
        'room_number': 'Room Number',
        'floor_number': 'Floor Number',
        'room_type': 'Room Type',
        'total_nights': 'Total Nights',
        'total_amount': 'Total Amount',
        'payment_status': 'Payment Status',
        'booking_date': 'Booking Date',
        'check_in_time': 'Check-in Time',
        'check_out_time': 'Check-out Time',
        'stay_duration': 'Stay Duration',
        'postal_code': 'Postal Code',
        'pir_motion': 'PIR Motion',
        'room_state': 'Room State',
        'addresss': 'Address',
        'lamp_location': 'Lamp Location',
        'value': 'Value',
        'reservation_active': 'Reservation Active',
        'floor': 'Floor',
        'pir_persona': 'PIR Persona',
        'lightning_persona': 'Lightning Persona',
        'n_occupants': 'N Occupants',
        'active_actors': 'Active Actors',
        'hurry_morning': 'Hurry Morning',
        'lazy_day': 'Lazy Day',
        'forgetful': 'Forgetful',
        'date': 'Date',
        'max_temp': 'Max Temp (°C)',
        'min_temp': 'Min Temp (°C)',
        'outside_temp': 'Outside Temp (°C)',
        'room_temp': 'Room Temp (°C)',
        'setpoint': 'Setpoint (°C)',
        'ideal_temp': 'Ideal Temp (°C)',
        'hvac_mode': 'HVAC Mode',
        'ac_persona': 'AC Persona',
        'facade': 'Facade',
        'room_type': 'Room Type',
        'size_m2': 'Size (m²)',
        'occupant_state': 'Occupant State',
    }

    return [column_mapping.get(col.lower(), col) for col in columns]


def get_table_row_count(table_name: str):
    """Get total row count for a table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def load_table_data_paginated(table_name: str, page: int = 1, page_size: int = 20):
    """Load paginated data from database table"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get database column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    db_columns = [row[1] for row in cursor.fetchall()]

    # Get display column names
    display_columns = get_table_columns(table_name)
    total_count = get_table_row_count(table_name)

    # Calculate pagination
    offset = (page - 1) * page_size
    total_pages = (total_count + page_size - 1) // page_size  # Ceiling division

    # Build query
    query = f"SELECT * FROM {table_name} LIMIT {page_size} OFFSET {offset}"

    cursor.execute(query)
    db_rows = cursor.fetchall()

    # Convert rows to use display column names
    rows = []
    for db_row in db_rows:
        row_dict = {}
        for i, db_col in enumerate(db_columns):
            display_col = display_columns[i]
            row_dict[display_col] = db_row[i]
        rows.append(row_dict)

    conn.close()
    return {
        'columns': display_columns,
        'rows': rows,
        'count': len(rows),
        'total_count': total_count,
        'page': page,
        'page_size': page_size,
        'total_pages': total_pages,
        'has_next': page < total_pages,
        'has_prev': page > 1
    }


def load_table_data(table_name: str):
    """Load all data from a database table for non-paginated use cases."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    db_columns = [row[1] for row in cursor.fetchall()]
    display_columns = get_table_columns(table_name)

    cursor.execute(f"SELECT * FROM {table_name}")
    db_rows = cursor.fetchall()

    rows = []
    for db_row in db_rows:
        row_dict = {}
        for i, db_col in enumerate(db_columns):
            display_col = display_columns[i]
            row_dict[display_col] = db_row[i]
        rows.append(row_dict)

    conn.close()
    return {
        'columns': display_columns,
        'rows': rows,
        'count': len(rows)
    }


def search_table_data(table_name: str, search_column=None, search_value='', page: int = 1, page_size: int = 20):
    """Search data in database table with pagination"""
    if not search_column or not search_value:
        return load_table_data_paginated(table_name, page, page_size)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get database column names and find the matching DB column
    cursor.execute(f"PRAGMA table_info({table_name})")
    db_columns = [row[1] for row in cursor.fetchall()]
    display_columns = get_table_columns(table_name)

    # Find the database column name that corresponds to the search column
    db_search_column = None
    for i, display_col in enumerate(display_columns):
        if display_col.lower() == search_column.lower():  # Case-insensitive matching
            db_search_column = db_columns[i]
            break

    if not db_search_column:
        conn.close()
        total_count = get_table_row_count(table_name)
        total_pages = (total_count + page_size - 1) // page_size
        return {
            'columns': display_columns,
            'rows': [],
            'count': 0,
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }

    # Get total count for search results
    if db_search_column.lower() == 'timestamp' and search_value and not ':' in search_value:
        # For timestamp column with date-only input, match entire day
        search_term = f"{search_value}%"
        count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {db_search_column} LIKE ?"
        cursor.execute(count_query, (search_term,))
    else:
        search_term = f"%{search_value}%"
        count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {db_search_column} LIKE ?"
        cursor.execute(count_query, (search_term,))
    total_count = cursor.fetchone()[0]

    # Calculate pagination
    offset = (page - 1) * page_size
    total_pages = (total_count + page_size - 1) // page_size  # Ceiling division

    # Build search query with pagination
    if db_search_column.lower() == 'timestamp' and search_value and not ':' in search_value:
        # For timestamp column with date-only input, match entire day
        search_term = f"{search_value}%"
        query = f"SELECT * FROM {table_name} WHERE {db_search_column} LIKE ? LIMIT {page_size} OFFSET {offset}"
        cursor.execute(query, (search_term,))
    else:
        search_term = f"%{search_value}%"
        query = f"SELECT * FROM {table_name} WHERE {db_search_column} LIKE ? LIMIT {page_size} OFFSET {offset}"
        cursor.execute(query, (search_term,))
    db_rows = cursor.fetchall()

    # Convert rows to use display column names
    rows = []
    for db_row in db_rows:
        row_dict = {}
        for i, db_col in enumerate(db_columns):
            display_col = display_columns[i]
            row_dict[display_col] = db_row[i]
        rows.append(row_dict)

    conn.close()
    return {
        'columns': display_columns,
        'rows': rows,
        'count': len(rows),
        'total_count': total_count,
        'page': page,
        'page_size': page_size,
        'total_pages': total_pages,
        'has_next': page < total_pages,
        'has_prev': page > 1
    }
def list_csv_files(data_dir: Path):
    """Legacy function - now returns table names as filenames"""
    tables = get_table_names()
    # Convert table names back to "filenames" for frontend compatibility
    filename_mapping = {
        'hotel_reservations': 'hotelReservationData.csv',
        'pir_sensor_data': 'PIRSensorData.csv',
        'lightning_data': 'lightningData.csv',
        'weather_antalya': 'WheatherDataAntalya.csv',
        'temperature_data': 'temperatureData.csv'
    }
    return [filename_mapping.get(table, table) for table in tables]


def load_csv_file(data_dir: Path, file_name: str, limit=None):
    """Legacy function - now loads from database"""
    # Convert filename to table name (remove .csv extension)
    table_name = file_name.replace('.csv', '').lower()
    # Handle specific mappings
    if 'hotelreservation' in table_name:
        table_name = 'hotel_reservations'
    elif 'pirsensor' in table_name:
        table_name = 'pir_sensor_data'
    elif 'lightning' in table_name:
        table_name = 'lightning_data'
    elif 'wheather' in table_name or 'weather' in table_name:
        table_name = 'weather_antalya'
    elif 'tempreture' in table_name or 'temperature' in table_name:
        table_name = 'temperature_data'
    # For backward compatibility, if limit is provided, use pagination with page 1
    if limit:
        return load_table_data_paginated(table_name, page=1, page_size=limit)
    else:
        return load_table_data_paginated(table_name, page=1, page_size=20)


def search_csv_file(data_dir: Path, file_name: str, search_column=None, search_value='', page: int = 1, page_size: int = 20):
    """Legacy function - now searches database with pagination"""
    table_name = file_name.replace('.csv', '').lower()
    # Handle specific mappings
    if 'hotelreservation' in table_name:
        table_name = 'hotel_reservations'
    elif 'pirsensor' in table_name:
        table_name = 'pir_sensor_data'
    elif 'lightning' in table_name:
        table_name = 'lightning_data'
    elif 'wheather' in table_name or 'weather' in table_name:
        table_name = 'weather_antalya'
    elif 'tempreture' in table_name or 'temperature' in table_name:
        table_name = 'temperature_data'
    return search_table_data(table_name, search_column, search_value, page, page_size)


def get_available_rooms():
    """Get list of available room numbers from PIR sensor and lightning data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT room_number FROM pir_sensor_data UNION SELECT DISTINCT room_number FROM lightning_data ORDER BY room_number"
    )
    rooms = [row[0] for row in cursor.fetchall()]
    conn.close()
    return rooms


def load_pir_data_filtered(room_number=None, start_timestamp=None, end_timestamp=None):
    """Load PIR sensor data filtered by room number and time range"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get database column names
    cursor.execute("PRAGMA table_info(pir_sensor_data)")
    db_columns = [row[1] for row in cursor.fetchall()]
    display_columns = get_table_columns('pir_sensor_data')

    # Build query with filters
    query = "SELECT * FROM pir_sensor_data WHERE 1=1"
    params = []

    if room_number:
        query += " AND room_number = ?"
        params.append(room_number)

    if start_timestamp:
        query += " AND timestamp >= ?"
        params.append(start_timestamp)

    if end_timestamp:
        query += " AND timestamp <= ?"
        params.append(end_timestamp)

    query += " ORDER BY timestamp"

    cursor.execute(query, params)
    db_rows = cursor.fetchall()

    # Convert rows to use display column names
    rows = []
    for db_row in db_rows:
        row_dict = {}
        for i, db_col in enumerate(db_columns):
            display_col = display_columns[i]
            row_dict[display_col] = db_row[i]
        rows.append(row_dict)

    conn.close()
    return rows


def load_lightning_data_filtered(room_number=None, start_timestamp=None, end_timestamp=None, lamp_location=None):
    """Load lightning sensor data filtered by room number, time range, and lamp location"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(lightning_data)")
    db_columns = [row[1] for row in cursor.fetchall()]
    display_columns = get_table_columns('lightning_data')

    query = "SELECT * FROM lightning_data WHERE 1=1"
    params = []

    if room_number:
        query += " AND room_number = ?"
        params.append(room_number)

    if start_timestamp:
        query += " AND timestamp >= ?"
        params.append(start_timestamp)

    if end_timestamp:
        query += " AND timestamp <= ?"
        params.append(end_timestamp)

    if lamp_location:
        query += " AND lamp_location = ?"
        params.append(lamp_location)

    query += " ORDER BY timestamp"

    cursor.execute(query, params)
    db_rows = cursor.fetchall()

    rows = []
    for db_row in db_rows:
        row_dict = {}
        for i, db_col in enumerate(db_columns):
            display_col = display_columns[i]
            row_dict[display_col] = db_row[i]
        rows.append(row_dict)

    conn.close()
    return rows
