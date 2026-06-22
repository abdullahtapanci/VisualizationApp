# CSV Visualization App

A simple fullstack app for browsing and searching CSV data stored in a SQLite database.

## Database Setup

The app uses SQLite to store CSV data efficiently. To set up the database:

```bash
python setup_database.py
```

This will:
- Create `data.db` SQLite database
- Import `hotelReservationData.csv` into `hotel_reservations` table
- Import `PIRSensorData.csv` into `pir_sensor_data` table
- Clean column names for database compatibility

**Note**: If the database already exists, the script will skip importing data. To reimport data:

```bash
python setup_database.py --force
```

This will clear existing data and reimport from the CSV files.

## Backend (Python)

- Uses Flask to expose database queries
- SQLite database for efficient data storage and retrieval
- Endpoints:
  - `GET /api/files` - Lists available datasets
  - `GET /api/data?file=...&limit=...&search_column=...&search_value=...` - Loads data with optional search and limit

## Frontend (React)

- Built with Vite and React
- Initially loads only first 20 rows for performance
- Search functionality with column selection and text input
- Real-time search results from backend

## Features

- **Database Storage**: CSV data stored in SQLite for better performance
- **Efficient Loading**: Only loads preview data initially
- **Server-Side Search**: Search happens in database, returns up to 100 matching results
- **Column-Specific Search**: Search within specific columns
- **Responsive UI**: Clean interface for browsing and searching data

## Run locally

1. Set up the database:
   ```bash
   python setup_database.py
   ```

2. Install backend dependencies:
   ```bash
   python3 -m pip install -r backend/requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend && npm install
   ```

4. Start the backend:
   ```bash
   export FLASK_APP=backend/main.py
   /Users/abdullahtapanci/.pyenv/versions/3.8.10/bin/python -m flask run --port 8000
   ```

5. Start the frontend:
   ```bash
   cd frontend && npm run dev
   ```

Open `http://localhost:5173` in your browser.
