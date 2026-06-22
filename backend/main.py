import os
import csv
import json
from pathlib import Path
import base64
import io

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from backend.data_loader import list_csv_files, load_csv_file, search_csv_file, load_table_data_paginated, load_table_data, search_table_data, get_available_rooms, load_pir_data_filtered, load_lightning_data_filtered, get_db_connection, get_table_columns
from backend.energy_calc import (
    compute_energy_for_room,
    DIMMABLE_LAMPS,
    NON_DIMMABLE_LAMPS,
)
from backend.hvac_energy import (
    compute_hvac_energy_for_room,
    RATED_POWER_W,
    SAMPLE_MINUTES as HVAC_SAMPLE_MINUTES,
)
from backend.lightning_recommendation import (
    compute_lightning_recommendation,
    compute_lightning_recommendation_energy_for_room,
)
from backend.prediction_services import (
    predict_occupancy,
    predict_lighting_persona,
    predict_tempreture_persona,
    predict_tempreture_recomendation,
    compute_tempreture_recomendation_energy_for_room,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BASE_DIR / 'frontend' / 'dist'

app = Flask(__name__)
CORS(app)


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': str(e)}), 500


@app.route('/api/files', methods=['GET'])
def get_files():
    return jsonify({'files': list_csv_files(None)})


@app.route('/api/rooms', methods=['GET'])
def get_rooms():
    try:
        rooms = get_available_rooms()
        return jsonify({'rooms': rooms})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_pir_activity_heatmap(df, room_number, start_timestamp, end_timestamp):
    """Generate a PIR activity heatmap with time of day vs date visualization"""
    try:
        # Clean column names
        df.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df.columns]
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter only motion detected events (pir_motion = 1)
        motion_df = df[df['pir_motion'] == 1].copy()
        
        if motion_df.empty:
            return jsonify({'error': 'No motion detected in the selected time range'}), 400
        
        # Extract date and time
        motion_df['date'] = motion_df['timestamp'].dt.strftime('%d.%m.%Y')
        motion_df['time_hours'] = motion_df['timestamp'].dt.hour + motion_df['timestamp'].dt.minute / 60
        motion_df['time_str'] = motion_df['timestamp'].dt.strftime('%H:%M')
        
        # Get unique dates for y-axis
        unique_dates = sorted(motion_df['date'].unique())
        date_to_y = {date: i for i, date in enumerate(unique_dates)}
        
        # Create figure with larger size for better readability
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create scatter plot
        y_coords = [date_to_y[date] for date in motion_df['date']]
        x_coords = motion_df['time_hours'].values
        
        scatter = ax.scatter(x_coords, y_coords, alpha=0.7, s=100, c='#e74c3c', edgecolors='#c0392b', linewidth=1.5)
        
        # Configure x-axis (time of day)
        ax.set_xlabel('Time of Day (HH:MM)', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.5, 24)
        hour_ticks = list(range(0, 25))
        hour_labels = [f'{h:02d}:00' for h in range(0, 25)]
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels(hour_labels, rotation=45, ha='right')
        
        # Configure y-axis (dates)
        ax.set_ylabel('Date (DD.MM.YYYY)', fontsize=12, fontweight='bold')
        ax.set_yticks(list(range(len(unique_dates))))
        ax.set_yticklabels(unique_dates)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Title
        title = f'Room {room_number} - PIR Motion Activity Map\n({len(motion_df)} motion events detected)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        ax.text(0.02, 0.98, f'Red dots = Motion detected by PIR sensor', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        chart_url = f'data:image/png;base64,{image_base64}'
        return jsonify({'chart_url': chart_url})
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate activity heatmap: {str(e)}'}), 500


def generate_lightning_value_heatmap(df, room_number, start_timestamp, end_timestamp, lamp_location=None):
    """Generate a heatmap for lightning value by time of day vs date"""
    try:
        df.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df.columns]
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        df = df.dropna(subset=['timestamp', 'value'])
        if df.empty:
            return jsonify({'error': 'No lightning measurements found in the selected time range'}), 400

        df['date'] = df['timestamp'].dt.strftime('%d.%m.%Y')
        df['time_hours'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60

        unique_dates = sorted(df['date'].unique())
        date_to_y = {date: i for i, date in enumerate(unique_dates)}
        y_coords = [date_to_y[date] for date in df['date']]
        x_coords = df['time_hours'].values
        colors = df['value'].values

        fig, ax = plt.subplots(figsize=(14, 8))
        scatter = ax.scatter(
            x_coords,
            y_coords,
            c=colors,
            cmap='Reds',
            vmin=0,
            vmax=100,
            s=80,
            edgecolor='none',
            alpha=0.9
        )

        ax.set_xlabel('Time of Day (HH:MM)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 24)
        hour_ticks = list(range(0, 25, 2))
        hour_labels = [f'{h:02d}:00' for h in hour_ticks]
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels(hour_labels, rotation=45, ha='right')

        ax.set_ylabel('Date (DD.MM.YYYY)', fontsize=12, fontweight='bold')
        ax.set_yticks(list(range(len(unique_dates))))
        ax.set_yticklabels(unique_dates)

        location_label = f' | {lamp_location.replace("_", " ").title()}' if lamp_location else ' | All Locations'
        ax.set_title(
            f'Room {room_number} - Lightning Value Heatmap{location_label}\n(Time of day vs Date)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        ax.grid(True, alpha=0.25, linestyle='--')

        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Lightning Value (0-100)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({'chart_url': f'data:image/png;base64,{image_base64}'})
    except Exception as e:
        return jsonify({'error': f'Failed to generate lightning heatmap: {str(e)}'}), 500


def generate_lightning_daily_trend(df, room_number, date, lamp_location=None):
    """Generate a column chart with one bar per data point over a single day"""
    try:
        df.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df.columns]
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'value'])

        if df.empty:
            return jsonify({'error': 'No lightning measurements found for the selected day'}), 400

        # Build a complete index of every 5-minute slot in the day (288 slots)
        full_index = pd.date_range(
            start=df['timestamp'].dt.normalize().iloc[0],
            periods=288, freq='5min'
        )
        full_labels = [t.strftime('%H:%M') for t in full_index]
        # x positions: one integer per slot
        x_pos = list(range(288))

        hour_tick_positions = list(range(0, 288, 12))
        hour_tick_labels   = [full_labels[i] for i in hour_tick_positions]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        if lamp_location:
            # Single location — one chart
            fig, ax = plt.subplots(figsize=(20, 5))
            slot_map = df.set_index('timestamp')['value'].reindex(full_index, fill_value=0)
            ax.bar(x_pos, slot_map.values, color='#c0392b', edgecolor='none', alpha=0.85, width=1.0)
            ax.set_xticks(hour_tick_positions)
            ax.set_xticklabels(hour_tick_labels, rotation=45, ha='right')
            ax.set_xlim(-1, 288)
            ax.set_xlabel('Time of Day (HH:MM)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Lightning Value (0–100)', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 105)
            ax.set_title(
                f'Room {room_number}  |  {lamp_location.replace("_", " ").title()}  —  {date}',
                fontsize=13, fontweight='bold', pad=14
            )
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        else:
            # All locations — one subplot per location
            locs = sorted(df['lamp_location'].unique())
            n    = len(locs)
            fig, axes = plt.subplots(
                n, 1,
                figsize=(20, 4.6 * n),
                sharex=True,
                gridspec_kw={'hspace': 0.85},
            )
            if n == 1:
                axes = [axes]   # keep iterable when only one location exists

            for i, (loc, ax) in enumerate(zip(locs, axes)):
                bar_color = colors[i % len(colors)]
                loc_df    = df[df['lamp_location'] == loc].set_index('timestamp')['value']
                slot_vals = loc_df.reindex(full_index, fill_value=0)
                ax.bar(x_pos, slot_vals.values,
                       color=bar_color, edgecolor='none', alpha=0.88, width=1.0)
                ax.set_xlim(-1, 288)
                ax.set_ylim(0, 105)
                ax.set_ylabel('Value (0–100)', fontsize=10, fontweight='bold')

                # Frame each subplot so it reads as one self-contained card
                for spine in ax.spines.values():
                    spine.set_edgecolor('#bdc3c7')
                    spine.set_linewidth(1.2)
                ax.set_facecolor('#fbfcfd')
                ax.grid(True, axis='y', alpha=0.3, linestyle='--')

                # Anchored, colored title that visually "belongs" to the graph
                # below it — left-aligned, with a filled background bar in the
                # same color as the bars, plus a leading marker (▎) for
                # extra visual association.
                ax.set_title(
                    f'  ▎ {i + 1}. {loc.replace("_", " ").title()}',
                    fontsize=13,
                    fontweight='bold',
                    color='white',
                    loc='left',
                    pad=8,
                    backgroundcolor=bar_color,
                )

            # X-axis labels only on the bottom subplot
            axes[-1].set_xticks(hour_tick_positions)
            axes[-1].set_xticklabels(hour_tick_labels, rotation=45, ha='right')
            axes[-1].set_xlabel('Time of Day (HH:MM)', fontsize=11, fontweight='bold')

            fig.suptitle(
                f'Room {room_number}  —  Daily Lightning Trend  |  All Locations  ({date})',
                fontsize=15, fontweight='bold', y=0.995
            )

        if lamp_location:
            plt.tight_layout()
        else:
            # tight_layout fights the manual hspace we set for the
            # multi-subplot case, so leave the spacing alone there.
            fig.subplots_adjust(top=0.94, bottom=0.07, left=0.06, right=0.98)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({'chart_url': f'data:image/png;base64,{image_base64}'})
    except Exception as e:
        return jsonify({'error': f'Failed to generate daily trend: {str(e)}'}), 500


def generate_weather_chart(df, chart_type):
    """Generate weather visualizations for WheatherDataAntalya.csv.
    Supports: yearly_trend, monthly_avg, temp_distribution, extremes."""
    try:
        df.columns = [c.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('°', '').lower() for c in df.columns]
        # After cleaning, the display columns "Max Temp (°C)" → "max_temp_c"
        # and "Min Temp (°C)" → "min_temp_c". Rename to canonical names.
        rename_map = {}
        for col in df.columns:
            if col.startswith('max_temp'):
                rename_map[col] = 'max_temp'
            elif col.startswith('min_temp'):
                rename_map[col] = 'min_temp'
        df = df.rename(columns=rename_map)

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['max_temp'] = pd.to_numeric(df['max_temp'], errors='coerce')
        df['min_temp'] = pd.to_numeric(df['min_temp'], errors='coerce')
        df = df.dropna(subset=['date', 'max_temp', 'min_temp']).sort_values('date').reset_index(drop=True)

        if df.empty:
            return jsonify({'error': 'No weather rows available'}), 400

        hot_color = '#e74c3c'
        cold_color = '#3498db'

        if chart_type == 'yearly_trend':
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.fill_between(df['date'], df['min_temp'], df['max_temp'],
                            color='#f1c40f', alpha=0.25, label='Daily range')
            ax.plot(df['date'], df['max_temp'], color=hot_color, linewidth=1.4, label='Max temperature')
            ax.plot(df['date'], df['min_temp'], color=cold_color, linewidth=1.4, label='Min temperature')
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax.set_title('Antalya — Yearly Temperature Trend', fontsize=14, fontweight='bold', pad=14)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right')
            fig.autofmt_xdate()

        elif chart_type == 'monthly_avg':
            df['month'] = df['date'].dt.month
            monthly = df.groupby('month')[['max_temp', 'min_temp']].mean().reindex(range(1, 13))
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x = list(range(12))
            width = 0.4
            fig, ax = plt.subplots(figsize=(14, 6))
            bars_max = ax.bar([i - width / 2 for i in x], monthly['max_temp'].values,
                              width=width, color=hot_color, edgecolor='#c0392b', label='Avg Max')
            bars_min = ax.bar([i + width / 2 for i in x], monthly['min_temp'].values,
                              width=width, color=cold_color, edgecolor='#2980b9', label='Avg Min')
            for bars in (bars_max, bars_min):
                for bar in bars:
                    h = bar.get_height()
                    if pd.notna(h):
                        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                                f'{h:.1f}', ha='center', va='bottom', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(month_labels)
            ax.set_xlabel('Month', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax.set_title('Antalya — Monthly Average Temperatures', fontsize=14, fontweight='bold', pad=14)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper left')

        elif chart_type == 'temp_distribution':
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(df['max_temp'], bins=20, color=hot_color, edgecolor='#922b21', alpha=0.85)
            axes[0].set_title('Max Temperature Distribution', fontsize=13, fontweight='bold')
            axes[0].set_xlabel('Max Temp (°C)', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Days', fontsize=11, fontweight='bold')
            axes[0].grid(True, axis='y', alpha=0.3, linestyle='--')
            axes[0].axvline(df['max_temp'].mean(), color='black', linestyle='--', linewidth=1.2,
                            label=f"Mean: {df['max_temp'].mean():.1f}°C")
            axes[0].legend()
            axes[1].hist(df['min_temp'], bins=20, color=cold_color, edgecolor='#1f4e79', alpha=0.85)
            axes[1].set_title('Min Temperature Distribution', fontsize=13, fontweight='bold')
            axes[1].set_xlabel('Min Temp (°C)', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Days', fontsize=11, fontweight='bold')
            axes[1].grid(True, axis='y', alpha=0.3, linestyle='--')
            axes[1].axvline(df['min_temp'].mean(), color='black', linestyle='--', linewidth=1.2,
                            label=f"Mean: {df['min_temp'].mean():.1f}°C")
            axes[1].legend()
            fig.suptitle('Antalya — Temperature Distribution (2025)', fontsize=15, fontweight='bold', y=1.02)

        elif chart_type == 'extremes':
            top_n = 10
            hottest = df.nlargest(top_n, 'max_temp')[['date', 'max_temp']].iloc[::-1]
            coldest = df.nsmallest(top_n, 'min_temp')[['date', 'min_temp']].iloc[::-1]
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            hot_labels = hottest['date'].dt.strftime('%d %b').tolist()
            axes[0].barh(hot_labels, hottest['max_temp'].values,
                         color=hot_color, edgecolor='#922b21')
            for i, v in enumerate(hottest['max_temp'].values):
                axes[0].text(v + 0.2, i, f'{v:.0f}°C', va='center', fontsize=10, fontweight='bold')
            axes[0].set_title(f'Top {top_n} Hottest Days', fontsize=13, fontweight='bold')
            axes[0].set_xlabel('Max Temp (°C)', fontsize=11, fontweight='bold')
            axes[0].grid(True, axis='x', alpha=0.3, linestyle='--')
            cold_labels = coldest['date'].dt.strftime('%d %b').tolist()
            axes[1].barh(cold_labels, coldest['min_temp'].values,
                         color=cold_color, edgecolor='#1f4e79')
            for i, v in enumerate(coldest['min_temp'].values):
                axes[1].text(v + 0.2, i, f'{v:.0f}°C', va='center', fontsize=10, fontweight='bold')
            axes[1].set_title(f'Top {top_n} Coldest Days', fontsize=13, fontweight='bold')
            axes[1].set_xlabel('Min Temp (°C)', fontsize=11, fontweight='bold')
            axes[1].grid(True, axis='x', alpha=0.3, linestyle='--')
            fig.suptitle('Antalya — Temperature Extremes (2025)', fontsize=15, fontweight='bold', y=1.02)

        else:
            return jsonify({'error': f'Unsupported weather chart type: {chart_type}'}), 400

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return jsonify({'chart_url': f'data:image/png;base64,{image_base64}'})
    except Exception as e:
        return jsonify({'error': f'Failed to generate weather chart: {str(e)}'}), 500


def generate_daily_temperature_trend(room_number, date):
    """Plot a single room's room_temp over the 24h of a single day,
    with setpoint and outside_temp shown as reference lines."""
    if not room_number or not date:
        return jsonify({'error': 'Room number and date are required.'}), 400

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, room_temp, setpoint, outside_temp, hvac_mode
            FROM temperature_data
            WHERE room_number = ? AND timestamp LIKE ?
            ORDER BY timestamp
            """,
            conn,
            params=(room_number, f"{date}%"),
        )
        if df.empty:
            return jsonify({'error': f'No temperature data for room {room_number} on {date}.'}), 400

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
        df['time_hours'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0

        fig, ax = plt.subplots(figsize=(16, 6))

        # Shade heating/cooling regions so the user can see when HVAC ran.
        if 'hvac_mode' in df.columns:
            mode_band_colors = {'heating': '#fadbd8', 'cooling': '#d6eaf8'}
            mode_arr = df['hvac_mode'].fillna('').to_numpy()
            time_arr = df['time_hours'].to_numpy()
            i = 0
            while i < len(mode_arr):
                m = mode_arr[i]
                if m in mode_band_colors:
                    j = i
                    while j + 1 < len(mode_arr) and mode_arr[j + 1] == m:
                        j += 1
                    ax.axvspan(time_arr[i], time_arr[min(j + 1, len(time_arr) - 1)],
                               color=mode_band_colors[m], alpha=0.55, linewidth=0)
                    i = j + 1
                else:
                    i += 1

        ax.plot(df['time_hours'], df['room_temp'],
                color='#1f4e79', linewidth=2.2, label='Room Temp')
        ax.plot(df['time_hours'], df['setpoint'],
                color='#27ae60', linewidth=1.4, linestyle='--', label='Setpoint')
        ax.plot(df['time_hours'], df['outside_temp'],
                color='#e67e22', linewidth=1.2, linestyle=':', label='Outside Temp')

        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)], rotation=45, ha='right')
        ax.set_xlabel('Time of Day (HH:MM)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title(f'Room {room_number}  —  Temperature Trend on {date}',
                     fontsize=14, fontweight='bold', pad=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Append the HVAC band legend entries so the shaded backgrounds are
        # explained alongside the lines.
        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor='#fadbd8', edgecolor='none', label='Heating active'))
        handles.append(Patch(facecolor='#d6eaf8', edgecolor='none', label='Cooling active'))
        ax.legend(handles=handles, loc='upper right')

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return jsonify({'chart_url': f'data:image/png;base64,{image_base64}'})
    except Exception as e:
        return jsonify({'error': f'Failed to generate daily temperature trend: {str(e)}'}), 500
    finally:
        conn.close()


def generate_temperature_chart(chart_type):
    """Generate temperatureData.csv visualizations using SQL aggregation
    so we never load the full ~900k-row table into Python.
    Supports: temp_outside_indoor, hvac_mode_distribution, avg_temp_by_floor,
              setpoint_deviation, hvac_by_room_state."""
    conn = get_db_connection()
    try:
        if chart_type == 'temp_outside_indoor':
            # Daily mean of outside_temp and room_temp.
            df = pd.read_sql_query(
                """
                SELECT substr(timestamp, 1, 10) AS day,
                       AVG(outside_temp)        AS outside,
                       AVG(room_temp)           AS indoor
                FROM temperature_data
                GROUP BY day
                ORDER BY day
                """,
                conn,
            )
            if df.empty:
                return jsonify({'error': 'No temperature data available'}), 400
            df['day'] = pd.to_datetime(df['day'])
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(df['day'], df['outside'], color='#e67e22', linewidth=1.6,
                    marker='o', markersize=3, label='Outside (avg)')
            ax.plot(df['day'], df['indoor'], color='#2980b9', linewidth=1.6,
                    marker='o', markersize=3, label='Indoor (avg)')
            ax.fill_between(df['day'], df['outside'], df['indoor'],
                            color='#bdc3c7', alpha=0.25, label='Heating gap')
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax.set_title('Outside vs Indoor Temperature — Daily Average',
                         fontsize=14, fontweight='bold', pad=14)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right')
            fig.autofmt_xdate()

        elif chart_type == 'hvac_mode_distribution':
            df = pd.read_sql_query(
                """
                SELECT hvac_mode, COUNT(*) AS n
                FROM temperature_data
                GROUP BY hvac_mode
                ORDER BY n DESC
                """,
                conn,
            )
            if df.empty:
                return jsonify({'error': 'No temperature data available'}), 400
            mode_colors = {
                'off': '#7f8c8d',
                'idle': '#95a5a6',
                'heating': '#e74c3c',
                'cooling': '#3498db',
            }
            colors_list = [mode_colors.get(m, '#9b59b6') for m in df['hvac_mode']]
            total = df['n'].sum()
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(df['hvac_mode'].astype(str), df['n'].values,
                          color=colors_list, edgecolor='#2c3e50', linewidth=0.8)
            for bar, n in zip(bars, df['n'].values):
                pct = 100 * n / total
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + total * 0.005,
                        f'{n:,}\n({pct:.1f}%)',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.set_xlabel('HVAC Mode', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Measurements', fontsize=12, fontweight='bold')
            ax.set_title('HVAC Mode Distribution Across All Rooms',
                         fontsize=14, fontweight='bold', pad=14)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, df['n'].max() * 1.15)

        elif chart_type == 'avg_temp_by_floor':
            df = pd.read_sql_query(
                """
                SELECT floor,
                       AVG(room_temp)    AS room_temp,
                       AVG(setpoint)     AS setpoint,
                       AVG(outside_temp) AS outside_temp
                FROM temperature_data
                GROUP BY floor
                ORDER BY floor
                """,
                conn,
            )
            if df.empty:
                return jsonify({'error': 'No temperature data available'}), 400
            x = list(range(len(df)))
            width = 0.28
            fig, ax = plt.subplots(figsize=(12, 6))
            b1 = ax.bar([i - width for i in x], df['room_temp'].values,
                        width=width, color='#2980b9', edgecolor='#1f4e79',
                        label='Avg Indoor')
            b2 = ax.bar(x, df['setpoint'].values,
                        width=width, color='#27ae60', edgecolor='#1e6f44',
                        label='Avg Setpoint')
            b3 = ax.bar([i + width for i in x], df['outside_temp'].values,
                        width=width, color='#e67e22', edgecolor='#a04000',
                        label='Avg Outside')
            for bars in (b1, b2, b3):
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([f'Floor {int(f)}' for f in df['floor']])
            ax.set_xlabel('Floor', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax.set_title('Average Temperatures by Floor',
                         fontsize=14, fontweight='bold', pad=14)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right')

        elif chart_type == 'setpoint_deviation':
            df = pd.read_sql_query(
                """
                SELECT (room_temp - setpoint) AS deviation
                FROM temperature_data
                WHERE room_temp IS NOT NULL AND setpoint IS NOT NULL
                """,
                conn,
            )
            if df.empty:
                return jsonify({'error': 'No temperature data available'}), 400
            mean_dev = df['deviation'].mean()
            median_dev = df['deviation'].median()
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.hist(df['deviation'], bins=60, color='#9b59b6',
                    edgecolor='#5b2c6f', alpha=0.85)
            ax.axvline(0, color='black', linestyle='-', linewidth=1.4,
                       label='At setpoint (0°C)')
            ax.axvline(mean_dev, color='#e74c3c', linestyle='--', linewidth=1.4,
                       label=f'Mean: {mean_dev:+.2f}°C')
            ax.axvline(median_dev, color='#27ae60', linestyle='--', linewidth=1.4,
                       label=f'Median: {median_dev:+.2f}°C')
            ax.set_xlabel('Room Temp − Setpoint (°C)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Measurements', fontsize=12, fontweight='bold')
            ax.set_title('Deviation of Room Temperature from Setpoint',
                         fontsize=14, fontweight='bold', pad=14)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right')

        elif chart_type == 'hvac_by_room_state':
            df = pd.read_sql_query(
                """
                SELECT room_state, hvac_mode, COUNT(*) AS n
                FROM temperature_data
                GROUP BY room_state, hvac_mode
                """,
                conn,
            )
            if df.empty:
                return jsonify({'error': 'No temperature data available'}), 400
            pivot = (
                df.pivot(index='room_state', columns='hvac_mode', values='n')
                  .fillna(0)
            )
            mode_colors = {
                'off': '#7f8c8d',
                'idle': '#95a5a6',
                'heating': '#e74c3c',
                'cooling': '#3498db',
            }
            mode_order = [m for m in ['off', 'idle', 'heating', 'cooling'] if m in pivot.columns]
            mode_order += [m for m in pivot.columns if m not in mode_order]
            pivot = pivot[mode_order]
            # Convert to row-percentage so each room_state sums to 100%.
            pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(12, 6))
            bottom = [0] * len(pct.index)
            for mode in pct.columns:
                vals = pct[mode].values
                bars = ax.bar(pct.index, vals, bottom=bottom,
                              color=mode_colors.get(mode, '#9b59b6'),
                              edgecolor='white', linewidth=1.0, label=mode)
                for bar, v, b in zip(bars, vals, bottom):
                    if v >= 4:  # only annotate visible chunks
                        ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                                f'{v:.1f}%', ha='center', va='center',
                                fontsize=10, fontweight='bold', color='white')
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xlabel('Room State', fontsize=12, fontweight='bold')
            ax.set_ylabel('Share of Measurements (%)', fontsize=12, fontweight='bold')
            ax.set_title('HVAC Mode Mix by Room State',
                         fontsize=14, fontweight='bold', pad=14)
            ax.set_ylim(0, 100)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', title='HVAC Mode')

        else:
            return jsonify({'error': f'Unsupported temperature chart type: {chart_type}'}), 400

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return jsonify({'chart_url': f'data:image/png;base64,{image_base64}'})
    except Exception as e:
        return jsonify({'error': f'Failed to generate temperature chart: {str(e)}'}), 500
    finally:
        conn.close()


@app.route('/api/data', methods=['GET'])
def get_data():
    file = request.args.get('file')
    limit = request.args.get('limit', type=int)
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 20, type=int)
    search_columns = request.args.getlist('search_column')
    search_values = request.args.getlist('search_value')
    if not search_columns:
        search_column = request.args.get('search_column')
        if search_column:
            search_columns = [col.strip() for col in search_column.split(',') if col.strip()]
    if not search_values:
        search_value = request.args.get('search_value', '')
        if search_value:
            search_values = [search_value]
    if search_columns and len(search_values) == 1 and len(search_columns) > 1:
        search_values = search_values * len(search_columns)

    if not file:
        return jsonify({'error': 'Missing file name.'}), 400

    try:
        # Convert filename to table name
        table_name = file.replace('.csv', '').lower()
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

        if search_columns and search_values:
            if len(search_columns) != len(search_values):
                return jsonify({'error': 'Search columns and values count do not match.'}), 400

            if len(search_columns) == 1:
                data = search_table_data(table_name, search_columns[0], search_values[0], page, page_size)
            else:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                db_columns = [row[1] for row in cursor.fetchall()]
                display_columns = get_table_columns(table_name)

                mappings = []
                for search_col, search_val in zip(search_columns, search_values):
                    for i, display_col in enumerate(display_columns):
                        if display_col.lower() == search_col.lower():
                            mappings.append((db_columns[i], search_val))
                            break

                if not mappings:
                    conn.close()
                    return jsonify({'error': 'Search columns not found.'}), 400

                where_clauses = []
                params = []
                for db_col, value in mappings:
                    if db_col.lower() == 'timestamp' and value and not ':' in value:
                        # For timestamp with date-only, match entire day
                        where_clauses.append(f'"{db_col}" LIKE ?')
                        params.append(f"{value}%")
                    else:
                        where_clauses.append(f'"{db_col}" LIKE ?')
                        params.append(f"%{value}%")

                count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {' AND '.join(where_clauses)}"
                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]
                total_pages = (total_count + page_size - 1) // page_size
                offset = (page - 1) * page_size

                query = f"SELECT * FROM {table_name} WHERE {' AND '.join(where_clauses)} LIMIT ? OFFSET ?"
                cursor.execute(query, params + [page_size, offset])
                db_rows = cursor.fetchall()

                rows = []
                for db_row in db_rows:
                    row_dict = {}
                    for i, db_col in enumerate(db_columns):
                        display_col = display_columns[i]
                        row_dict[display_col] = db_row[i]
                    rows.append(row_dict)

                conn.close()
                data = {
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
            return jsonify(data)
        else:
            data = load_table_data_paginated(table_name, page, page_size)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/filter', methods=['POST'])
def post_filter():
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'Invalid JSON payload.'}), 400
    file = payload.get('file')
    filters = payload.get('filters', {})
    if not file:
        return jsonify({'error': 'Missing file name.'}), 400
    try:
        data = load_csv_file(DATA_DIR, file)
        filtered = {
            'columns': data['columns'],
            'rows': filter_rows(data['rows'], filters),
        }
        return jsonify(filtered)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download', methods=['GET'])
def download_data():
    file = request.args.get('file')
    fmt = request.args.get('format', 'csv')          # 'csv' or 'json'
    search_columns = request.args.getlist('search_column')
    search_values  = request.args.getlist('search_value')

    if not file:
        return jsonify({'error': 'Missing file name.'}), 400

    table_name = file.replace('.csv', '').lower()
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

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        db_columns      = [row[1] for row in cursor.fetchall()]
        display_columns = get_table_columns(table_name)

        if search_columns and search_values:
            # Map display column names → db column names
            mappings = []
            for s_col, s_val in zip(search_columns, search_values):
                for i, d_col in enumerate(display_columns):
                    if d_col.lower() == s_col.lower():
                        mappings.append((db_columns[i], s_val))
                        break

            if not mappings:
                conn.close()
                return jsonify({'error': 'Search columns not found.'}), 400

            where_clauses, params = [], []
            for db_col, value in mappings:
                if db_col.lower() == 'timestamp' and value and ':' not in value:
                    where_clauses.append(f'"{db_col}" LIKE ?')
                    params.append(f'{value}%')
                else:
                    where_clauses.append(f'"{db_col}" LIKE ?')
                    params.append(f'%{value}%')

            query = f"SELECT * FROM {table_name} WHERE {' AND '.join(where_clauses)}"
            cursor.execute(query, params)
        else:
            cursor.execute(f"SELECT * FROM {table_name}")

        db_rows = cursor.fetchall()
        conn.close()

        # Build list of dicts with display column names
        rows = [
            {display_columns[i]: db_row[i] for i in range(len(db_columns))}
            for db_row in db_rows
        ]

        label    = 'filtered' if (search_columns and search_values) else 'full'
        basename = table_name

        if fmt == 'csv':
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=display_columns)
            writer.writeheader()
            writer.writerows(rows)
            return Response(
                buf.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename="{basename}_{label}.csv"'}
            )

        elif fmt == 'json':
            payload = json.dumps({'columns': display_columns, 'rows': rows, 'count': len(rows)}, indent=2, default=str)
            return Response(
                payload,
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename="{basename}_{label}.json"'}
            )

        else:
            return jsonify({'error': f'Unsupported format: {fmt}. Use csv or json.'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['GET'])
def visualize_data():
    file = request.args.get('file')
    x_column = request.args.get('x_column')
    y_column = request.args.get('y_column')
    chart_type = request.args.get('chart_type', 'scatter')
    room_number = request.args.get('room_number')  # For PIR data filtering
    start_timestamp = request.args.get('start_timestamp')  # For PIR data filtering
    end_timestamp = request.args.get('end_timestamp')  # For PIR data filtering
    pir_mode = request.args.get('pir_mode', 'default')  # 'default' or 'activity_heatmap'
    lamp_location = request.args.get('lamp_location')  # For lightning data filtering

    if not file or not x_column or not y_column:
        return jsonify({'error': 'Missing required parameters: file, x_column, y_column'}), 400

    # Convert filename to table name
    table_name = file.replace('.csv', '').lower()
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

    try:
        # Weather chart routing — its own self-contained generator handles
        # all weather chart types so we can return early.
        if table_name == 'weather_antalya' and chart_type in (
            'yearly_trend', 'monthly_avg', 'temp_distribution', 'extremes'
        ):
            data = load_table_data(table_name)
            if not data['rows']:
                return jsonify({'error': 'No weather data available'}), 400
            df = pd.DataFrame(data['rows'])
            return generate_weather_chart(df, chart_type)

        # Temperature chart routing — uses SQL aggregation directly because
        # this dataset has ~900k rows and we don't want to materialize them
        # all in Python.
        if table_name == 'temperature_data' and chart_type in (
            'temp_outside_indoor', 'hvac_mode_distribution', 'avg_temp_by_floor',
            'setpoint_deviation', 'hvac_by_room_state'
        ):
            return generate_temperature_chart(chart_type)

        if table_name == 'temperature_data' and chart_type == 'daily_temp_trend':
            date_label = (start_timestamp or '')[:10]
            return generate_daily_temperature_trend(room_number, date_label)

        # Load data based on filters
        if table_name == 'pir_sensor_data' and (room_number or start_timestamp or end_timestamp):
            # Use filtered PIR data
            rows = load_pir_data_filtered(room_number, start_timestamp, end_timestamp)
            if not rows:
                return jsonify({'error': 'No data found for the specified filters'}), 400
            df = pd.DataFrame(rows)
            
            # Special handling for activity heatmap visualization
            if pir_mode == 'activity_heatmap':
                return generate_pir_activity_heatmap(df, room_number, start_timestamp, end_timestamp)
        elif table_name == 'lightning_data' and (room_number or start_timestamp or end_timestamp):
            rows = load_lightning_data_filtered(room_number, start_timestamp, end_timestamp, lamp_location)
            if not rows:
                return jsonify({'error': 'No data found for the specified filters'}), 400
            df = pd.DataFrame(rows)
            if chart_type == 'heatmap':
                return generate_lightning_value_heatmap(df, room_number, start_timestamp, end_timestamp, lamp_location)
            if chart_type == 'daily_trend':
                date_label = start_timestamp[:10] if start_timestamp else ''
                return generate_lightning_daily_trend(df, room_number, date_label, lamp_location)
        else:
            # Load all data for visualization (no pagination for charts)
            data = load_table_data(table_name)

            if not data['rows']:
                return jsonify({'error': 'No data available for visualization'}), 400

            # Convert to DataFrame
            df = pd.DataFrame(data['rows'])

        # Clean column names for DataFrame access
        df.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df.columns]

        # Map display column names back to DataFrame column names
        x_col_clean = x_column.replace(' ', '_').replace('-', '_').lower()
        y_col_clean = y_column.replace(' ', '_').replace('-', '_').lower()

        if x_col_clean not in df.columns or y_col_clean not in df.columns:
            return jsonify({'error': f'Columns not found: {x_column}, {y_column}'}), 400

        if x_col_clean == 'timestamp':
            df[x_col_clean] = pd.to_datetime(df[x_col_clean], errors='coerce')
        if y_col_clean == 'timestamp':
            df[y_col_clean] = pd.to_datetime(df[y_col_clean], errors='coerce')

        if table_name == 'lightning_data' and x_col_clean == 'timestamp' and y_col_clean == 'value':
            df[y_col_clean] = pd.to_numeric(df[y_col_clean], errors='coerce')
            if chart_type == 'bar':
                df['date'] = df[x_col_clean].dt.strftime('%d.%m.%Y')
                df = df.dropna(subset=['date', y_col_clean])
                daily = df.groupby('date')[y_col_clean].mean().reset_index()
                daily.columns = ['date', 'avg_value']
                fig, ax = plt.subplots(figsize=(14, 6))
                bars = ax.bar(daily['date'], daily['avg_value'], color='#3498db', edgecolor='#2980b9', linewidth=0.8)
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average Lighting Value (0–100)', fontsize=12, fontweight='bold')
                ax.set_title('Daily Average Lightning Value', fontsize=14, fontweight='bold', pad=16)
                ax.set_ylim(0, 100)
                ax.set_xticks(range(len(daily)))
                ax.set_xticklabels(daily['date'], rotation=45, ha='right')
                ax.grid(True, axis='y', alpha=0.3, linestyle='--')
                for bar, val in zip(bars, daily['avg_value']):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=8)
                plt.tight_layout()
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                return jsonify({'chart_url': f'data:image/png;base64,{image_base64}'})
            else:
                df['time_of_day'] = df[x_col_clean].dt.hour + df[x_col_clean].dt.minute / 60.0
                df = df.dropna(subset=['time_of_day', y_col_clean])
                x_col_clean = 'time_of_day'
                x_column = 'Time of Day'
                y_column = 'Lighting Value'

        # Generate chart
        plt.figure(figsize=(10, 6))

        if chart_type == 'scatter':
            plt.scatter(df[x_col_clean], df[y_col_clean], alpha=0.6)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f'Scatter Plot: {x_column} vs {y_column}')

        elif chart_type == 'line':
            plt.plot(df[x_col_clean], df[y_col_clean], marker='o', linestyle='-', alpha=0.8)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f'Line Chart: {x_column} vs {y_column}')
            if table_name == 'lightning_data' and y_col_clean == 'value':
                plt.ylim(0, 100)
                plt.xlim(0, 24)
                plt.xticks(range(0, 25, 2), [f'{h:02d}:00' for h in range(0, 25, 2)])
                plt.grid(True, alpha=0.25)

        elif chart_type == 'bar':
            # For bar charts, we need to group by x and aggregate y
            if df[x_col_clean].dtype == 'object' or len(df[x_col_clean].unique()) < 20:
                df_clean = df[df[x_col_clean].notna() & (df[x_col_clean].astype(str).str.strip() != '') & (df[x_col_clean].astype(str) != 'None')].copy()
                if df_clean.empty:
                    return jsonify({'error': 'No valid data to display after filtering invalid X values'}), 400

                # Special handling for persona vs pir_motion: calculate average per guest
                if x_col_clean == 'persona' and y_col_clean == 'pir_motion':
                    # Check if guest_id column exists
                    guest_col = 'guest_id'
                    if guest_col not in df_clean.columns:
                        # Fallback to total if no guest_id
                        grouped = df_clean.groupby(x_col_clean)[y_col_clean].sum().reset_index()
                        ylabel = f'Total {y_column}'
                        title = f'Bar Chart: Total {y_column} by {x_column}'
                    else:
                        # Calculate average per guest
                        grouped = (
                            df_clean
                            .groupby(x_col_clean)
                            .agg(
                                total_motion=(y_col_clean, 'sum'),
                                guest_count=(guest_col, 'nunique')
                            )
                            .assign(avg_motion_per_guest=lambda d: d['total_motion'] / d['guest_count'])
                            .reset_index()
                        )
                        ylabel = f'Average {y_column} per Guest'
                        title = f'Bar Chart: Average {y_column} per Guest by {x_column}'
                        y_col_clean = 'avg_motion_per_guest'  # Use the calculated column
                else:
                    grouped = df_clean.groupby(x_col_clean)[y_col_clean].sum().reset_index()
                    ylabel = f'Total {y_column}'
                    title = f'Bar Chart: Total {y_column} by {x_column}'

                if grouped.empty:
                    return jsonify({'error': 'No data available after filtering'}), 400
                plt.bar(grouped[x_col_clean].astype(str), grouped[y_col_clean])
                plt.xlabel(x_column)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.xticks(rotation=45, ha='right')
            else:
                return jsonify({'error': 'Bar chart requires categorical X-axis or fewer than 20 unique values'}), 400

        elif chart_type == 'histogram':
            plt.hist(df[y_col_clean], bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel(y_column)
            plt.ylabel('Frequency')
            plt.title(f'Histogram: {y_column}')

        else:
            return jsonify({'error': f'Unsupported chart type: {chart_type}'}), 400

        plt.tight_layout()

        # Convert plot to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        chart_url = f'data:image/png;base64,{image_base64}'

        return jsonify({'chart_url': chart_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _fig_to_data_url(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{encoded}'


def _build_energy_charts(result, room_number, start_ts, end_ts):
    """Render the four PNGs the frontend shows for an energy report."""
    df = result['df']
    by_lamp = result['by_lamp']
    title_suffix = f"Room {room_number}  |  {start_ts}  ->  {end_ts}"
    actual_color = '#2c7fb8'
    max_color = '#e36b6b'
    charts = {}

    # 1. Total actual vs max
    fig, ax = plt.subplots(figsize=(7, 4.5))
    totals = [result['summary']['actual_wh'], result['summary']['max_wh']]
    bars = ax.bar(['Actual', 'Max-level (no dimming)'], totals,
                  color=[actual_color, max_color], edgecolor='#2c3e50',
                  linewidth=0.8)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f'{v:.1f} Wh', ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    ax.set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
    ax.set_title('Total energy: actual vs max-level\n' + title_suffix,
                 fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(totals) * 1.15 if max(totals) > 0 else 1)
    fig.tight_layout()
    charts['total'] = _fig_to_data_url(fig)

    # 2. Per-lamp comparison
    if not by_lamp.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(by_lamp))
        w = 0.4
        ax.bar(x - w / 2, by_lamp['E_actual_Wh'], w,
               label='Actual', color=actual_color, edgecolor='#1f4e79')
        ax.bar(x + w / 2, by_lamp['E_max_Wh'], w,
               label='Max-level', color=max_color, edgecolor='#922b21')
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace('_', ' ').title()
                            for l in by_lamp.index],
                           rotation=30, ha='right')
        ax.set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
        ax.set_title('Per-lamp energy: actual vs max-level\n' + title_suffix,
                     fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        fig.tight_layout()
        charts['per_lamp'] = _fig_to_data_url(fig)

    # 3. Time series (hourly or daily depending on span)
    if not df.empty:
        ts = (df.set_index('timestamp')[['E_actual_Wh', 'E_max_Wh']]
                .sort_index())
        span_hours = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() / 3600.0
        rule = 'h' if span_hours <= 72 else 'D'
        bucket = ts.resample(rule).sum()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(bucket.index, bucket['E_actual_Wh'], marker='o',
                color=actual_color, linewidth=1.8, label='Actual')
        ax.plot(bucket.index, bucket['E_max_Wh'], marker='s',
                color=max_color, linewidth=1.8, label='Max-level')
        ax.fill_between(bucket.index, bucket['E_actual_Wh'],
                        bucket['E_max_Wh'], alpha=0.18, color=max_color,
                        label='Savings from dimming')
        ax.set_ylabel(f"Energy per {'hour' if rule == 'h' else 'day'} (Wh)",
                      fontsize=12, fontweight='bold')
        ax.set_title(f"Energy over time ({'hourly' if rule == 'h' else 'daily'})\n"
                     + title_suffix,
                     fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        fig.autofmt_xdate()
        fig.tight_layout()
        charts['timeseries'] = _fig_to_data_url(fig)

    # 4. Dimmable-only savings (per dimmable lamp: actual vs max + saved %)
    if not by_lamp.empty:
        dimmable_only = by_lamp[by_lamp.index.isin(DIMMABLE_LAMPS)]
        if not dimmable_only.empty:
            dim_actual_total = float(dimmable_only['E_actual_Wh'].sum())
            dim_max_total = float(dimmable_only['E_max_Wh'].sum())
            dim_saved_total = dim_max_total - dim_actual_total
            dim_saved_pct = (100.0 * dim_saved_total / dim_max_total) if dim_max_total else 0.0

            fig, ax = plt.subplots(figsize=(11, 5))
            x = np.arange(len(dimmable_only))
            w = 0.4
            ax.bar(x - w / 2, dimmable_only['E_actual_Wh'], w,
                   label='Actual', color=actual_color, edgecolor='#1f4e79')
            ax.bar(x + w / 2, dimmable_only['E_max_Wh'], w,
                   label='Max-level', color=max_color, edgecolor='#922b21')
            # Annotate each dimmable lamp with its savings %
            for i, lamp in enumerate(dimmable_only.index):
                actual_v = float(dimmable_only.loc[lamp, 'E_actual_Wh'])
                max_v = float(dimmable_only.loc[lamp, 'E_max_Wh'])
                saved_pct = (100.0 * (max_v - actual_v) / max_v) if max_v else 0.0
                top = max(actual_v, max_v)
                ax.text(i, top, f'-{saved_pct:.1f}%',
                        ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color='#1e6f44')
            ax.set_xticks(x)
            ax.set_xticklabels([l.replace('_', ' ').title()
                                for l in dimmable_only.index],
                               rotation=30, ha='right')
            ax.set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
            ax.set_title(
                f'Dimmable lamps — energy savings vs max-level\n'
                f'Saved {dim_saved_total:.1f} Wh ({dim_saved_pct:.1f}%)  |  ' + title_suffix,
                fontsize=13, fontweight='bold', pad=12,
            )
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right')
            top_y = float(dimmable_only[['E_actual_Wh', 'E_max_Wh']].values.max())
            ax.set_ylim(0, top_y * 1.18 if top_y > 0 else 1)
            fig.tight_layout()
            charts['dimmable_savings'] = _fig_to_data_url(fig)

    # 5. Dimmable LED vs non-dimmable bulb contribution
    df_typed = df.copy()
    df_typed['lamp_type'] = np.where(
        df_typed['lamp_location'].isin(DIMMABLE_LAMPS),
        'Dimmable LED', 'Non-dimmable bulb')
    grp = df_typed.groupby('lamp_type')[['E_actual_Wh', 'E_max_Wh']].sum()
    if not grp.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = np.arange(len(grp))
        w = 0.4
        ax.bar(x - w / 2, grp['E_actual_Wh'], w,
               label='Actual', color=actual_color, edgecolor='#1f4e79')
        ax.bar(x + w / 2, grp['E_max_Wh'], w,
               label='Max-level', color=max_color, edgecolor='#922b21')
        for i, lamp_type in enumerate(grp.index):
            for offset, col in ((-w / 2, 'E_actual_Wh'),
                                (w / 2, 'E_max_Wh')):
                v = grp.loc[lamp_type, col]
                ax.text(i + offset, v, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(grp.index)
        ax.set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
        ax.set_title('LED vs bulb contribution\n' + title_suffix,
                     fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        fig.tight_layout()
        charts['lamp_type'] = _fig_to_data_url(fig)

    return charts


@app.route('/api/energy', methods=['GET'])
def get_energy():
    room_number = request.args.get('room_number', type=int)
    start_ts = request.args.get('start_timestamp')
    end_ts = request.args.get('end_timestamp')

    if not room_number or not start_ts or not end_ts:
        return jsonify({'error': 'room_number, start_timestamp and '
                                  'end_timestamp are required.'}), 400
    if end_ts <= start_ts:
        return jsonify({'error': 'end_timestamp must be after '
                                  'start_timestamp.'}), 400

    try:
        result = compute_energy_for_room(room_number, start_ts, end_ts)
    except Exception as e:
        return jsonify({'error': f'Energy calculation failed: {e}'}), 500

    if result.get('empty'):
        return jsonify({'error': 'No lamp activity found for the chosen '
                                  'room and interval.'}), 400

    try:
        ai_recommendation_energy = compute_lightning_recommendation_energy_for_room(
            room_number,
            start_ts,
            end_ts,
        )
    except Exception as e:
        ai_recommendation_energy = {
            'empty': True,
            'message': f'AI lighting recommendation energy comparison failed: {e}',
        }

    charts = _build_energy_charts(result, room_number, start_ts, end_ts)
    return jsonify({
        'summary': result['summary'],
        'dimmable_summary': result['dimmable_summary'],
        'ai_recommendation_energy': ai_recommendation_energy,
        'by_lamp': result['by_lamp_records'],
        'charts': charts,
        'classification': {
            'dimmable': sorted(DIMMABLE_LAMPS),
            'non_dimmable': sorted(NON_DIMMABLE_LAMPS),
        },
    })


def _build_hvac_charts(result, room_number, start_ts, end_ts):
    """Render PNGs for the HVAC energy report."""
    df = result['df']
    summary = result['summary']
    title_suffix = f"Room {room_number}  |  {start_ts}  ->  {end_ts}"
    mode_colors = {
        'heating': '#e74c3c',
        'cooling': '#3498db',
        'off': '#95a5a6',
    }
    actual_color = '#2c7fb8'
    max_color = '#e36b6b'
    charts = {}

    # 0. Actual vs Max-baseline total
    fig, ax = plt.subplots(figsize=(7, 4.5))
    totals = [summary['total_wh'], summary['max_wh']]
    bars = ax.bar(['Actual', f"Max ({summary['dominant_mode']} continuous)"],
                  totals, color=[actual_color, max_color],
                  edgecolor='#2c3e50', linewidth=0.8)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f'{v:.0f} Wh', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
    ax.set_title(
        f"HVAC: actual vs always-on baseline\n"
        f"Saved {summary['saved_wh']:.0f} Wh ({summary['saved_pct']:.1f}%)  |  "
        + title_suffix,
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    top = float(max(totals)) if max(totals) > 0 else 1.0
    ax.set_ylim(0, top * 1.18)
    fig.tight_layout()
    charts['actual_vs_max'] = _fig_to_data_url(fig)

    # 1. Energy by mode (heating / cooling)
    if not df.empty:
        active = df[df['hvac_mode'].isin(['heating', 'cooling'])]
        per_mode = (active.groupby('hvac_mode')['energy_wh'].sum()
                          .reindex(['heating', 'cooling'], fill_value=0.0))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(['Heating', 'Cooling'], per_mode.values,
                      color=[mode_colors['heating'], mode_colors['cooling']],
                      edgecolor='#2c3e50', linewidth=0.8)
        for bar, v in zip(bars, per_mode.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f'{v:.0f} Wh', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
        ax.set_title('HVAC energy by mode\n' + title_suffix,
                     fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        top = float(max(per_mode.values.max(), 1.0))
        ax.set_ylim(0, top * 1.18)
        fig.tight_layout()
        charts['by_mode'] = _fig_to_data_url(fig)

    # 2. Active vs idle time share (minutes)
    if not df.empty:
        mode_counts = df['hvac_mode'].value_counts()
        labels, values, cols = [], [], []
        for mode in ['heating', 'cooling', 'off']:
            if mode in mode_counts:
                labels.append(mode.title())
                values.append(int(mode_counts[mode]) * HVAC_SAMPLE_MINUTES)
                cols.append(mode_colors[mode])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.pie(values, labels=labels, colors=cols, autopct='%1.1f%%',
               startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.2})
        ax.set_title('Time share by HVAC mode\n' + title_suffix,
                     fontsize=13, fontweight='bold', pad=12)
        fig.tight_layout()
        charts['time_share'] = _fig_to_data_url(fig)

    # 3. Energy over time (hourly or daily) with max baseline overlaid
    if not df.empty:
        ts = df.set_index('timestamp')[['energy_wh']].sort_index()
        span_hours = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() / 3600.0
        rule = 'h' if span_hours <= 72 else 'D'
        bucket = ts.resample(rule).sum()
        # Max-per-bucket = rated dominant power kept on for the full bucket.
        bucket_hours = 1.0 if rule == 'h' else 24.0
        max_per_bucket = summary['max_power_w'] * bucket_hours
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(bucket.index, bucket['energy_wh'].values,
               color=actual_color, edgecolor='#1f4e79', alpha=0.85,
               width=(0.8 / 24 if rule == 'h' else 0.8),
               label='Actual')
        ax.axhline(max_per_bucket, color=max_color, linewidth=2.0,
                   linestyle='--',
                   label=f"Max ({summary['dominant_mode']} continuous): {max_per_bucket:.0f} Wh")
        ax.set_ylabel(f"Energy per {'hour' if rule == 'h' else 'day'} (Wh)",
                      fontsize=12, fontweight='bold')
        ax.set_title(f"HVAC energy over time ({'hourly' if rule == 'h' else 'daily'})\n"
                     + title_suffix,
                     fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        fig.autofmt_xdate()
        fig.tight_layout()
        charts['timeseries'] = _fig_to_data_url(fig)

    return charts


@app.route('/api/hvac_energy', methods=['GET'])
def get_hvac_energy():
    room_number = request.args.get('room_number', type=int)
    start_ts = request.args.get('start_timestamp')
    end_ts = request.args.get('end_timestamp')

    if not room_number or not start_ts or not end_ts:
        return jsonify({'error': 'room_number, start_timestamp and '
                                  'end_timestamp are required.'}), 400
    if end_ts <= start_ts:
        return jsonify({'error': 'end_timestamp must be after '
                                  'start_timestamp.'}), 400

    try:
        result = compute_hvac_energy_for_room(room_number, start_ts, end_ts)
    except Exception as e:
        return jsonify({'error': f'HVAC energy calculation failed: {e}'}), 500

    if result.get('empty'):
        return jsonify({'error': 'No HVAC data found for the chosen '
                                  'room and interval.'}), 400

    try:
        ai_recommendation_energy = compute_tempreture_recomendation_energy_for_room(
            room_number,
            start_ts,
            end_ts,
        )
    except Exception as e:
        ai_recommendation_energy = {
            'empty': True,
            'message': f'AI recommendation energy comparison failed: {e}',
        }

    charts = _build_hvac_charts(result, room_number, start_ts, end_ts)
    return jsonify({
        'summary': result['summary'],
        'ai_recommendation_energy': ai_recommendation_energy,
        'by_mode': result['by_mode_records'],
        'rated_power_w': result['rated_power_w'],
        'sample_minutes': HVAC_SAMPLE_MINUTES,
        'charts': charts,
    })


@app.route('/api/lightning_recommendation', methods=['POST'])
def get_lightning_recommendation():
    payload = request.get_json(silent=True) or {}
    if not payload.get('room_number') or not payload.get('timestamp'):
        return jsonify({'error': 'room_number and timestamp are required.'}), 400

    try:
        if payload.get('use_model_predictions'):
            occupancy_result = predict_occupancy({
                **payload,
                'lookback_hours': 1,
                'horizon_minutes': 60,
            })
            persona_result = predict_lighting_persona(payload)
            payload = {
                **payload,
                'occupancy_prediction': occupancy_result['prediction'],
                'lighting_persona_prediction': persona_result['prediction'],
                'model_predictions': {
                    'occupancy': occupancy_result,
                    'lighting_persona': persona_result,
                },
            }
        result = compute_lightning_recommendation(payload)
    except Exception as e:
        return jsonify({'error': f'Lightning recommendation failed: {e}'}), 500

    if result.get('empty'):
        return jsonify({'error': result.get('message', 'No lighting history found.')}), 400
    return jsonify(result)


@app.route('/api/predict_occupancy', methods=['POST'])
def get_occupancy_prediction():
    payload = request.get_json(silent=True) or {}
    if not payload.get('room_number') or not payload.get('timestamp'):
        return jsonify({'error': 'room_number and timestamp are required.'}), 400
    try:
        return jsonify(predict_occupancy(payload))
    except Exception as e:
        return jsonify({'error': f'Occupancy prediction failed: {e}'}), 500


@app.route('/api/predict_lighting_persona', methods=['POST'])
def get_lighting_persona_prediction():
    payload = request.get_json(silent=True) or {}
    if not payload.get('room_number') or not payload.get('timestamp'):
        return jsonify({'error': 'room_number and timestamp are required.'}), 400
    try:
        return jsonify(predict_lighting_persona(payload))
    except Exception as e:
        return jsonify({'error': f'Lighting persona prediction failed: {e}'}), 500


@app.route('/api/predict_tempreture_persona', methods=['POST'])
def get_tempreture_persona_prediction():
    payload = request.get_json(silent=True) or {}
    if not payload.get('room_number') or not payload.get('timestamp'):
        return jsonify({'error': 'room_number and timestamp are required.'}), 400

    try:
        return jsonify(predict_tempreture_persona(payload))
    except Exception as e:
        return jsonify({'error': f'Temperature persona prediction failed: {e}'}), 500


@app.route('/api/tempreture_recomendation', methods=['POST'])
def get_tempreture_recomendation():
    payload = request.get_json(silent=True) or {}
    if not payload.get('room_number') or not payload.get('timestamp'):
        return jsonify({'error': 'room_number and timestamp are required.'}), 400

    try:
        if payload.get('use_model_predictions'):
            occupancy_result = predict_occupancy({
                **payload,
                'lookback_hours': 1,
                'horizon_minutes': 60,
            })
            persona_result = predict_tempreture_persona(payload)
            payload = {
                **payload,
                'occupancy_prediction': occupancy_result['prediction'],
                'temperature_persona_prediction': persona_result['prediction'],
                'model_predictions': {
                    'occupancy': occupancy_result,
                    'temperature_persona': persona_result,
                },
            }
        result = predict_tempreture_recomendation(payload)
    except Exception as e:
        return jsonify({'error': f'Temperature recommendation failed: {e}'}), 500

    if result.get('empty'):
        return jsonify({'error': result.get('message', 'No temperature history found.')}), 400
    return jsonify(result)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path and (FRONTEND_DIST / path).exists():
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, 'index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
