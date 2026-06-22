import { useEffect, useRef, useState } from 'react';
import './App.css';

function App() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [columns, setColumns] = useState([]);
  const [rows, setRows] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [totalPages, setTotalPages] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [hasPrev, setHasPrev] = useState(false);
  const [searchColumns, setSearchColumns] = useState([]);
  const [searchValues, setSearchValues] = useState({});
  const [isSearching, setIsSearching] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [status, setStatus] = useState('Loading files...');

  const readJsonResponse = async (response) => {
    const text = await response.text();
    let data = {};

    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error('The server returned an unreadable response.');
      }
    }

    if (!response.ok) {
      throw new Error(data.error || `Request failed with status ${response.status}.`);
    }

    return data;
  };

  // Predefined visualizations
  const predefinedVisualizations = {
    'hotelReservationData.csv': [
      {
        id: 'price_vs_nights',
        name: 'Price vs Nights Stay',
        description: 'Total amount vs total nights - shows pricing patterns',
        x_column: 'Total Amount',
        y_column: 'Total Nights',
        chart_type: 'scatter'
      },
      {
        id: 'room_type_pricing',
        name: 'Room Type Pricing',
        description: 'Average price by room type',
        x_column: 'Room Type',
        y_column: 'Total Amount',
        chart_type: 'bar'
      },
      {
        id: 'payment_status',
        name: 'Payment Status Distribution',
        description: 'Distribution of payment statuses',
        x_column: 'Payment Status',
        y_column: 'Total Amount',
        chart_type: 'bar'
      },
      {
        id: 'booking_trends',
        name: 'Booking Trends Over Time',
        description: 'Total amount trends by check-in date',
        x_column: 'Check-in Date',
        y_column: 'Total Amount',
        chart_type: 'line'
      }
    ],
    'PIRSensorData.csv': [
      {
        id: 'persona_activity_comparison',
        name: 'PIR Activity by Persona Type',
        description: 'Average PIR motion activity per guest compared by persona type - shows which persona types move the most on average',
        x_column: 'persona',
        y_column: 'pir_motion',
        chart_type: 'bar'
      },
      {
        id: 'room_state_distribution',
        name: 'Room State Distribution',
        description: 'Distribution of room states',
        x_column: 'room_state',
        y_column: 'pir_motion',
        chart_type: 'bar'
      },
      {
        id: 'motion_by_room',
        name: 'Motion by Room State',
        description: 'Motion patterns grouped by room state',
        x_column: 'room_state',
        y_column: 'pir_motion',
        chart_type: 'bar'
      }
    ],
    'lightningData.csv': [
      {
        id: 'light_value_over_time',
        name: 'Lightning Value Over Time',
        description: 'Shows the lightning sensor value changes over time for each room',
        x_column: 'timestamp',
        y_column: 'Value',
        chart_type: 'bar'
      },
      {
        id: 'lamp_location_comparison',
        name: 'Lamp Location Comparison',
        description: 'Compare measured lightning values across lamp locations',
        x_column: 'lamp_location',
        y_column: 'Value',
        chart_type: 'bar'
      },
      {
        id: 'reservation_light_scatter',
        name: 'Reservation Active vs Lightning Value',
        description: 'Scatter plot of lightning value by active reservation status',
        x_column: 'reservation_active',
        y_column: 'Value',
        chart_type: 'scatter'
      }
    ],
    'WheatherDataAntalya.csv': [
      {
        id: 'weather_yearly_trend',
        name: 'Yearly Temperature Trend',
        description: 'Daily max and min temperatures across the whole year, with the daily range shaded between them',
        x_column: 'date',
        y_column: 'max_temp',
        chart_type: 'yearly_trend'
      },
      {
        id: 'weather_monthly_avg',
        name: 'Monthly Average Temperatures',
        description: 'Side-by-side bars of average max and min temperature for each month',
        x_column: 'date',
        y_column: 'max_temp',
        chart_type: 'monthly_avg'
      },
      {
        id: 'weather_temp_distribution',
        name: 'Temperature Distribution',
        description: 'How often each temperature occurs across the year (max and min, with mean lines)',
        x_column: 'date',
        y_column: 'max_temp',
        chart_type: 'temp_distribution'
      },
      {
        id: 'weather_extremes',
        name: 'Hottest & Coldest Days',
        description: 'Top 10 hottest days and top 10 coldest days of the year',
        x_column: 'date',
        y_column: 'max_temp',
        chart_type: 'extremes'
      }
    ],
    'temperatureData.csv': [
      {
        id: 'temp_outside_indoor',
        name: 'Outside vs Indoor Temperature',
        description: 'Daily average outside temperature vs daily average room temperature, with the heating gap shaded between them',
        x_column: 'timestamp',
        y_column: 'room_temp',
        chart_type: 'temp_outside_indoor'
      },
      {
        id: 'hvac_mode_distribution',
        name: 'HVAC Mode Distribution',
        description: 'How often each HVAC mode (off, idle, heating, cooling) was active across all rooms',
        x_column: 'hvac_mode',
        y_column: 'count',
        chart_type: 'hvac_mode_distribution'
      },
      {
        id: 'avg_temp_by_floor',
        name: 'Average Temperatures by Floor',
        description: 'Compare average indoor, setpoint, and outside temperatures floor by floor',
        x_column: 'floor',
        y_column: 'room_temp',
        chart_type: 'avg_temp_by_floor'
      },
      {
        id: 'setpoint_deviation',
        name: 'Setpoint Deviation Histogram',
        description: 'How far rooms drift from their setpoint (room_temp − setpoint), with mean and median markers',
        x_column: 'room_temp',
        y_column: 'setpoint',
        chart_type: 'setpoint_deviation'
      },
      {
        id: 'hvac_by_room_state',
        name: 'HVAC Mode Mix by Room State',
        description: 'For each room state (Vacant, Occupied, Cleaning), the share of time spent in each HVAC mode',
        x_column: 'room_state',
        y_column: 'hvac_mode',
        chart_type: 'hvac_by_room_state'
      }
    ]
  };

  // Function to get appropriate input type for search inputs
  const getInputType = (column) => {
    const lowerColumn = column.toLowerCase();
    if (lowerColumn.includes('timestamp')) return 'date';
    if (lowerColumn.includes('date') && !lowerColumn.includes('time')) return 'date';
    if (lowerColumn.includes('time') && !lowerColumn.includes('date')) return 'time';
    if (['room_number', 'adults', 'children', 'guest_id', 'pir_motion', 'floor_number', 'total_nights', 'total_amount', 'value'].some(numCol => lowerColumn.includes(numCol))) return 'number';
    return 'text';
  };

  // Function to format search values for backend
  const formatValue = (column, value) => {
    if (!value) return value;
    const lowerColumn = column.toLowerCase();
    if (lowerColumn.includes('timestamp')) {
      // date input gives YYYY-MM-DD, already correct for day search
      return value;
    }
    if (lowerColumn.includes('date') && !lowerColumn.includes('time')) {
      // date input gives YYYY-MM-DD, already correct
      return value;
    }
    if (lowerColumn.includes('time') && !lowerColumn.includes('date')) {
      // time input gives HH:MM, add :00 for seconds
      return value + ':00';
    }
    return value;
  };

  // Visualization state
  const [selectedVisualization, setSelectedVisualization] = useState(null);
  const [chartImage, setChartImage] = useState('');
  const [isGeneratingChart, setIsGeneratingChart] = useState(false);
  const [showChart, setShowChart] = useState(true);

  // Room-specific PIR analysis state
  const [roomNumber, setRoomNumber] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [availableRooms, setAvailableRooms] = useState([]);
  const [pirMode, setPirMode] = useState('activity_heatmap'); // 'default' or 'activity_heatmap'
  const [lampLocation, setLampLocation] = useState('');
  const [availableLampLocations] = useState([
    'bed_left', 'bed_right', 'cabinet', 'closet', 'corridor_left',
    'corridor_right', 'dinner_table', 'hidden_top', 'shower', 'sink', 'table'
  ]);

  // Daily lightning trend state
  const [dailyRoomNumber, setDailyRoomNumber] = useState('');
  const [dailyDate, setDailyDate] = useState('');
  const [dailyLampLocation, setDailyLampLocation] = useState('');
  const [dailyChartImage, setDailyChartImage] = useState('');
  const [showDailyChart, setShowDailyChart] = useState(true);
  const [isGeneratingDailyChart, setIsGeneratingDailyChart] = useState(false);

  // Daily temperature trend state
  const [dailyTempRoomNumber, setDailyTempRoomNumber] = useState('');
  const [dailyTempDate, setDailyTempDate] = useState('');
  const [dailyTempChartImage, setDailyTempChartImage] = useState('');
  const [showDailyTempChart, setShowDailyTempChart] = useState(true);
  const [isGeneratingDailyTempChart, setIsGeneratingDailyTempChart] = useState(false);

  // Energy-consumption state (lightningData.csv)
  const [energyRoomNumber, setEnergyRoomNumber] = useState('');
  const [energyStartDate, setEnergyStartDate] = useState('');
  const [energyEndDate, setEnergyEndDate] = useState('');
  const [energyResult, setEnergyResult] = useState(null);
  const [showEnergy, setShowEnergy] = useState(true);
  const [isGeneratingEnergy, setIsGeneratingEnergy] = useState(false);

  // Lightning recommendation state
  const [recommendationRoomNumber, setRecommendationRoomNumber] = useState('1');
  const [recommendationTimestamp, setRecommendationTimestamp] = useState('2022-01-22T08:25');
  const [recommendationOccupancy, setRecommendationOccupancy] = useState('Occupied');
  const [occupancyModelType, setOccupancyModelType] = useState('random_forest');
  const [recommendationPersona, setRecommendationPersona] = useState('Routine');
  const [lightingPersonaModelType, setLightingPersonaModelType] = useState('random_forest');
  const [lightingRecommendationModelType, setLightingRecommendationModelType] = useState('hist_gradient_boosting');
  const [recommendationAdults, setRecommendationAdults] = useState('2');
  const [recommendationChildren, setRecommendationChildren] = useState('0');
  const [recommendationNationality, setRecommendationNationality] = useState('');
  const [recommendationRoomType, setRecommendationRoomType] = useState('Deluxe');
  const [recommendationLookback, setRecommendationLookback] = useState('24');
  const [recommendationResult, setRecommendationResult] = useState(null);
  const [showRecommendation, setShowRecommendation] = useState(true);
  const [isGeneratingRecommendation, setIsGeneratingRecommendation] = useState(false);
  const [occupancyPredictionResult, setOccupancyPredictionResult] = useState(null);
  const [personaPredictionResult, setPersonaPredictionResult] = useState(null);
  const [isPredictingOccupancy, setIsPredictingOccupancy] = useState(false);
  const [isPredictingPersona, setIsPredictingPersona] = useState(false);
  const recommendationResultRef = useRef(null);

  // HVAC energy state (temperatureData.csv)
  const [hvacRoomNumber, setHvacRoomNumber] = useState('');
  const [hvacStartDate, setHvacStartDate] = useState('');
  const [hvacEndDate, setHvacEndDate] = useState('');
  const [hvacResult, setHvacResult] = useState(null);
  const [showHvac, setShowHvac] = useState(true);
  const [isGeneratingHvac, setIsGeneratingHvac] = useState(false);

  // Temperature persona prediction state
  const [tempPersonaRoomNumber, setTempPersonaRoomNumber] = useState('1');
  const [tempPersonaTimestamp, setTempPersonaTimestamp] = useState('2022-01-22T08:25');
  const [tempPersonaLookback, setTempPersonaLookback] = useState('2');
  const [temperatureRecommendationModelType, setTemperatureRecommendationModelType] = useState('transformer');
  const [tempPersonaResult, setTempPersonaResult] = useState(null);
  const [isPredictingTempPersona, setIsPredictingTempPersona] = useState(false);
  const [tempRecommendationResult, setTempRecommendationResult] = useState(null);
  const [isGeneratingTempRecommendation, setIsGeneratingTempRecommendation] = useState(false);

  useEffect(() => {
    fetch('/api/files')
      .then((response) => response.json())
      .then((data) => {
        setFiles(data.files || []);
        if (data.files?.length) {
          setSelectedFile(data.files[0]);
          setStatus('Choose a dataset to view.');
        } else {
          setStatus('No CSV files found in Data/.');
        }
      })
      .catch(() => setStatus('Unable to load dataset list.'));
  }, []);

  useEffect(() => {
    if (recommendationResult && showRecommendation && recommendationResultRef.current) {
      recommendationResultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [recommendationResult, showRecommendation]);

  useEffect(() => {
    if (!selectedFile) {
      return;
    }
    setCurrentPage(1); // Reset to first page when file changes
    setSearchColumns([]);
    setSearchValues({});
    loadInitialData();
    // Load available rooms for PIR and lightning data
    if (selectedFile === 'PIRSensorData.csv' || selectedFile === 'lightningData.csv') {
      loadAvailableRooms();
    }
  }, [selectedFile]);

  useEffect(() => {
    if (!selectedFile || currentPage === 1) {
      return; // Skip on initial load or when no file selected
    }
    if (searchColumns.length > 0) {
      fetchFilteredPage(currentPage, searchColumns, searchValues);
    } else {
      loadInitialData();
    }
  }, [currentPage]);

  const loadInitialData = () => {
    setStatus(`Loading page ${currentPage} of ${selectedFile}...`);
    fetch(`/api/data?file=${encodeURIComponent(selectedFile)}&page=${currentPage}&page_size=${pageSize}`)
      .then((response) => response.json())
      .then((data) => {
        setColumns(data.columns || []);
        setRows(data.rows || []);
        setTotalCount(data.total_count || 0);
        setTotalPages(data.total_pages || 0);
        setHasNext(data.has_next || false);
        setHasPrev(data.has_prev || false);
        setStatus(`Loaded page ${currentPage} of ${selectedFile} (${data.count} rows, ${data.total_count} total).`);
      })
      .catch(() => setStatus('Unable to load dataset.'));
  };

  const fetchFilteredPage = (page, cols, vals) => {
    setIsSearching(true);
    const params = new URLSearchParams({
      file: selectedFile,
      page: page,
      page_size: pageSize
    });
    cols.forEach((column) => {
      params.append('search_column', column);
      params.append('search_value', formatValue(column, vals[column].trim()));
    });
    fetch(`/api/data?${params.toString()}`)
      .then((response) => response.json())
      .then((data) => {
        setRows(data.rows || []);
        setTotalCount(data.total_count || 0);
        setTotalPages(data.total_pages || 0);
        setHasNext(data.has_next || false);
        setHasPrev(data.has_prev || false);
        const summary = cols.map((c) => `${c}: ${vals[c].trim()}`).join(', ');
        setStatus(`Found ${data.count} results for "${summary}" (page ${page}).`);
        setIsSearching(false);
      })
      .catch(() => {
        setStatus('Search failed.');
        setIsSearching(false);
      });
  };

  const handleSearch = () => {
    if (!searchColumns.length) {
      setCurrentPage(1);
      loadInitialData();
      return;
    }

    const missingValue = searchColumns.some((column) => !(searchValues[column] || '').trim());
    if (missingValue) {
      setStatus('Enter a value for each selected search column.');
      return;
    }

    setCurrentPage(1);
    fetchFilteredPage(1, searchColumns, searchValues);
  };

  const generatePredefinedChart = (visualization) => {
    setShowChart(true);
    setSelectedVisualization(visualization);
    setIsGeneratingChart(true);
    setStatus(`Generating ${visualization.name}...`);

    fetch(`/api/visualize?file=${encodeURIComponent(selectedFile)}&x_column=${encodeURIComponent(visualization.x_column)}&y_column=${encodeURIComponent(visualization.y_column)}&chart_type=${visualization.chart_type}`)
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          setStatus(`Chart generation failed: ${data.error}`);
        } else {
          setChartImage(data.chart_url);
          setStatus(`Generated ${visualization.name}.`);
        }
        setIsGeneratingChart(false);
      })
      .catch(() => {
        setStatus('Chart generation failed.');
        setIsGeneratingChart(false);
      });
  };

  const loadAvailableRooms = () => {
    fetch('/api/rooms')
      .then((response) => response.json())
      .then((data) => {
        setAvailableRooms(data.rooms || []);
      })
      .catch(() => {
        setStatus('Failed to load available rooms.');
      });
  };

  const generateRoomSpecificChart = () => {
    if (!roomNumber || !startDate || !endDate) {
      setStatus('Please enter a room number and select a time range.');
      return;
    }

    const isLightning = selectedFile === 'lightningData.csv';
    const displayName = isLightning ? 'Lightning' : 'PIR Motion';
    const yColumn = isLightning ? 'Value' : 'pir_motion';
    const chartType = isLightning ? 'heatmap' : 'scatter';

    setShowChart(true);
    setSelectedVisualization({ id: 'room_specific', name: `Room ${roomNumber} - ${displayName} Trend` });
    setIsGeneratingChart(true);
    setStatus(`Generating ${displayName} visualization for Room ${roomNumber}...`);

    const params = new URLSearchParams({
      file: selectedFile,
      x_column: 'timestamp',
      y_column: yColumn,
      chart_type: chartType,
      room_number: roomNumber,
      start_timestamp: startDate ? startDate + ' 00:00:00' : '',
      end_timestamp: endDate ? endDate + ' 23:59:59' : ''
    });

    if (!isLightning) {
      params.append('pir_mode', pirMode);
    }

    if (isLightning && lampLocation) {
      params.append('lamp_location', lampLocation);
    }

    fetch(`/api/visualize?${params}`)
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          setStatus(`Chart generation failed: ${data.error}`);
        } else {
          setChartImage(data.chart_url);
          setStatus(`Generated ${displayName} visualization for Room ${roomNumber}.`);
        }
        setIsGeneratingChart(false);
      })
      .catch(() => {
        setStatus('Chart generation failed.');
        setIsGeneratingChart(false);
      });
  };

  const generateDailyLightningChart = () => {
    if (!dailyRoomNumber || !dailyDate) {
      setStatus('Please enter a room number and a date.');
      return;
    }

    setShowDailyChart(true);
    setIsGeneratingDailyChart(true);
    setStatus(`Generating daily lightning trend for Room ${dailyRoomNumber} on ${dailyDate}...`);

    const params = new URLSearchParams({
      file: 'lightningData.csv',
      x_column: 'timestamp',
      y_column: 'Value',
      chart_type: 'daily_trend',
      room_number: dailyRoomNumber,
      start_timestamp: dailyDate + ' 00:00:00',
      end_timestamp: dailyDate + ' 23:59:59',
    });

    if (dailyLampLocation) {
      params.append('lamp_location', dailyLampLocation);
    }

    fetch(`/api/visualize?${params}`)
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          setStatus(`Chart generation failed: ${data.error}`);
        } else {
          setDailyChartImage(data.chart_url);
          setStatus(`Generated daily lightning trend for Room ${dailyRoomNumber} on ${dailyDate}.`);
        }
        setIsGeneratingDailyChart(false);
      })
      .catch(() => {
        setStatus('Chart generation failed.');
        setIsGeneratingDailyChart(false);
      });
  };

  const generateDailyTemperatureChart = () => {
    if (!dailyTempRoomNumber || !dailyTempDate) {
      setStatus('Please enter a room number and a date.');
      return;
    }

    setShowDailyTempChart(true);
    setIsGeneratingDailyTempChart(true);
    setStatus(`Generating temperature trend for Room ${dailyTempRoomNumber} on ${dailyTempDate}...`);

    const params = new URLSearchParams({
      file: 'temperatureData.csv',
      x_column: 'timestamp',
      y_column: 'room_temp',
      chart_type: 'daily_temp_trend',
      room_number: dailyTempRoomNumber,
      start_timestamp: dailyTempDate + ' 00:00:00',
      end_timestamp: dailyTempDate + ' 23:59:59',
    });

    fetch(`/api/visualize?${params}`)
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          setStatus(`Chart generation failed: ${data.error}`);
        } else {
          setDailyTempChartImage(data.chart_url);
          setStatus(`Generated temperature trend for Room ${dailyTempRoomNumber} on ${dailyTempDate}.`);
        }
        setIsGeneratingDailyTempChart(false);
      })
      .catch(() => {
        setStatus('Chart generation failed.');
        setIsGeneratingDailyTempChart(false);
      });
  };

  const generateEnergyReport = () => {
    if (!energyRoomNumber || !energyStartDate || !energyEndDate) {
      setStatus('Please enter a room number and pick a start and end date.');
      return;
    }
    if (energyEndDate < energyStartDate) {
      setStatus('End date must be on or after the start date.');
      return;
    }

    setShowEnergy(true);
    setIsGeneratingEnergy(true);
    setStatus(`Calculating energy consumption for Room ${energyRoomNumber}...`);

    const params = new URLSearchParams({
      room_number: energyRoomNumber,
      start_timestamp: energyStartDate + ' 00:00:00',
      end_timestamp: energyEndDate + ' 23:59:59',
    });

    fetch(`/api/energy?${params}`)
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          setEnergyResult(null);
          setStatus(`Energy calculation failed: ${data.error}`);
        } else {
          setEnergyResult(data);
          setStatus(
            `Room ${energyRoomNumber} used ${data.summary.actual_wh.toFixed(1)} Wh ` +
            `(${data.summary.saved_pct.toFixed(1)}% saved by dimming).`
          );
        }
        setIsGeneratingEnergy(false);
      })
      .catch(() => {
        setStatus('Energy calculation failed.');
        setIsGeneratingEnergy(false);
      });
  };

  const requestOccupancyPrediction = async () => {
    const data = await fetch('/api/predict_occupancy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        room_number: Number(recommendationRoomNumber),
        timestamp: recommendationTimestamp.replace('T', ' ') + ':00',
        lookback_hours: 1,
        horizon_minutes: 60,
        occupancy_model_type: occupancyModelType,
      }),
    }).then(readJsonResponse);

    setOccupancyPredictionResult(data);
    setRecommendationOccupancy(data.prediction);
    return data;
  };

  const requestLightingPersonaPrediction = async () => {
    const data = await fetch('/api/predict_lighting_persona', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        room_number: Number(recommendationRoomNumber),
        timestamp: recommendationTimestamp.replace('T', ' ') + ':00',
        lookback_hours: Number(recommendationLookback || 24),
        lighting_persona_model_type: lightingPersonaModelType,
      }),
    }).then(readJsonResponse);

    setPersonaPredictionResult(data);
    setRecommendationPersona(data.prediction);
    return data;
  };

  const generateLightningRecommendation = async () => {
    if (!recommendationRoomNumber || !recommendationTimestamp) {
      setStatus('Please enter a room number and timestamp for the recommendation.');
      return;
    }

    setShowRecommendation(true);
    setIsGeneratingRecommendation(true);
    setIsPredictingOccupancy(true);
    setIsPredictingPersona(true);
    setStatus(`Predicting occupancy and persona, then generating recommendation for Room ${recommendationRoomNumber}...`);

    try {
      const [occupancyData, personaData] = await Promise.all([
        requestOccupancyPrediction(),
        requestLightingPersonaPrediction(),
      ]);
      setIsPredictingOccupancy(false);
      setIsPredictingPersona(false);

      const payload = {
        room_number: Number(recommendationRoomNumber),
        timestamp: recommendationTimestamp.replace('T', ' ') + ':00',
        occupancy_prediction: occupancyData.prediction,
        lighting_persona_prediction: personaData.prediction,
        occupancy_model_type: occupancyModelType,
        lighting_persona_model_type: lightingPersonaModelType,
        lighting_recommendation_model_type: lightingRecommendationModelType,
        use_model_predictions: true,
        lookback_hours: Number(recommendationLookback || 24),
        guest: {
          adults: Number(recommendationAdults || 0),
          children: Number(recommendationChildren || 0),
          nationality: recommendationNationality,
          room_type: recommendationRoomType,
        },
      };

      const data = await fetch('/api/lightning_recommendation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).then(readJsonResponse);

      setRecommendationResult(data);
      setStatus(
        `Recommended ${data.summary.recommended_active_lamps} active lamps for Room ${recommendationRoomNumber}, ` +
        `using AI occupancy (${occupancyData.prediction}) and persona (${personaData.prediction}) predictions.`
      );
    } catch (error) {
      setRecommendationResult(null);
      setStatus(`Recommendation failed: ${error.message}`);
    } finally {
      setIsPredictingOccupancy(false);
      setIsPredictingPersona(false);
      setIsGeneratingRecommendation(false);
    }
  };

  const predictOccupancy = async () => {
    if (!recommendationRoomNumber || !recommendationTimestamp) {
      setStatus('Please enter a room number and timestamp before predicting occupancy.');
      return;
    }

    setIsPredictingOccupancy(true);
    setStatus(`Predicting occupancy for Room ${recommendationRoomNumber}...`);

    try {
      const data = await requestOccupancyPrediction();
      setStatus(`Predicted occupancy: ${data.prediction} (${(data.confidence * 100).toFixed(1)}% confidence).`);
    } catch (error) {
      setOccupancyPredictionResult(null);
      setStatus(`Occupancy prediction failed: ${error.message}`);
    } finally {
      setIsPredictingOccupancy(false);
    }
  };

  const predictLightingPersona = async () => {
    if (!recommendationRoomNumber || !recommendationTimestamp) {
      setStatus('Please enter a room number and timestamp before predicting lighting persona.');
      return;
    }

    setIsPredictingPersona(true);
    setStatus(`Predicting lighting persona for Room ${recommendationRoomNumber}...`);

    try {
      const data = await requestLightingPersonaPrediction();
      setStatus(`Predicted lighting persona: ${data.prediction} (${(data.confidence * 100).toFixed(1)}% confidence).`);
    } catch (error) {
      setPersonaPredictionResult(null);
      setStatus(`Lighting persona prediction failed: ${error.message}`);
    } finally {
      setIsPredictingPersona(false);
    }
  };

  const predictBoth = async () => {
    if (!recommendationRoomNumber || !recommendationTimestamp) {
      setStatus('Please enter a room number and timestamp before predicting.');
      return;
    }

    setIsPredictingOccupancy(true);
    setIsPredictingPersona(true);
    setStatus(`Predicting occupancy and lighting persona for Room ${recommendationRoomNumber}...`);

    try {
      const [occupancyData, personaData] = await Promise.all([
        requestOccupancyPrediction(),
        requestLightingPersonaPrediction(),
      ]);
      setStatus(
        `Predicted occupancy: ${occupancyData.prediction}; ` +
        `lighting persona: ${personaData.prediction}.`
      );
    } catch (error) {
      setStatus(`Prediction failed: ${error.message}`);
    } finally {
      setIsPredictingOccupancy(false);
      setIsPredictingPersona(false);
    }
  };

  const predictTemperaturePersona = async () => {
    if (!tempPersonaRoomNumber || !tempPersonaTimestamp) {
      setStatus('Please enter a room number and timestamp before predicting temperature persona.');
      return;
    }

    setIsPredictingTempPersona(true);
    setStatus(`Predicting temperature persona for Room ${tempPersonaRoomNumber}...`);

    try {
      const data = await fetch('/api/predict_tempreture_persona', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          room_number: Number(tempPersonaRoomNumber),
          timestamp: tempPersonaTimestamp.replace('T', ' ') + ':00',
          lookback_hours: Number(tempPersonaLookback || 2),
        }),
      }).then(readJsonResponse);

      setTempPersonaResult(data);
      setStatus(`Predicted temperature persona: ${data.prediction} (${(data.confidence * 100).toFixed(1)}% confidence).`);
    } catch (error) {
      setTempPersonaResult(null);
      setStatus(`Temperature persona prediction failed: ${error.message}`);
    } finally {
      setIsPredictingTempPersona(false);
    }
  };

  const generateTemperatureRecommendation = async () => {
    if (!tempPersonaRoomNumber || !tempPersonaTimestamp) {
      setStatus('Please enter a room number and timestamp before recommending an HVAC setpoint.');
      return;
    }

    setIsGeneratingTempRecommendation(true);
    setStatus(`Building HVAC setpoint recommendation for Room ${tempPersonaRoomNumber}...`);

    try {
      const data = await fetch('/api/tempreture_recomendation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          room_number: Number(tempPersonaRoomNumber),
          timestamp: tempPersonaTimestamp.replace('T', ' ') + ':00',
          lookback_hours: Number(tempPersonaLookback || 2),
          temperature_recommendation_model_type: temperatureRecommendationModelType,
          use_model_predictions: true,
        }),
      }).then(readJsonResponse);

      setTempRecommendationResult(data);
      if (data.input?.model_predictions?.temperature_persona) {
        setTempPersonaResult(data.input.model_predictions.temperature_persona);
      }
      setStatus(
        `Recommended HVAC setpoint: ${data.recommended_setpoint.toFixed(1)}°C ` +
        `(${data.action} from ${data.current_setpoint.toFixed(1)}°C).`
      );
    } catch (error) {
      setTempRecommendationResult(null);
      setStatus(`Temperature recommendation failed: ${error.message}`);
    } finally {
      setIsGeneratingTempRecommendation(false);
    }
  };

  const generateHvacReport = () => {
    if (!hvacRoomNumber || !hvacStartDate || !hvacEndDate) {
      setStatus('Please enter a room number and pick a start and end date for HVAC.');
      return;
    }
    if (hvacEndDate < hvacStartDate) {
      setStatus('End date must be on or after the start date.');
      return;
    }

    setShowHvac(true);
    setIsGeneratingHvac(true);
    setStatus(`Calculating HVAC energy for Room ${hvacRoomNumber}...`);

    const params = new URLSearchParams({
      room_number: hvacRoomNumber,
      start_timestamp: hvacStartDate + ' 00:00:00',
      end_timestamp: hvacEndDate + ' 23:59:59',
    });

    fetch(`/api/hvac_energy?${params}`)
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          setHvacResult(null);
          setStatus(`HVAC energy calculation failed: ${data.error}`);
        } else {
          setHvacResult(data);
          setStatus(
            `Room ${hvacRoomNumber} HVAC used ${data.summary.total_wh.toFixed(1)} Wh ` +
            `(${data.summary.saved_pct.toFixed(1)}% saved vs always-on baseline).`
          );
        }
        setIsGeneratingHvac(false);
      })
      .catch(() => {
        setStatus('HVAC energy calculation failed.');
        setIsGeneratingHvac(false);
      });
  };

  const handleDownload = async (format) => {
    setIsDownloading(true);
    const params = new URLSearchParams({ file: selectedFile, format });
    if (searchColumns.length > 0 && searchColumns.some((c) => (searchValues[c] || '').trim())) {
      searchColumns.forEach((column) => {
        params.append('search_column', column);
        params.append('search_value', formatValue(column, searchValues[column].trim()));
      });
    }
    try {
      const response = await fetch(`/api/download?${params}`);
      if (!response.ok) {
        const err = await response.json();
        setStatus(`Download failed: ${err.error}`);
        setIsDownloading(false);
        return;
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const label = searchColumns.length > 0 ? 'filtered' : 'full';
      a.download = `${selectedFile.replace('.csv', '')}_${label}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setStatus(`Downloaded ${label} dataset as ${format.toUpperCase()}.`);
    } catch {
      setStatus('Download failed.');
    }
    setIsDownloading(false);
  };

  const handlePrint = () => window.print();

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="app-shell">
      <header className="header">
        <div>
          <h1>CSV Visualization</h1>
          <p>Search and browse CSV files efficiently.</p>
        </div>
        <div className="status">{status}</div>
      </header>

      <section className="controls">
        <label>
          Select dataset
          <select value={selectedFile} onChange={(event) => setSelectedFile(event.target.value)}>
            {files.map((file) => (
              <option key={file} value={file}>
                {file}
              </option>
            ))}
          </select>
        </label>

        <label>
          Search columns
          <select
            multiple
            value={searchColumns}
            onChange={(event) => {
              const selected = Array.from(event.target.selectedOptions, (option) => option.value);
              setSearchColumns(selected);
              setSearchValues((prev) => {
                const next = {};
                selected.forEach((col) => {
                  next[col] = prev[col] || '';
                });
                return next;
              });
            }}
            size={Math.min(columns.length, 6)}
          >
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <small className="hint-text">Hold Control/Cmd to select multiple columns.</small>
        </label>

        <label>
          Search values
          <div className="multi-search-inputs">
            {searchColumns.map((column) => (
              <div key={column} className="search-field">
                <span>{column}</span>
                <input
                  type={getInputType(column)}
                  value={searchValues[column] || ''}
                  onChange={(event) => setSearchValues((prev) => ({ ...prev, [column]: event.target.value }))}
                  onKeyPress={handleKeyPress}
                  placeholder={getInputType(column) === 'text' ? `Enter value for ${column}` : ''}
                />
              </div>
            ))}
          </div>
        </label>

        <button onClick={handleSearch} disabled={isSearching || !searchColumns.length || searchColumns.some((column) => !(searchValues[column] || '').trim())}>
          {isSearching ? 'Searching...' : 'Search'}
        </button>
      </section>

      {selectedFile && predefinedVisualizations[selectedFile] && (
        <section className="visualization">
          <h3>Data Visualizations</h3>
          {isGeneratingChart && (
            <div className="processing-banner">
              <div className="processing-icon" />
              <span>Processing visualization request… please wait.</span>
            </div>
          )}
          <div className="viz-grid">
            {predefinedVisualizations[selectedFile].map((viz) => (
              <div
                key={viz.id}
                className={`viz-card ${selectedVisualization?.id === viz.id ? 'active' : ''} ${isGeneratingChart ? 'disabled' : ''}`}
                onClick={() => !isGeneratingChart && generatePredefinedChart(viz)}
              >
                <h4>{viz.name}</h4>
                <p>{viz.description}</p>
                <div className="viz-meta">
                  <span className="chart-type">{viz.chart_type}</span>
                  <span className="axes">{viz.x_column} × {viz.y_column}</span>
                </div>
              </div>
            ))}
          </div>

          {chartImage && selectedVisualization && showChart && selectedVisualization.id !== 'room_specific' && (
            <div className="chart-container">
              <button
                className="chart-close-button"
                onClick={() => setShowChart(false)}
                aria-label="Close visualization"
              >
                ×
              </button>
              <h4>{selectedVisualization.name}</h4>
              <img src={chartImage} alt={selectedVisualization.name} />
              <div className="chart-info">
                <p><strong>Chart Type:</strong> {selectedVisualization.chart_type}</p>
                <p><strong>X-Axis:</strong> {selectedVisualization.x_column}</p>
                <p><strong>Y-Axis:</strong> {selectedVisualization.y_column}</p>
              </div>
            </div>
          )}

          {isGeneratingChart && (
            <div className="loading-chart">
              <p>Generating visualization...</p>
            </div>
          )}
        </section>
      )}

      {(selectedFile === 'PIRSensorData.csv' || selectedFile === 'lightningData.csv') && (
        <section className="room-specific">
          <h3>{selectedFile === 'lightningData.csv' ? 'Room-Specific Lightning Trend' : 'Room-Specific PIR Motion Activity Map'}</h3>
          <p className="room-help-text">
            Available rooms: {availableRooms.length > 0 ? availableRooms.slice(0, 10).join(', ') + (availableRooms.length > 10 ? '...' : '') : 'Loading...'}
          </p>
          <div className="room-controls">
            <label>
              Room Number
              <input
                type="number"
                value={roomNumber}
                onChange={(event) => setRoomNumber(event.target.value)}
                placeholder="e.g., 101, 1001"
                min="1"
              />
            </label>

            <label>
              Start Date
              <input
                type="date"
                value={startDate}
                onChange={(event) => setStartDate(event.target.value)}
              />
            </label>

            <label>
              End Date
              <input
                type="date"
                value={endDate}
                onChange={(event) => setEndDate(event.target.value)}
              />
            </label>

            {selectedFile === 'lightningData.csv' && (
              <label>
                Lamp Location
                <select value={lampLocation} onChange={(event) => setLampLocation(event.target.value)}>
                  <option value="">All locations</option>
                  {availableLampLocations.map((loc) => (
                    <option key={loc} value={loc}>{loc.replace(/_/g, ' ')}</option>
                  ))}
                </select>
              </label>
            )}

            <button
              onClick={generateRoomSpecificChart}
              disabled={isGeneratingChart || !roomNumber || !startDate || !endDate}
            >
              {isGeneratingChart ? 'Analyzing...' : selectedFile === 'lightningData.csv' ? 'Generate Lightning Trend' : 'Generate Activity Map'}
            </button>
          </div>

          {chartImage && selectedVisualization?.id === 'room_specific' && showChart && (
            <div className="chart-container">
              <button
                className="chart-close-button"
                onClick={() => setShowChart(false)}
                aria-label="Close visualization"
              >
                ×
              </button>
              <h4>
                Room {roomNumber} - {selectedFile === 'lightningData.csv' ? 'Lightning Trend' : 'Motion Activity Map'}
              </h4>
              <img src={chartImage} alt={`Room ${roomNumber} ${selectedFile === 'lightningData.csv' ? 'Lightning Trend' : 'Activity Map'}`} />
              <div className="chart-info">
                <p><strong>Room Number:</strong> {roomNumber}</p>
                <p><strong>Analysis Period:</strong> {startDate} to {endDate}</p>
                {selectedFile === 'lightningData.csv' ? (
                  <>
                    <p><strong>Measured Value:</strong> Lightning sensor value over time</p>
                    <p><strong>Visualization:</strong> X-axis = Time of Day, Y-axis = Date</p>
                    {lampLocation && <p><strong>Lamp Location:</strong> {lampLocation.replace(/_/g, ' ')}</p>}
                  </>
                ) : (
                  <>
                    <p><strong>Visualization:</strong> X-axis = Time of Day (00:00-23:59), Y-axis = Date (DD.MM.YYYY)</p>
                    <p><strong>Red Dots:</strong> Motion detected by PIR sensor at that time and date</p>
                  </>
                )}
              </div>
            </div>
          )}
        </section>
      )}

      {selectedFile === 'lightningData.csv' && (
        <section className="room-specific">
          <h3>Daily Lightning Trend</h3>
          <p className="room-help-text">
            See how lighting values change throughout a single day for a specific room and lamp.
          </p>
          <div className="room-controls">
            <label>
              Room Number
              <input
                type="number"
                value={dailyRoomNumber}
                onChange={(e) => setDailyRoomNumber(e.target.value)}
                placeholder="e.g., 101, 1001"
                min="1"
              />
            </label>

            <label>
              Date
              <input
                type="date"
                value={dailyDate}
                onChange={(e) => setDailyDate(e.target.value)}
              />
            </label>

            <label>
              Lamp Location
              <select value={dailyLampLocation} onChange={(e) => setDailyLampLocation(e.target.value)}>
                <option value="">All locations</option>
                {availableLampLocations.map((loc) => (
                  <option key={loc} value={loc}>{loc.replace(/_/g, ' ')}</option>
                ))}
              </select>
            </label>

            <button
              onClick={generateDailyLightningChart}
              disabled={isGeneratingDailyChart || !dailyRoomNumber || !dailyDate}
            >
              {isGeneratingDailyChart ? 'Generating...' : 'Generate Daily Trend'}
            </button>
          </div>

          {dailyChartImage && showDailyChart && (
            <div className="chart-container">
              <button
                className="chart-close-button"
                onClick={() => setShowDailyChart(false)}
                aria-label="Close visualization"
              >
                ×
              </button>
              <h4>Room {dailyRoomNumber} - Lightning Trend on {dailyDate}{dailyLampLocation ? ` | ${dailyLampLocation.replace(/_/g, ' ')}` : ''}</h4>
              <img src={dailyChartImage} alt={`Room ${dailyRoomNumber} daily lightning trend`} />
              <div className="chart-info">
                <p><strong>Room Number:</strong> {dailyRoomNumber}</p>
                <p><strong>Date:</strong> {dailyDate}</p>
                {dailyLampLocation && <p><strong>Lamp Location:</strong> {dailyLampLocation.replace(/_/g, ' ')}</p>}
                <p><strong>Visualization:</strong> X-axis = Time of Day, Y-axis = Lightning Value (0–100)</p>
              </div>
              <div className="chart-download-bar">
                <span className="download-label">Download chart:</span>
                <a
                  className="download-btn download-csv"
                  href={dailyChartImage}
                  download={`room${dailyRoomNumber}_lightning_${dailyDate}${dailyLampLocation ? '_' + dailyLampLocation : '_all_locations'}.png`}
                >
                  ⬇ PNG
                </a>
              </div>
            </div>
          )}
        </section>
      )}

      {selectedFile === 'lightningData.csv' && (
        <section className="room-specific">
          <h3>Lighting Recommendation</h3>
          <p className="room-help-text">
            Enter the current guest/context predictions, then generate per-lamp lighting levels from the room's recent lighting history. The comparison estimates the next 5-minute sample, using the same dimmer power table as the energy report.
          </p>
          <div className="recommendation-control-stack">
            <div className="room-controls recommendation-controls recommendation-input-box">
              <label>
                Room Number
                <input
                  type="number"
                  value={recommendationRoomNumber}
                  onChange={(e) => setRecommendationRoomNumber(e.target.value)}
                  placeholder="e.g., 1, 50, 100"
                  min="1"
                />
              </label>

              <label>
                Timestamp
                <input
                  type="datetime-local"
                  value={recommendationTimestamp}
                  onChange={(e) => setRecommendationTimestamp(e.target.value)}
                  min="2022-01-06T00:00"
                  max="2022-07-06T00:00"
                />
              </label>

              <label>
                Occupancy Prediction
                <select value={recommendationOccupancy} onChange={(e) => setRecommendationOccupancy(e.target.value)}>
                  <option value="Occupied">Occupied</option>
                  <option value="Vacant">Vacant</option>
                  <option value="Cleaning">Cleaning</option>
                </select>
              </label>

              <label>
                Lighting Persona
                <select value={recommendationPersona} onChange={(e) => setRecommendationPersona(e.target.value)}>
                  <option value="Balanced">Balanced</option>
                  <option value="Routine">Routine</option>
                  <option value="StaticBright">StaticBright</option>
                  <option value="StaticDim">StaticDim</option>
                  <option value="NightFocused">NightFocused</option>
                  <option value="Housekeeping">Housekeeping</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </label>

              <label>
                Adults
                <input
                  type="number"
                  value={recommendationAdults}
                  onChange={(e) => setRecommendationAdults(e.target.value)}
                  min="0"
                  max="6"
                />
              </label>

              <label>
                Children
                <input
                  type="number"
                  value={recommendationChildren}
                  onChange={(e) => setRecommendationChildren(e.target.value)}
                  min="0"
                  max="6"
                />
              </label>

              <label>
                Nationality
                <input
                  type="text"
                  value={recommendationNationality}
                  onChange={(e) => setRecommendationNationality(e.target.value)}
                  placeholder="Optional"
                />
              </label>

              <label>
                Room Type
                <select value={recommendationRoomType} onChange={(e) => setRecommendationRoomType(e.target.value)}>
                  <option value="Deluxe">Deluxe</option>
                  <option value="Standard">Standard</option>
                  <option value="Suite">Suite</option>
                </select>
              </label>

              <label>
                History Window
                <select value={recommendationLookback} onChange={(e) => setRecommendationLookback(e.target.value)}>
                  <option value="3">Last 3 hours</option>
                  <option value="6">Last 6 hours</option>
                  <option value="24">Last 24 hours</option>
                </select>
              </label>
            </div>

            <div className="room-controls recommendation-controls recommendation-model-box">
              <label>
                Occupancy AI Model
                <select value={occupancyModelType} onChange={(e) => setOccupancyModelType(e.target.value)}>
                  <option value="random_forest">Random Forest</option>
                  <option value="transformer">Transformer</option>
                  <option value="auto">Auto</option>
                </select>
              </label>

              <label>
                Persona AI Model
                <select value={lightingPersonaModelType} onChange={(e) => setLightingPersonaModelType(e.target.value)}>
                  <option value="random_forest">Random Forest</option>
                  <option value="transformer">Transformer</option>
                  <option value="auto">Auto</option>
                </select>
              </label>

              <label>
                Recommendation AI Model
                <select value={lightingRecommendationModelType} onChange={(e) => setLightingRecommendationModelType(e.target.value)}>
                  <option value="hist_gradient_boosting">HistGradientBoosting</option>
                  <option value="transformer">Transformer</option>
                  <option value="auto">Auto</option>
                </select>
              </label>
            </div>

            <div className="recommendation-button-row">
              <button
                onClick={generateLightningRecommendation}
                disabled={isGeneratingRecommendation || isPredictingOccupancy || isPredictingPersona || !recommendationRoomNumber || !recommendationTimestamp}
              >
                {isGeneratingRecommendation ? 'Generating...' : 'Get Recommendation'}
              </button>
            </div>
          </div>

          <div className="prediction-actions">
            <button
              onClick={predictOccupancy}
              disabled={isPredictingOccupancy || !recommendationRoomNumber || !recommendationTimestamp}
            >
              {isPredictingOccupancy ? 'Predicting...' : 'Predict Occupancy'}
            </button>
            <button
              onClick={predictLightingPersona}
              disabled={isPredictingPersona || !recommendationRoomNumber || !recommendationTimestamp}
            >
              {isPredictingPersona ? 'Predicting...' : 'Predict Persona'}
            </button>
            <button
              onClick={predictBoth}
              disabled={isPredictingOccupancy || isPredictingPersona || !recommendationRoomNumber || !recommendationTimestamp}
            >
              Predict Both
            </button>
          </div>

          {(occupancyPredictionResult || personaPredictionResult) && (
            <div className="prediction-grid">
              {occupancyPredictionResult && (
                <div className="prediction-card">
                  <div className="prediction-card-header">
                    <span>Occupancy</span>
                    <strong>{occupancyPredictionResult.prediction}</strong>
                  </div>
                  <p>{(occupancyPredictionResult.confidence * 100).toFixed(1)}% confidence</p>
                  {occupancyPredictionResult.features.model && (
                    <p className="prediction-model">{occupancyPredictionResult.features.model}</p>
                  )}
                  <div className="prediction-probabilities">
                    {Object.entries(occupancyPredictionResult.probabilities).map(([label, value]) => (
                      <div className="prediction-probability" key={label}>
                        <span>{label}</span>
                        <div className="bar-track">
                          <div className="bar-fill recommended" style={{ width: `${Math.max(2, value * 100)}%` }} />
                        </div>
                        <strong>{(value * 100).toFixed(1)}%</strong>
                      </div>
                    ))}
                  </div>
                  <div className="prediction-features">
                    <span>Motion rate: {(occupancyPredictionResult.features.motion_rate * 100).toFixed(1)}%</span>
                    <span>Latest state: {occupancyPredictionResult.features.latest_state}</span>
                  </div>
                </div>
              )}

              {personaPredictionResult && (
                <div className="prediction-card">
                  <div className="prediction-card-header">
                    <span>Lighting Persona</span>
                    <strong>{personaPredictionResult.prediction}</strong>
                  </div>
                  <p>{(personaPredictionResult.confidence * 100).toFixed(1)}% confidence</p>
                  {personaPredictionResult.features.model && (
                    <p className="prediction-model">{personaPredictionResult.features.model}</p>
                  )}
                  <div className="prediction-probabilities">
                    {Object.entries(personaPredictionResult.probabilities).map(([label, value]) => (
                      <div className="prediction-probability" key={label}>
                        <span>{label}</span>
                        <div className="bar-track">
                          <div className="bar-fill recommended" style={{ width: `${Math.max(2, value * 100)}%` }} />
                        </div>
                        <strong>{(value * 100).toFixed(1)}%</strong>
                      </div>
                    ))}
                  </div>
                  <div className="prediction-features">
                    <span>Mean on level: {personaPredictionResult.features.mean_on_level.toFixed(1)}</span>
                    <span>Lit ratio: {(personaPredictionResult.features.lit_ratio * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {isGeneratingRecommendation && (
            <div className="processing-banner">
              <div className="processing-icon" />
              <span>Building recommendation from recent lighting history…</span>
            </div>
          )}

          {recommendationResult && showRecommendation && (
            <div
              id="lighting-recommendation-result"
              ref={recommendationResultRef}
              className="chart-container recommendation-panel recommendation-panel-ready"
            >
              <button
                className="chart-close-button"
                onClick={() => setShowRecommendation(false)}
                aria-label="Close recommendation"
              >
                ×
              </button>
              <h4>Room {recommendationResult.input.room_number} — Lighting Recommendation</h4>

              <div className="energy-summary">
                <div className="energy-stat">
                  <span className="stat-label">Current dimmable energy</span>
                  <span className="stat-value">{recommendationResult.summary.actual_wh.toFixed(3)} Wh</span>
                  <span className="stat-sub">Dimmable lamps only</span>
                </div>
                <div className="energy-stat">
                  <span className="stat-label">Recommended dimmable energy</span>
                  <span className="stat-value">{recommendationResult.summary.recommended_wh.toFixed(3)} Wh</span>
                  <span className="stat-sub">
                    {recommendationResult.summary.recommended_active_dimmable_lamps} active dimmable lamps
                  </span>
                </div>
                <div className="energy-stat energy-stat-savings">
                  <span className="stat-label">Saving vs current</span>
                  <span className="stat-value">{recommendationResult.summary.saved_pct.toFixed(1)}%</span>
                  <span className="stat-sub">{recommendationResult.summary.saved_wh.toFixed(3)} Wh, dimmable only</span>
                </div>
                <div className="energy-stat energy-stat-savings">
                  <span className="stat-label">Total saving vs full brightness</span>
                  <span className="stat-value">{recommendationResult.summary.max_baseline_saved_pct.toFixed(1)}%</span>
                  <span className="stat-sub">
                    {recommendationResult.summary.max_baseline_saved_wh.toFixed(3)} Wh of {recommendationResult.summary.max_dimmable_wh.toFixed(3)} Wh max
                  </span>
                </div>
              </div>

              <div className="recommendation-bars">
                {recommendationResult.recommendations.map((row) => {
                  const currentWidth = Math.max(2, row.current_level);
                  const recommendedWidth = Math.max(2, row.recommended_level);
                  return (
                    <div className="recommendation-row" key={row.lamp}>
                      <div className="recommendation-lamp">
                        <strong>{row.lamp.replace(/_/g, ' ')}</strong>
                        <span>{row.lamp_type}</span>
                      </div>
                      <div className="recommendation-bar-stack">
                        <div className="recommendation-bar-line">
                          <span className="bar-label">Current {row.current_level.toFixed(0)}</span>
                          <div className="bar-track">
                            <div className="bar-fill current" style={{ width: `${currentWidth}%` }} />
                          </div>
                        </div>
                        <div className="recommendation-bar-line">
                          <span className="bar-label">Recommended {row.recommended_level.toFixed(0)}</span>
                          <div className="bar-track">
                            <div className="bar-fill recommended" style={{ width: `${recommendedWidth}%` }} />
                          </div>
                        </div>
                      </div>
                      <div className="recommendation-saving">
                        <strong>{row.saved_wh.toFixed(3)} Wh</strong>
                        <span>
                          {!row.saving_counted ? 'Not included in AI savings because this lamp is not dimmable. ' : ''}
                          {row.reason}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="chart-info">
                <p><strong>Timestamp:</strong> {recommendationResult.input.timestamp}</p>
                <p><strong>Occupancy:</strong> {recommendationResult.input.occupancy_prediction}</p>
                <p><strong>Persona:</strong> {recommendationResult.input.lighting_persona_prediction}</p>
                <p><strong>Recommendation model:</strong> {recommendationResult.input.recommendation_model}</p>
                <p><strong>History:</strong> Last {recommendationResult.input.lookback_hours} hours</p>
              </div>
            </div>
          )}

          <h3>Energy Consumption</h3>
          <p className="room-help-text">
            Compute the actual electrical energy used by every lamp in a room over the chosen interval, plus the energy that <em>would</em> have been used if no dimming were applied (every lamp running at maximum). Power per dimmer level is taken from <code>EnergyConsumption/DimmerGraphs.xlsx</code>: dimmable LEDs (bed_left, bed_right, dinner_table, table, hidden_top) are looked up by their recorded dimmer level; non-dimmable bulbs (closet, corridor_left, corridor_right, shower, cabinet, sink) always run at full bulb power when on.
          </p>
          <div className="room-controls">
            <label>
              Room Number
              <input
                type="number"
                value={energyRoomNumber}
                onChange={(e) => setEnergyRoomNumber(e.target.value)}
                placeholder="e.g., 1, 50, 100"
                min="1"
                max="100"
              />
            </label>

            <label>
              Start Date
              <input
                type="date"
                value={energyStartDate}
                onChange={(e) => setEnergyStartDate(e.target.value)}
                min="2022-01-01"
                max="2022-02-01"
              />
            </label>

            <label>
              End Date
              <input
                type="date"
                value={energyEndDate}
                onChange={(e) => setEnergyEndDate(e.target.value)}
                min="2022-01-01"
                max="2022-02-01"
              />
            </label>

            <button
              onClick={generateEnergyReport}
              disabled={isGeneratingEnergy || !energyRoomNumber || !energyStartDate || !energyEndDate}
            >
              {isGeneratingEnergy ? 'Calculating...' : 'Calculate Energy'}
            </button>
          </div>

          {isGeneratingEnergy && (
            <div className="processing-banner">
              <div className="processing-icon" />
              <span>Calculating energy consumption… this can take a few seconds.</span>
            </div>
          )}

          {energyResult && showEnergy && (
            <div className="chart-container">
              <button
                className="chart-close-button"
                onClick={() => setShowEnergy(false)}
                aria-label="Close energy report"
              >
                ×
              </button>
              <h4>Room {energyRoomNumber} — Energy Report ({energyStartDate} to {energyEndDate})</h4>

              <div className="energy-summary">
                <div className="energy-stat">
                  <span className="stat-label">Actual energy used</span>
                  <span className="stat-value">{energyResult.summary.actual_wh.toFixed(2)} Wh</span>
                  <span className="stat-sub">({(energyResult.summary.actual_wh / 1000).toFixed(3)} kWh)</span>
                </div>
                <div className="energy-stat">
                  <span className="stat-label">If all lamps at max level</span>
                  <span className="stat-value">{energyResult.summary.max_wh.toFixed(2)} Wh</span>
                  <span className="stat-sub">({(energyResult.summary.max_wh / 1000).toFixed(3)} kWh)</span>
                </div>
                <div className="energy-stat energy-stat-savings">
                  <span className="stat-label">Saved by dimming</span>
                  <span className="stat-value">{energyResult.summary.saved_wh.toFixed(2)} Wh</span>
                  <span className="stat-sub">({energyResult.summary.saved_pct.toFixed(1)}%)</span>
                </div>
                {energyResult.dimmable_summary && (
                  <div className="energy-stat energy-stat-savings">
                    <span className="stat-label">Dimmable lamps — total saved</span>
                    <span className="stat-value">{energyResult.dimmable_summary.saved_wh.toFixed(2)} Wh</span>
                    <span className="stat-sub">
                      ({energyResult.dimmable_summary.saved_pct.toFixed(1)}% of {energyResult.dimmable_summary.max_wh.toFixed(1)} Wh max)
                    </span>
                  </div>
                )}
                {energyResult.ai_recommendation_energy && !energyResult.ai_recommendation_energy.empty && (
                  <>
                    <div className="energy-stat">
                      <span className="stat-label">AI lighting energy</span>
                      <span className="stat-value">{energyResult.ai_recommendation_energy.summary.recommended_wh.toFixed(2)} Wh</span>
                      <span className="stat-sub">
                        Full brightness baseline: {energyResult.ai_recommendation_energy.summary.full_brightness_wh.toFixed(2)} Wh
                      </span>
                    </div>
                    <div className={`energy-stat ${energyResult.ai_recommendation_energy.summary.energy_change_wh < 0 ? 'energy-stat-negative' : 'energy-stat-savings'}`}>
                      <span className="stat-label">AI recommendation energy change</span>
                      <span className="stat-value">{energyResult.ai_recommendation_energy.summary.energy_change_wh.toFixed(2)} Wh</span>
                      <span className="stat-sub">
                        {energyResult.ai_recommendation_energy.summary.energy_change_pct.toFixed(1)}% across {energyResult.ai_recommendation_energy.summary.n_timestamps} samples
                      </span>
                    </div>
                  </>
                )}
              </div>

              {energyResult.charts.total && (
                <img src={energyResult.charts.total} alt="Total energy actual vs max" />
              )}
              {energyResult.charts.per_lamp && (
                <img src={energyResult.charts.per_lamp} alt="Per-lamp energy comparison" />
              )}
              {energyResult.charts.dimmable_savings && (
                <img src={energyResult.charts.dimmable_savings} alt="Dimmable lamps energy savings" />
              )}
              {energyResult.charts.timeseries && (
                <img src={energyResult.charts.timeseries} alt="Energy over time" />
              )}
              {energyResult.charts.lamp_type && (
                <img src={energyResult.charts.lamp_type} alt="LED vs bulb contribution" />
              )}

              <div className="chart-info">
                <p><strong>Per-lamp breakdown:</strong></p>
                <table className="energy-table">
                  <thead>
                    <tr>
                      <th>Lamp</th>
                      <th>Actual (Wh)</th>
                      <th>Max-level (Wh)</th>
                      <th>Saved %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {energyResult.by_lamp.map((row) => (
                      <tr key={row.lamp}>
                        <td>{row.lamp.replace(/_/g, ' ')}</td>
                        <td>{row.actual_wh.toFixed(2)}</td>
                        <td>{row.max_wh.toFixed(2)}</td>
                        <td>{row.saved_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="hint-text">
                  Dimmable LEDs: {energyResult.classification.dimmable.join(', ')}.<br />
                  Non-dimmable bulbs: {energyResult.classification.non_dimmable.join(', ')}.
                  {energyResult.ai_recommendation_energy && !energyResult.ai_recommendation_energy.empty && (
                    <> AI recommendation energy compares full-brightness dimmable-lamp usage against Transformer/AI-recommended dimmable levels for each sample in the selected interval.</>
                  )}
                </p>
              </div>
            </div>
          )}
        </section>
      )}

      {selectedFile === 'temperatureData.csv' && (
        <section className="room-specific">
          <h3>Temperature Persona Prediction</h3>
          <p className="room-help-text">
            Predict the guest's HVAC comfort persona from recent room temperature, setpoint, HVAC mode, occupancy state, and weather context.
          </p>
          <div className="room-controls">
            <label>
              Room Number
              <input
                type="number"
                value={tempPersonaRoomNumber}
                onChange={(e) => setTempPersonaRoomNumber(e.target.value)}
                placeholder="e.g., 1, 50, 100"
                min="1"
                max="100"
              />
            </label>

            <label>
              Timestamp
              <input
                type="datetime-local"
                value={tempPersonaTimestamp}
                onChange={(e) => setTempPersonaTimestamp(e.target.value)}
                min="2022-01-01T00:00"
                max="2022-02-01T23:55"
              />
            </label>

            <label>
              History Window
              <select value={tempPersonaLookback} onChange={(e) => setTempPersonaLookback(e.target.value)}>
                <option value="2">Last 2 hours</option>
                <option value="4">Last 4 hours</option>
                <option value="8">Last 8 hours</option>
                <option value="24">Last 24 hours</option>
              </select>
            </label>

            <label>
              Recommendation Model
              <select
                value={temperatureRecommendationModelType}
                onChange={(e) => setTemperatureRecommendationModelType(e.target.value)}
              >
                <option value="transformer">Transformer Regressor</option>
                <option value="hist_gradient_boosting">Hist Gradient Boosting Regressor</option>
              </select>
            </label>

            <button
              onClick={predictTemperaturePersona}
              disabled={isPredictingTempPersona || isGeneratingTempRecommendation || !tempPersonaRoomNumber || !tempPersonaTimestamp}
            >
              {isPredictingTempPersona ? 'Predicting...' : 'Predict Temperature Persona'}
            </button>

            <button
              onClick={generateTemperatureRecommendation}
              disabled={isGeneratingTempRecommendation || isPredictingTempPersona || !tempPersonaRoomNumber || !tempPersonaTimestamp}
            >
              {isGeneratingTempRecommendation ? 'Recommending...' : 'Recommend HVAC Setpoint'}
            </button>
          </div>

          {(tempPersonaResult || tempRecommendationResult) && (
            <div className="prediction-grid">
              {tempPersonaResult && (
                <div className="prediction-card">
                  <div className="prediction-card-header">
                    <span>Temperature Persona</span>
                    <strong>{tempPersonaResult.prediction}</strong>
                  </div>
                  <p>{(tempPersonaResult.confidence * 100).toFixed(1)}% confidence</p>
                  {tempPersonaResult.features.model && (
                    <p className="prediction-model">{tempPersonaResult.features.model}</p>
                  )}
                  <div className="prediction-probabilities">
                    {Object.entries(tempPersonaResult.probabilities).map(([label, value]) => (
                      <div className="prediction-probability" key={label}>
                        <span>{label}</span>
                        <div className="bar-track">
                          <div className="bar-fill recommended" style={{ width: `${Math.max(2, value * 100)}%` }} />
                        </div>
                        <strong>{(value * 100).toFixed(1)}%</strong>
                      </div>
                    ))}
                  </div>
                  <div className="prediction-features">
                    <span>Room temp: {tempPersonaResult.features.room_temp.toFixed(1)}°C</span>
                    <span>Setpoint: {tempPersonaResult.features.setpoint.toFixed(1)}°C</span>
                    <span>HVAC: {tempPersonaResult.features.hvac_mode}</span>
                    <span>Room state: {tempPersonaResult.features.room_state}</span>
                  </div>
                </div>
              )}

              {tempRecommendationResult && (
                <div className="prediction-card">
                  <div className="prediction-card-header">
                    <span>HVAC Setpoint</span>
                    <strong>{tempRecommendationResult.recommended_setpoint.toFixed(1)}°C</strong>
                  </div>
                  <p className="prediction-model">{tempRecommendationResult.model}</p>
                  <div className="energy-summary compact-summary">
                    <div className="energy-stat">
                      <span className="stat-label">Current setpoint</span>
                      <span className="stat-value">{tempRecommendationResult.current_setpoint.toFixed(1)}°C</span>
                    </div>
                    <div className="energy-stat energy-stat-savings">
                      <span className="stat-label">Recommended</span>
                      <span className="stat-value">{tempRecommendationResult.recommended_setpoint.toFixed(1)}°C</span>
                    </div>
                    <div className="energy-stat">
                      <span className="stat-label">Target mode</span>
                      <span className="stat-value">{tempRecommendationResult.target_mode}</span>
                    </div>
                    <div className="energy-stat">
                      <span className="stat-label">Current energy</span>
                      <span className="stat-value">{tempRecommendationResult.energy.current_wh.toFixed(2)} Wh</span>
                      <span className="stat-sub">{tempRecommendationResult.energy.current_power_w.toFixed(0)} W estimate</span>
                    </div>
                    <div className="energy-stat">
                      <span className="stat-label">AI energy</span>
                      <span className="stat-value">{tempRecommendationResult.energy.recommended_wh.toFixed(2)} Wh</span>
                      <span className="stat-sub">{tempRecommendationResult.energy.recommended_power_w.toFixed(0)} W estimate</span>
                    </div>
                    <div className={`energy-stat ${tempRecommendationResult.energy.saved_wh < 0 ? 'energy-stat-negative' : 'energy-stat-savings'}`}>
                      <span className="stat-label">AI energy change</span>
                      <span className="stat-value">{tempRecommendationResult.energy.saved_pct.toFixed(1)}%</span>
                      <span className="stat-sub">{tempRecommendationResult.energy.saved_wh.toFixed(2)} Wh for next {tempRecommendationResult.energy.sample_minutes} min</span>
                    </div>
                  </div>
                  <div className="prediction-features">
                    <span>Occupancy: {tempRecommendationResult.input.occupancy_prediction}</span>
                    <span>Persona: {tempRecommendationResult.input.temperature_persona_prediction}</span>
                    <span>Room temp: {tempRecommendationResult.features.room_temp.toFixed(1)}°C</span>
                    <span>Outside: {tempRecommendationResult.features.outside_temp.toFixed(1)}°C</span>
                  </div>
                  <p>{tempRecommendationResult.reason}</p>
                </div>
              )}
            </div>
          )}

          <h3>Daily Temperature Trend</h3>
          <p className="room-help-text">
            Pick a room and a date to see how its temperature changes through the day. Setpoint and outside temperature are overlaid for context, and heating/cooling periods are shaded. Data covers 2022-01-01 to 2022-02-01.
          </p>
          <div className="room-controls">
            <label>
              Room Number
              <input
                type="number"
                value={dailyTempRoomNumber}
                onChange={(e) => setDailyTempRoomNumber(e.target.value)}
                placeholder="e.g., 1, 50, 100"
                min="1"
                max="100"
              />
            </label>

            <label>
              Date
              <input
                type="date"
                value={dailyTempDate}
                onChange={(e) => setDailyTempDate(e.target.value)}
                min="2022-01-01"
                max="2022-02-01"
              />
            </label>

            <button
              onClick={generateDailyTemperatureChart}
              disabled={isGeneratingDailyTempChart || !dailyTempRoomNumber || !dailyTempDate}
            >
              {isGeneratingDailyTempChart ? 'Generating...' : 'Generate Temperature Trend'}
            </button>
          </div>

          {dailyTempChartImage && showDailyTempChart && (
            <div className="chart-container">
              <button
                className="chart-close-button"
                onClick={() => setShowDailyTempChart(false)}
                aria-label="Close visualization"
              >
                ×
              </button>
              <h4>Room {dailyTempRoomNumber} — Temperature Trend on {dailyTempDate}</h4>
              <img src={dailyTempChartImage} alt={`Room ${dailyTempRoomNumber} temperature trend on ${dailyTempDate}`} />
              <div className="chart-info">
                <p><strong>Room Number:</strong> {dailyTempRoomNumber}</p>
                <p><strong>Date:</strong> {dailyTempDate}</p>
                <p><strong>Visualization:</strong> X-axis = Time of Day (00:00–24:00), Y-axis = Temperature (°C)</p>
                <p><strong>Lines:</strong> Room temp (solid blue), Setpoint (dashed green), Outside temp (dotted orange)</p>
                <p><strong>Shaded bands:</strong> Pink = heating active, Light blue = cooling active</p>
              </div>
              <div className="chart-download-bar">
                <span className="download-label">Download chart:</span>
                <a
                  className="download-btn download-csv"
                  href={dailyTempChartImage}
                  download={`room${dailyTempRoomNumber}_temperature_${dailyTempDate}.png`}
                >
                  ⬇ PNG
                </a>
              </div>
            </div>
          )}

          <h3>HVAC Energy Consumption</h3>
          <p className="room-help-text">
            Estimate the electrical energy consumed by a room's HVAC unit over the chosen interval. The temperature dataset records the HVAC mode (<code>heating</code>, <code>cooling</code>, <code>off</code>) every 5 minutes. Because there are no direct power readings, each active sample estimates power from room temperature, setpoint, outside temperature, room size, room state, and PIR motion, then contributes <code>estimated_power × (5/60) Wh</code>. The <strong>max baseline</strong> is the energy the unit would have used if it ran continuously at its dominant active mode's rated power cap for the entire interval; savings = max − actual.
          </p>
          <div className="room-controls">
            <label>
              Room Number
              <input
                type="number"
                value={hvacRoomNumber}
                onChange={(e) => setHvacRoomNumber(e.target.value)}
                placeholder="e.g., 1, 50, 100"
                min="1"
                max="100"
              />
            </label>

            <label>
              Start Date
              <input
                type="date"
                value={hvacStartDate}
                onChange={(e) => setHvacStartDate(e.target.value)}
                min="2022-01-01"
                max="2022-02-01"
              />
            </label>

            <label>
              End Date
              <input
                type="date"
                value={hvacEndDate}
                onChange={(e) => setHvacEndDate(e.target.value)}
                min="2022-01-01"
                max="2022-02-01"
              />
            </label>

            <button
              onClick={generateHvacReport}
              disabled={isGeneratingHvac || !hvacRoomNumber || !hvacStartDate || !hvacEndDate}
            >
              {isGeneratingHvac ? 'Calculating...' : 'Calculate HVAC Energy'}
            </button>
          </div>

          {isGeneratingHvac && (
            <div className="loading-message">
              <span className="loading-spinner"></span>
              <span>Calculating HVAC energy… this can take a few seconds.</span>
            </div>
          )}

          {hvacResult && showHvac && (
            <div className="chart-container">
              <button
                className="chart-close-button"
                onClick={() => setShowHvac(false)}
                aria-label="Close HVAC report"
              >
                ×
              </button>
              <h4>Room {hvacRoomNumber} — HVAC Energy Report ({hvacStartDate} to {hvacEndDate})</h4>

              <div className="energy-summary">
                <div className="energy-stat">
                  <span className="stat-label">Actual HVAC energy</span>
                  <span className="stat-value">{hvacResult.summary.total_wh.toFixed(2)} Wh</span>
                  <span className="stat-sub">({(hvacResult.summary.total_wh / 1000).toFixed(3)} kWh)</span>
                </div>
                <div className="energy-stat">
                  <span className="stat-label">Max ({hvacResult.summary.dominant_mode} continuous)</span>
                  <span className="stat-value">{hvacResult.summary.max_wh.toFixed(2)} Wh</span>
                  <span className="stat-sub">({(hvacResult.summary.max_wh / 1000).toFixed(3)} kWh)</span>
                </div>
                <div className="energy-stat energy-stat-savings">
                  <span className="stat-label">Saved by smart control</span>
                  <span className="stat-value">{hvacResult.summary.saved_wh.toFixed(2)} Wh</span>
                  <span className="stat-sub">({hvacResult.summary.saved_pct.toFixed(1)}%)</span>
                </div>
                <div className="energy-stat">
                  <span className="stat-label">Heating energy</span>
                  <span className="stat-value">{hvacResult.summary.heating_wh.toFixed(2)} Wh</span>
                  <span className="stat-sub">({(hvacResult.summary.heating_wh / 1000).toFixed(3)} kWh)</span>
                </div>
                <div className="energy-stat">
                  <span className="stat-label">Cooling energy</span>
                  <span className="stat-value">{hvacResult.summary.cooling_wh.toFixed(2)} Wh</span>
                  <span className="stat-sub">({(hvacResult.summary.cooling_wh / 1000).toFixed(3)} kWh)</span>
                </div>
                <div className="energy-stat">
                  <span className="stat-label">Active time</span>
                  <span className="stat-value">{hvacResult.summary.active_minutes} min</span>
                  <span className="stat-sub">
                    of {hvacResult.summary.total_minutes} min ({hvacResult.summary.total_minutes ? (100 * hvacResult.summary.active_minutes / hvacResult.summary.total_minutes).toFixed(1) : '0.0'}%)
                  </span>
                </div>
                {hvacResult.ai_recommendation_energy && !hvacResult.ai_recommendation_energy.empty && (
                  <>
                    <div className="energy-stat">
                      <span className="stat-label">AI setpoint energy</span>
                      <span className="stat-value">{hvacResult.ai_recommendation_energy.summary.recommended_wh.toFixed(2)} Wh</span>
                      <span className="stat-sub">
                        Current comparable: {hvacResult.ai_recommendation_energy.summary.current_wh.toFixed(2)} Wh
                      </span>
                    </div>
                    <div className={`energy-stat ${hvacResult.ai_recommendation_energy.summary.saved_wh < 0 ? 'energy-stat-negative' : 'energy-stat-savings'}`}>
                      <span className="stat-label">
                        {hvacResult.ai_recommendation_energy.summary.saved_wh < 0
                          ? 'AI recommendation additional energy'
                          : 'AI recommendation energy saving'}
                      </span>
                      <span className="stat-value">{Math.abs(hvacResult.ai_recommendation_energy.summary.saved_pct).toFixed(1)}%</span>
                      <span className="stat-sub">
                        {Math.abs(hvacResult.ai_recommendation_energy.summary.saved_wh).toFixed(2)} Wh across {hvacResult.ai_recommendation_energy.summary.n_rows} samples
                      </span>
                    </div>
                  </>
                )}
              </div>

              {hvacResult.charts.actual_vs_max && (
                <img src={hvacResult.charts.actual_vs_max} alt="HVAC actual vs max-baseline energy" />
              )}
              {hvacResult.charts.by_mode && (
                <img src={hvacResult.charts.by_mode} alt="HVAC energy by mode" />
              )}
              {hvacResult.charts.time_share && (
                <img src={hvacResult.charts.time_share} alt="HVAC time share by mode" />
              )}
              {hvacResult.charts.timeseries && (
                <img src={hvacResult.charts.timeseries} alt="HVAC energy over time" />
              )}

              <div className="chart-info">
                <p><strong>Per-mode breakdown:</strong></p>
                <table className="energy-table">
                  <thead>
                    <tr>
                      <th>Mode</th>
                      <th>Rated Power (W)</th>
                      <th>Avg Estimated Power (W)</th>
                      <th>Active Minutes</th>
                      <th>Energy (Wh)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {hvacResult.by_mode.map((row) => (
                      <tr key={row.mode}>
                        <td>{row.mode}</td>
                        <td>{row.rated_w.toFixed(0)}</td>
                        <td>{row.avg_power_w !== undefined ? row.avg_power_w.toFixed(0) : '-'}</td>
                        <td>{row.minutes}</td>
                        <td>{row.energy_wh.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="hint-text">
                  Each row in the temperature dataset is a {hvacResult.sample_minutes}-minute sample.
                  Energy per sample = estimated_power × ({hvacResult.sample_minutes}/60) Wh.
                  Estimated power uses HVAC mode, room temperature, setpoint, outside temperature, room size, room state, and PIR motion.
                  Rated power caps: heating {hvacResult.rated_power_w.heating} W, cooling {hvacResult.rated_power_w.cooling} W.
                  {hvacResult.ai_recommendation_energy && !hvacResult.ai_recommendation_energy.empty && (
                    <> AI recommendation energy compares the current setpoint against the Transformer-recommended setpoint for each sample in the selected interval.</>
                  )}
                </p>
              </div>
            </div>
          )}
        </section>
      )}

      <section className="summary">
        <span>
          Showing {rows.length} of {totalCount} total rows
          {searchColumns.length && Object.values(searchValues).some((value) => value?.trim()) ? ` matching "${Object.entries(searchValues).filter(([col, val]) => val?.trim()).map(([col, val]) => `${col}: ${val.trim()}`).join(', ')}"` : ''}
          {totalPages > 1 && ` (Page ${currentPage} of ${totalPages})`}
        </span>
        {rows.length > 0 && (
          <div className="download-bar">
            <span className="download-label">
              {searchColumns.length > 0 && searchColumns.some((c) => (searchValues[c] || '').trim())
                ? `Export all ${totalCount.toLocaleString()} filtered rows:`
                : `Export all ${totalCount.toLocaleString()} rows:`}
            </span>
            <button className="download-btn download-csv" onClick={() => handleDownload('csv')} disabled={isDownloading}>
              {isDownloading ? 'Preparing…' : '⬇ CSV'}
            </button>
            <button className="download-btn download-json" onClick={() => handleDownload('json')} disabled={isDownloading}>
              {isDownloading ? 'Preparing…' : '⬇ JSON'}
            </button>
            <button className="download-btn download-print" onClick={handlePrint}>
              🖨 Print / PDF
            </button>
          </div>
        )}
      </section>

      {totalPages > 1 && (
        <section className="pagination">
          <button
            onClick={() => setCurrentPage(currentPage - 1)}
            disabled={!hasPrev || isSearching}
          >
            ← Previous
          </button>
          <span className="page-info">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(currentPage + 1)}
            disabled={!hasNext || isSearching}
          >
            Next →
          </button>
        </section>
      )}

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index}>
                {columns.map((column) => (
                  <td key={`${index}-${column}`}>{row[column] ?? ''}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <footer className="footer">
        Select a column and enter a search term to find specific data. Press Enter or click Search.
      </footer>
    </div>
  );
}

export default App;
