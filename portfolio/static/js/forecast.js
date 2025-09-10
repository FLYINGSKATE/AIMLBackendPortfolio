// Global variables
let chart;
let currentModel = null;
const API_BASE_URL = '/api';

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any event listeners or setup here
    console.log('Forecast visualization page loaded');
    
    // Set default values
    document.getElementById('modelStatus').textContent = 'No model trained';
    document.getElementById('modelTypeDisplay').textContent = '-';
    document.getElementById('trainingPoints').textContent = '0';
    document.getElementById('lastTrained').textContent = '-';
});

// Load sample data from the API
async function loadSampleData() {
    const options = document.getElementById('sampleOptions');
    if (options.style.display === 'none') {
        options.style.display = 'block';
        // Initial load of sample data
        await updateSampleData();
    } else {
        options.style.display = 'none';
    }
}

// Load a predefined dataset with specific characteristics
async function loadPredefinedData(datasetType) {
    const settings = {
        'monthly_sales': { trend: 0.7, seasonality: 0.8, noise: 0.2, points: 24 },
        'website_traffic': { trend: 0.5, seasonality: 0.9, noise: 0.3, points: 30 },
        'stock_price': { trend: 0.3, seasonality: 0.2, noise: 0.5, points: 90 },
        'temperature': { trend: 0.1, seasonality: 0.9, noise: 0.1, points: 36 }
    };
    
    const config = settings[datasetType] || settings.monthly_sales;
    
    // Update sliders
    document.getElementById('trend').value = config.trend;
    document.getElementById('seasonality').value = config.seasonality;
    document.getElementById('noise').value = config.noise;
    document.getElementById('dataPoints').value = config.points;
    
    // Update the display values
    document.getElementById('trendValue').textContent = config.trend.toFixed(1);
    document.getElementById('seasonalityValue').textContent = config.seasonality.toFixed(1);
    document.getElementById('noiseValue').textContent = config.noise.toFixed(1);
    document.getElementById('dataPointsValue').textContent = config.points;
    
    // Update the data
    await updateSampleData();
}

// Update sample data based on user inputs
async function updateSampleData() {
    const nPoints = document.getElementById('dataPoints').value;
    const trend = parseFloat(document.getElementById('trend').value);
    const seasonality = parseFloat(document.getElementById('seasonality').value);
    const noise = parseFloat(document.getElementById('noise').value);
    
    // Update the displayed values
    document.getElementById('trendValue').textContent = trend.toFixed(1);
    document.getElementById('seasonalityValue').textContent = seasonality.toFixed(1);
    document.getElementById('noiseValue').textContent = noise.toFixed(1);
    document.getElementById('dataPointsValue').textContent = nPoints;
    
    try {
        // Call the sample data API
        const response = await fetch(`${API_BASE_URL}/time-series/forecast/sample-data?n_points=${nPoints}&trend=${trend}&seasonality=${seasonality}&noise=${noise}`);
        const data = await response.json();
        
        if (response.ok) {
            // Update the textarea with the sample data
            document.getElementById('trainingData').value = JSON.stringify(data.values, null, 2);
        } else {
            showError('trainingError', `Error loading sample data: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        showError('trainingError', `Error: ${error.message}`);
    }
}

// Train a new model
async function trainModel() {
    // Clear previous errors and show loading
    clearError('trainingError');
    showLoading('trainingStatus', true);
    
    try {
        // Get and validate model ID
        const modelId = document.getElementById('modelId').value.trim();
        if (!modelId) {
            throw new Error('Please enter a model ID');
        }
        
        // Get model type
        const modelType = document.getElementById('modelType').value;
        
        // Parse and validate training data
        let trainingData;
        try {
            trainingData = JSON.parse(document.getElementById('trainingData').value.trim());
            if (!Array.isArray(trainingData)) {
                throw new Error('Training data must be a JSON array of numbers');
            }
            if (trainingData.length < 10) {
                throw new Error('At least 10 data points are required for training');
            }
        } catch (e) {
            throw new Error(`Invalid training data: ${e.message}`);
        }
        
        // Prepare the request data
        const requestData = {
            data: {
                values: trainingData,
                freq: "D"  // Daily frequency
            },
            config: {
                model_type: modelType,
                model_id: modelId,
                p: 1,  // AR order
                d: 1,  // I order (differencing)
                q: 1,  // MA order
                lstm_units: 50,
                epochs: 50,
                batch_size: 16,
                look_back: 7
            }
        };
        
        // Show loading state
        document.getElementById('trainButtonText').textContent = 'Training...';
        document.getElementById('trainingSpinner').classList.remove('d-none');
        
        // Make the API call
        const response = await fetch(`${API_BASE_URL}/time-series/forecast/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to train model');
        }
        
        // Update the model info
        currentModel = {
            id: modelId,
            type: modelType,
            trainingPoints: trainingData.length,
            lastTrained: new Date().toLocaleString(),
            config: requestData.config
        };
        
        updateModelInfo();
        showSuccess('trainingStatus', 'Model trained successfully!');
        
        // Update the UI
        document.getElementById('trainingMetrics').innerHTML = `
            <p class="mb-1"><strong>RMSE:</strong> ${data.metrics?.rmse?.toFixed(4) || 'N/A'}</p>
            <p class="mb-1"><strong>MAE:</strong> ${data.metrics?.mae?.toFixed(4) || 'N/A'}</p>
            <p class="mb-1"><strong>R² Score:</strong> ${data.metrics?.r2_score?.toFixed(4) || 'N/A'}</p>
            <p class="mb-0"><strong>Training Time:</strong> ${data.training_time?.toFixed(2) || 'N/A'}s</p>
        `;
        
    } catch (error) {
        showError('trainingError', `Training failed: ${error.message}`);
        console.error('Training error:', error);
    } finally {
        // Reset loading state
        document.getElementById('trainButtonText').textContent = 'Train Model';
        document.getElementById('trainingSpinner').classList.add('d-none');
        showLoading('trainingStatus', false);
    }
}

// Generate forecast using the trained model
async function generateForecast() {
    if (!currentModel) {
        showError('forecastError', 'Please train a model first');
        return;
    }
    
    // Clear previous errors and show loading
    clearError('forecastError');
    showLoading('forecastStatus', true);
    
    try {
        const forecastSteps = parseInt(document.getElementById('forecastSteps').value) || 5;
        
        // Show loading state
        document.getElementById('forecastButtonText').textContent = 'Generating...';
        document.getElementById('forecastSpinner').classList.remove('d-none');
        
        // Prepare the request data
        const requestData = {
            model_id: currentModel.id,
            steps: forecastSteps
        };
        
        // Make the API call
        const response = await fetch(`${API_BASE_URL}/time-series/forecast/forecast`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to generate forecast');
        }
        
        // Update the chart with the forecast
        updateChart(data);
        
        // Update the forecast values display
        const forecastValues = data.forecast || [];
        const forecastHtml = forecastValues.length > 0 
            ? `<pre>${JSON.stringify(forecastValues, null, 2)}</pre>`
            : '<p class="text-muted">No forecast values available</p>';
        
        document.getElementById('forecastValues').innerHTML = forecastHtml;
        
        showSuccess('forecastStatus', 'Forecast generated successfully!');
        
    } catch (error) {
        showError('forecastError', `Forecast failed: ${error.message}`);
        console.error('Forecast error:', error);
    } finally {
        // Reset loading state
        document.getElementById('forecastButtonText').textContent = 'Generate Forecast';
        document.getElementById('forecastSpinner').classList.add('d-none');
        showLoading('forecastStatus', false);
    }
}

// Update the chart with forecast data
function updateChart(forecastData) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Prepare data for Chart.js
    const labels = forecastData.timestamps || [];
    const forecastValues = forecastData.forecast || [];
    const lowerBounds = forecastData.lower_bound || [];
    const upperBounds = forecastData.upper_bound || [];
    
    // If we have historical data, use it; otherwise, create some
    let historicalLabels = [];
    let historicalValues = [];
    
    if (forecastData.historical_dates && forecastData.historical_values) {
        // Use provided historical data
        historicalLabels = forecastData.historical_dates;
        historicalValues = forecastData.historical_values;
    } else {
        // Generate some historical data (2x forecast length, up to 30 points)
        const histLength = Math.min(forecastValues.length * 2, 30);
        for (let i = 0; i < histLength; i++) {
            if (forecastValues.length > 0) {
                const baseValue = forecastValues[0] * (1 - (histLength - i) * 0.05);
                const noise = (Math.random() - 0.5) * (forecastValues[0] * 0.1);
                historicalValues.push(Math.max(0, baseValue + noise));
            } else {
                historicalValues.push(Math.random() * 100);
            }
            historicalLabels.push(`Day -${histLength - i}`);
        }
    }
    
    // Combine historical and forecast data
    const allLabels = [...historicalLabels, ...labels];
    const allValues = [...historicalValues, ...forecastValues];
    
    // Create or update chart
    if (chart) {
        chart.destroy();
    }
    
    // Find the index where forecast starts
    const forecastStartIndex = historicalLabels.length;
    
    // Prepare datasets
    const datasets = [
        {
            label: 'Historical Data',
            data: allValues.map((v, i) => i < forecastStartIndex ? v : null),
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 3,
            fill: false
        },
        {
            label: 'Forecast',
            data: allValues.map((v, i) => i >= forecastStartIndex ? v : null),
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.3,
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 3,
            fill: false
        }
    ];
    
    // Add confidence interval if available
    if (lowerBounds.length > 0 && upperBounds.length > 0) {
        datasets.push({
            label: 'Confidence Interval',
            data: allValues.map((v, i) => {
                if (i < forecastStartIndex) return null;
                const idx = i - forecastStartIndex;
                return (upperBounds[idx] + lowerBounds[idx]) / 2;
            }),
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderColor: 'rgba(255, 99, 132, 0.1)',
            borderWidth: 0,
            pointRadius: 0,
            fill: true,
            showLine: false
        });
    }
    
    // Create the chart
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2);
                            }
                            return label;
                        }
                    }
                },
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Time Series Forecast',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Value'
                    },
                    beginAtZero: false
                }
            },
            animation: {
                duration: 1000
            }
        }
    });
}

// Update the model information display
function updateModelInfo() {
    if (!currentModel) {
        document.getElementById('modelStatus').textContent = 'No model trained';
        document.getElementById('modelTypeDisplay').textContent = '-';
        document.getElementById('trainingPoints').textContent = '0';
        document.getElementById('lastTrained').textContent = '-';
        return;
    }
    
    document.getElementById('modelStatus').textContent = 'Trained';
    document.getElementById('modelTypeDisplay').textContent = currentModel.type;
    document.getElementById('trainingPoints').textContent = currentModel.trainingPoints;
    document.getElementById('lastTrained').textContent = currentModel.lastTrained;
}

// Download the chart as an image
function downloadChart() {
    if (!chart) {
        alert('No chart available to download');
        return;
    }
    
    const link = document.createElement('a');
    link.download = `forecast-${new Date().toISOString().split('T')[0]}.png`;
    link.href = chart.toBase64Image('image/png', 1);
    link.click();
}

// Helper function to show loading state
function showLoading(elementId, show) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    if (show) {
        element.classList.add('loading');
    } else {
        element.classList.remove('loading');
        element.textContent = '';
    }
}

// Helper function to show success message
function showSuccess(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.textContent = message;
    element.className = 'text-success small mt-2';
    
    // Clear the success message after 5 seconds
    setTimeout(() => {
        element.textContent = '';
    }, 5000);
}

// Helper function to show error message
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.textContent = message;
    element.className = 'text-danger small mt-2';
}

// Helper function to clear error message
function clearError(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = '';
        element.className = 'text-danger small mt-2';
    }
}
