import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import io
import base64

# Use Agg backend for non-interactive plotting (required for server)
matplotlib.use('Agg')

def generate_sample_time_series(
    n_points: int = 30,
    trend: float = 0.5,
    seasonality: float = 0.5,
    noise: float = 0.2
) -> Tuple[List[datetime], List[float]]:
    """Generate sample time series data for demonstration purposes.
    
    Args:
        n_points: Number of data points to generate
        trend: Strength of the trend component
        seasonality: Strength of the seasonality component
        noise: Amount of random noise to add
        
    Returns:
        Tuple of (dates, values) for the generated time series
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_points-1)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    
    # Generate trend component
    x = np.linspace(0, 10, n_points)
    trend_component = trend * x
    
    # Generate seasonality component (weekly seasonality)
    seasonality_component = seasonality * np.sin(2 * np.pi * np.arange(n_points) / 7)
    
    # Generate noise
    noise_component = noise * np.random.normal(0, 1, n_points)
    
    # Combine components
    values = 10 + trend_component + seasonality_component + noise_component
    
    return dates, values.tolist()

def create_forecast_plot(
    historical_dates: Optional[List[datetime]] = None,
    historical_values: Optional[List[float]] = None,
    forecast_dates: Optional[List[datetime]] = None,
    forecast_values: Optional[List[float]] = None,
    lower_bounds: Optional[List[float]] = None,
    upper_bounds: Optional[List[float]] = None,
    title: str = "Time Series Forecast",
    x_label: str = "Date",
    y_label: str = "Value",
    figsize: Tuple[int, int] = (12, 6),
    use_sample_data: bool = False
) -> str:
    """
    Create a time series forecast plot with historical data and forecasted values.
    
    Args:
        historical_dates: List of datetime objects for historical data (or None to use sample data)
        historical_values: List of historical values (or None to use sample data)
        forecast_dates: List of datetime objects for forecasted values
        forecast_values: List of forecasted values
        lower_bounds: Optional list of lower bound values for confidence interval
        upper_bounds: Optional list of upper bound values for confidence interval
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        figsize: Figure size as (width, height)
        use_sample_data: If True, generates sample data when no data is provided
    """
    # Generate sample data if requested or if no data is provided
    if use_sample_data or not all([historical_dates, historical_values]):
        sample_dates, sample_values = generate_sample_time_series(n_points=30)
        historical_dates = historical_dates or sample_dates
        historical_values = historical_values or sample_values
        
        # If no forecast is provided, generate a simple forecast
        if not all([forecast_dates, forecast_values]):
            n_forecast = 10
            last_date = max(historical_dates)
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, n_forecast + 1)]
            
            # Simple forecast: linear extrapolation of the last 5 points
            last_values = np.array(historical_values[-5:])
            x = np.arange(len(last_values))
            coeffs = np.polyfit(x, last_values, 1)
            forecast_values = (coeffs[0] * np.arange(1, n_forecast + 1) + coeffs[1]).tolist()
            
            # Add some uncertainty
            noise = np.random.normal(0, 0.5, n_forecast)
            forecast_values = [v + n for v, n in zip(forecast_values, noise)]
            
            # Add confidence intervals
            std_dev = np.std(historical_values[-10:])  # Use recent volatility
            lower_bounds = [v - std_dev for v in forecast_values]
            upper_bounds = [v + std_dev for v in forecast_values]
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot historical data
    plt.plot(
        historical_dates, 
        historical_values, 
        'b-', 
        label='Historical Data',
        linewidth=2
    )
    
    # Plot forecasted values
    forecast_line = plt.plot(
        forecast_dates, 
        forecast_values, 
        'r-', 
        label='Forecast',
        linewidth=2
    )
    
    # Plot confidence interval if provided
    if lower_bounds is not None and upper_bounds is not None and len(lower_bounds) == len(upper_bounds) == len(forecast_values):
        plt.fill_between(
            forecast_dates,
            lower_bounds,
            upper_bounds,
            color='red',
            alpha=0.2,
            label=f'{int(0.95 * 100)}% Confidence Interval'
        )
    
    # Add vertical line at the end of historical data
    if historical_dates and forecast_dates:
        last_historical_date = max(historical_dates)
        plt.axvline(
            x=last_historical_date, 
            color='k', 
            linestyle='--', 
            linewidth=1,
            label='Forecast Start'
        )
    
    # Customize the plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Encode plot to base64 string
    data = base64.b64encode(buf.getbuffer()).decode('ascii')
    return f"data:image/png;base64,{data}"
