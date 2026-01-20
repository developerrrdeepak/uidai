"""Trend Analysis Utilities"""
import pandas as pd
import numpy as np

class TrendDataGenerator:
    """Generate trend data for district analysis"""
    
    def generate(self, district_name):
        """Generate trend data for a district"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        
        np.random.seed(hash(district_name) % 1000)
        enrolment = 100 + np.cumsum(np.random.normal(0, 5, len(dates)))
        updates = 80 + np.cumsum(np.random.normal(0, 3, len(dates)))
        
        if district_name in ['Kalahandi', 'Koraput']:
            enrolment[-3:] *= 0.7
            updates[-2:] *= 0.6
        
        return pd.DataFrame({
            'Date': dates,
            'Enrolment': enrolment,
            'Updates': updates
        })
