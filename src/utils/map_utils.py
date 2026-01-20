"""Map Visualization Utilities"""
import pandas as pd

class MapDataGenerator:
    """Generate data for map visualizations"""
    
    COORDINATES = {
        'Kalahandi': [20.1, 83.2], 'Koraput': [18.8, 82.7], 'Rayagada': [19.2, 83.4],
        'Gajapati': [18.9, 84.1], 'Kandhamal': [20.2, 84.0], 'Bolangir': [20.7, 83.5],
        'Nuapada': [20.8, 82.5], 'Bargarh': [21.3, 83.6], 'Jharsuguda': [21.9, 84.0],
        'Sambalpur': [21.5, 83.9], 'Dhenkanal': [20.7, 85.6], 'Angul': [20.8, 85.1],
        'Deogarh': [21.5, 84.7], 'Sundargarh': [22.1, 84.0], 'Keonjhar': [21.6, 85.6]
    }
    
    def generate(self, results):
        """Generate map data from results"""
        map_data = []
        for _, row in results.iterrows():
            district = row['District']
            if district in self.COORDINATES:
                lat, lon = self.COORDINATES[district]
                map_data.append({
                    'District': district,
                    'lat': lat,
                    'lon': lon,
                    'Risk Score': row['Risk Score'],
                    'Risk Level': row['Risk Level'],
                    'Alert': row['Alert'],
                    'Risk Type': row['Risk Type'],
                    'Action': row['Action']
                })
        return pd.DataFrame(map_data)
