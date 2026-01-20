"""
Flask API Backend for UIDAI React Dashboard
"""
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

def load_data():
    """Load and process data"""
    try:
        bio_df = pd.read_csv('aadhaar_biometric_cleaned.csv')
        demo_df = pd.read_csv('aadhaar_demographic_cleaned.csv')
        
        # Aggregate by district
        bio_agg = bio_df.groupby('district_clean').agg({'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'}).reset_index()
        demo_agg = demo_df.groupby('district_clean').agg({'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'}).reset_index()
        
        df = pd.merge(bio_agg, demo_agg, on='district_clean', how='outer').fillna(0)
        df['total_bio'] = df['bio_age_5_17'] + df['bio_age_17_']
        df['total_demo'] = df['demo_age_5_17'] + df['demo_age_17_']
        df['coverage_ratio'] = df['total_bio'] / (df['total_demo'] + 1)
        df['risk_score'] = 1 - df['coverage_ratio']
        df['risk_level'] = pd.cut(df['risk_score'], bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Get all districts with risk data"""
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data available'}), 404
    
    df['risk_level'] = df['risk_level'].astype(str).replace('nan', 'Low')
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    data = df[['district_clean', 'risk_score', 'risk_level', 'total_bio', 'total_demo']].to_dict('records')
    return jsonify(data)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get summary statistics"""
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data available'}), 404
    
    df['risk_level'] = df['risk_level'].astype(str)
    stats = {
        'total_districts': int(len(df)),
        'high_risk': int(len(df[df['risk_level'] == 'High'])),
        'medium_risk': int(len(df[df['risk_level'] == 'Medium'])),
        'low_risk': int(len(df[df['risk_level'] == 'Low'])),
        'avg_risk_score': float(df['risk_score'].replace([np.inf, -np.inf], np.nan).mean())
    }
    return jsonify(stats)

@app.route('/api/top-risk', methods=['GET'])
def get_top_risk():
    """Get top 10 high-risk districts"""
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data available'}), 404
    
    df['risk_level'] = df['risk_level'].astype(str).replace('nan', 'Low')
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    top_risk = df.nlargest(10, 'risk_score')[['district_clean', 'risk_score', 'risk_level']].to_dict('records')
    return jsonify(top_risk)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
