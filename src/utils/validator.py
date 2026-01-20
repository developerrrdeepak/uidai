"""Data Validation Module"""
import pandas as pd

class DataValidator:
    """Validate data quality and completeness"""
    
    def validate(self, df):
        """Run validation checks"""
        checks = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_districts': df['district'].duplicated().sum(),
            'valid': True,
            'warnings': []
        }
        
        # Check for missing critical columns
        required = ['district', 'enrolment_pct_change', 'update_pct_change']
        missing = [col for col in required if col not in df.columns]
        if missing:
            checks['valid'] = False
            checks['warnings'].append(f"Missing columns: {missing}")
        
        # Check for extreme outliers
        if 'enrolment_pct_change' in df.columns:
            extreme = df[abs(df['enrolment_pct_change']) > 100]
            if len(extreme) > 0:
                checks['warnings'].append(f"{len(extreme)} extreme outliers detected")
        
        return checks
