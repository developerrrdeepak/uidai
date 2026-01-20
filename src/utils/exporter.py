"""Export Utilities"""
import pandas as pd
from datetime import datetime

class ReportExporter:
    """Export reports in multiple formats"""
    
    def export_csv(self, results, filename=None):
        """Export results to CSV"""
        if filename is None:
            filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(filename, index=False)
        return filename
    
    def export_excel(self, results, filename=None):
        """Export results to Excel with formatting"""
        if filename is None:
            filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            results.to_excel(writer, sheet_name='Risk Assessment', index=False)
            
            # Add summary sheet
            summary = pd.DataFrame({
                'Metric': ['Total Districts', 'High Risk', 'Medium Risk', 'Low Risk'],
                'Count': [
                    len(results),
                    len(results[results['Risk Level'] == 'High']),
                    len(results[results['Risk Level'] == 'Medium']),
                    len(results[results['Risk Level'] == 'Low'])
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
        return filename
    
    def export_json(self, results, filename=None):
        """Export results to JSON"""
        if filename is None:
            filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results.to_json(filename, orient='records', indent=2)
        return filename
