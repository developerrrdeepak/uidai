"""Command Line Interface for UIDAI Risk Monitoring"""
import argparse
from src.pipeline import UIDaiPipeline
from src.utils import ReportExporter, PolicyEngine

def main():
    parser = argparse.ArgumentParser(description='UIDAI Risk Monitoring System')
    parser.add_argument('--export', choices=['csv', 'json', 'excel'], 
                       help='Export format')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--policy-report', action='store_true',
                       help='Generate policy recommendations')
    
    args = parser.parse_args()
    
    # Run pipeline
    print("Running UIDAI Risk Assessment Pipeline...")
    pipeline = UIDaiPipeline()
    results = pipeline.run()
    
    print(f"\n✓ Analysis Complete: {len(results)} districts processed")
    print(f"  High Risk: {len(results[results['Risk Level'] == 'High'])}")
    print(f"  Medium Risk: {len(results[results['Risk Level'] == 'Medium'])}")
    print(f"  Low Risk: {len(results[results['Risk Level'] == 'Low'])}")
    
    # Export results
    if args.export:
        exporter = ReportExporter()
        if args.export == 'csv':
            filename = exporter.export_csv(results, args.output)
        elif args.export == 'json':
            filename = exporter.export_json(results, args.output)
        else:
            filename = exporter.export_excel(results, args.output)
        print(f"\n✓ Report exported: {filename}")
    
    # Policy report
    if args.policy_report:
        policy_engine = PolicyEngine()
        report = policy_engine.generate_report(results)
        print(f"\n📋 POLICY RECOMMENDATIONS")
        print(f"{'='*60}")
        for rec in report['recommendations']:
            print(f"\n{rec['district']} - {rec['urgency']} Priority")
            print(f"  Action: {rec['action']}")
            print(f"  Timeline: {rec['timeline']}")
            print(f"  Budget: {rec['budget']}")

if __name__ == "__main__":
    main()
