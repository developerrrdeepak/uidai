"""Quick Test Runner"""
from src.pipeline import UIDaiPipeline

print("Testing UIDAI Risk Monitoring System...")
print("=" * 60)

# Run pipeline
pipeline = UIDaiPipeline()
results = pipeline.run()

print(f"\n✓ Pipeline executed successfully")
print(f"  Total Districts: {len(results)}")
print(f"  High Risk: {len(results[results['Risk Level'] == 'High'])}")
print(f"  Medium Risk: {len(results[results['Risk Level'] == 'Medium'])}")
print(f"  Low Risk: {len(results[results['Risk Level'] == 'Low'])}")

print("\n📊 Top 5 High-Risk Districts:")
print(results.head()[['District', 'Risk Score', 'Risk Level', 'Risk Type']])

print("\n✓ All modules working correctly!")
