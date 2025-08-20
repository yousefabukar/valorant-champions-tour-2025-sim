print("🤖 VCT ML Model Test")
print("This should be different from features.py output!")

from config import CURRENT_CHAMPIONSHIP_POINTS

print("\n📊 Championship Points Test:")
for region, teams in CURRENT_CHAMPIONSHIP_POINTS.items():
    print(f"{region}: {len(teams)} teams")

print("\n✅ Model test complete!")
