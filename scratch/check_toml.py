import tomllib

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

extras = data.get("project", {}).get("optional-dependencies", {})
print(f"Extras found: {list(extras.keys())}")
print(f"Dev dependencies: {extras.get('dev', [])}")
print(f"Viz dependencies: {extras.get('viz', [])}")
