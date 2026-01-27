# Test Adapter
from app.adapters.core_adapter import get_core_adapter

print("Testing CoreAdapter...")
adapter = get_core_adapter()
print("Success! Adapter created.")
print("Supported AA:", len(adapter.get_supported_aa()), "amino acids")
