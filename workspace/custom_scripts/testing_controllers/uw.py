import torch

# Check if CUDA is available at all
print(f"CUDA available: {torch.cuda.is_available()}")

# Check number of CUDA devices
print(f"Number of CUDA devices: {torch.cuda.device_count()}")

# List all available devices
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Check if cuda:1 specifically is available
if torch.cuda.device_count() > 1:
    print("cuda:1 is available!")
    print(f"cuda:1 device name: {torch.cuda.get_device_name(1)}")
else:
    print("cuda:1 is NOT available")


import torch

def check_cuda_devices():
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system")
        return
    
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA is available with {device_count} device(s)")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n📱 Device {i} (cuda:{i}):")
        print(f"   Name: {props.name}")
        print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
    
    # Check if cuda:1 specifically exists
    if device_count > 1:
        print(f"\n✅ cuda:1 is available!")
        
        # Test if you can actually use cuda:1
        try:
            test_tensor = torch.tensor([1.0]).to('cuda:1')
            print(f"✅ Successfully created tensor on cuda:1")
        except Exception as e:
            print(f"❌ Error using cuda:1: {e}")
    else:
        print(f"\n❌ cuda:1 is NOT available (only {device_count} device(s) found)")

# Run the check
check_cuda_devices()


# Add this at the very beginning of your simulation.py file
import torch
import os

print("=== CUDA Device Check ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

if torch.cuda.device_count() > 1:
    print("✅ cuda:1 is available!")
    for i in range(torch.cuda.device_count()):
        print(f"   cuda:{i} - {torch.cuda.get_device_name(i)}")
else:
    print("❌ cuda:1 is NOT available")
print("========================")