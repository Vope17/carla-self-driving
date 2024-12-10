import carla

print("Start to connect...")
client = carla.Client('localhost', 2000)
client.load_world('Town03')
print("Complete!")