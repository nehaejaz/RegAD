import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

resnet_data = pd.read_csv('./GPU-stats-con-tiny-stn.csv')
resnet_data[' timestamp'] = pd.to_datetime(resnet_data[' timestamp'])
resnet_data[' timestamp'] = resnet_data[' timestamp'].dt.strftime('%H:%M:%S')
resnet_data[' power.draw [W]'] = resnet_data[' power.draw [W]'].str.replace('W', '').astype(float)
print(resnet_data.head())

plt.figure(figsize=(20,5))
plt.plot(np.arange(0,len(resnet_data[' power.draw [W]']),1), resnet_data[' power.draw [W]'], color='coral', label='Power Draw')
plt.xlabel('Timestamps (every 10 seconds)')
plt.xticks(rotation=30)
plt.ylabel('Power Draw (W)')
plt.title('GPU Power Draw during Training')
plt.savefig('con-tiny-stn.png')  # Save as a PNG file
plt.show()

total_energy_usage = resnet_data[' power.draw [W]'].sum()
kWh = total_energy_usage * ( len(resnet_data[' power.draw [W]']) / 3600 ) / 1000
print(f' Total energy usage = {total_energy_usage}W or {round(kWh, 2)}KWh')