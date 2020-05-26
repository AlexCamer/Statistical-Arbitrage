import json
import matplotlib.pyplot as plt
from agent import *
from broker import *





# read in data for back test
df = pd.read_csv('Cache/prices.csv', index_col=0)
with open('Cache/sectors.json', 'r') as json_file:
    sectors = json.load(json_file)

# initialize agent and environment
broker = Broker(df, window=120)
agent = Agent(sectors, trade_size=1e4)

# run simulation loop
prices = broker.init()
net_gains, leverages = [], []
calibration_period = 60
i = 0
done = False
while not done:
    orders = []
    if i % calibration_period == 0: orders += agent.calibrate(prices)
    ords, lev = agent.get_orders(prices)
    orders += ords
    prices, net_gain, done = broker.step(orders)
    net_gains.append(net_gain)
    leverages.append(lev)
    i += 1
    print(f'Day: {i}\tValue: {net_gain}')

# calculate return on leverage (RoL)
total_RoL = (net_gains[-1] / np.mean(leverages)) * 100
annual_RoL = round(total_RoL / (len(df) / 365), 2)

# display results
print(f'\nAverage annual return on leverage: {annual_RoL}%')
plt.plot(net_gains)
plt.show()