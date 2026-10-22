import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	n = len(consumption_data)
	base_consumption = []
	for i  , data in enumerate(consumption_data):
		fluct_i = 10 * math.sin( 2 * PI * i/10)
		base_consumption.append(data - fluct_i)
	
	# 2) Perform linear regression on the detrended data.
	# Step 2: Fit linear regression using least squares
	x = list(range(1, n + 1))

	sum_x = sum(x)
	sum_y = sum(base_consumption)
	sum_xy = sum(a * b for a, b in zip(x, base_consumption))
	sum_x2 = sum(a * a for a in x)

	denominator = n * sum_x2 - sum_x ** 2

	m = (n * sum_xy - sum_x * sum_y) / denominator
	b = (sum_y - m * sum_x) / n

	# 3) Predict day 15's base consumption.
	day = 15
	base_15 = m * 15 + b
	# 4) Add the day 15 fluctuation back.
	fluct_15 = 10 * math.sin( 2 * PI * 15/10)
	pred_15 = base_15 + fluct_15
	# 5) Round, then add a 5% safety margin (rounded up).
	final_consumption = math.ceil(pred_15 * 1.05)
	# 6) Return the final integer.
	return final_consumption