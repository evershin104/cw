import matplotlib.pyplot as plt


class Plotter:

	def __init__(self, control, model, x_hat, name):
		self.control = control
		self.model = model
		self.x_hat = x_hat
		self.name = name
		self.__plot_model()

	def __plot_model(self):
		for i in range(0, len(self.model[0])):
			plt.plot(
				self.model.keys(), [j[i] for j in self.model.values()],
				label = f'x{i + 1} = {self.model[0][i]}'
				)
		self.control.append(0)
		plt.plot(
			self.model.keys(),
			self.control,
			label = 'Control',
			color = 'blue'
			)
		plt.hlines(
			self.x_hat, 0,
			len(self.model),
			colors = 'black',
			linestyles = '--',
			label = 'Goal'
			)
		plt.hlines(
			0, 0,
			len(self.model),
			colors = 'red',
			linestyles = 'dotted',
			label = '0'
		)
		plt.title(r'Model evolution, $\xi \sim \mathcal{N}(\mu = 0, \sigma^{2} = 0.3^2)$')
		# , $X \sim \mathcal{N}(\mu = 0, \sigma^{2} = 0.2)$
		plt.xlabel('Time')
		plt.ylabel('Value')
		plt.legend(title = 'Initial opinions')
		# plt.text(0, 0, r'$X \sim \mathcal{N}(\mu = 0.5, \sigma^{2} = 1)$')
		plt.savefig(self.name)
		plt.clf()
