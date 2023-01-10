# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layer.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kkim <kkim@student.42.fr>                  +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/01/04 14:19:03 by kkim              #+#    #+#              #
#    Updated: 2023/01/10 12:58:22 by kkim             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# ------------------------------------------------------------------------------
# import : DSLR
from DSLR.util import OptType as opt

# import : library
import numpy

# ------------------------------------------------------------------------------
# Layer: BaseLayer
#   각 층의 부모 클래스 생성
class BaseLayer:
	def __init__(self, _opt_type):
		self._opt_type = _opt_type
		if self._opt_type == opt.SGD:
			# self._optimizer = 
			print("SGD")
		elif self._opt_type == opt.MOMENTUM:
			print("MOMENTUM")
		elif self._opt_type == opt.ADAGRAD:
			print("ADAGRAD")
		elif self._opt_type == opt.RMSPROP:
			print("RMSPROP")
		elif self._opt_type == opt.ADAM:
			print("ADAM")

	def update(self, _learn):
		self.optimizer.update(_learn, [self._w, self._b], [self._grad_w, self._grad_b])

# ------------------------------------------------------------------------------
# Layer: MiddleLayer
class MiddleLayer(BaseLayer):
	def __init__(self, _opt_type, _init_type, _len_in, _len_out):
		super().__init__(_opt_type)
		if _init_type == "01":
			_w_b_width = 0.1
			self._w = _w_b_width * numpy.random.randn(_len_in, _len_out)	# 가중치
			self._b = _w_b_width * numpy.random.randn(_len_out)				# 편향
		elif _init_type == "02": # He 초기값
			self._w = numpy.random.randn(_len_in, _len_out) * numpy.sqrt(2 / _len_in)   # 가중치
			self._b = numpy.zeros(_len_out)                                             # 편향

	def forward(self, _x):
		self._x = _x
		self._u = numpy.dot(_x, self._w) + self._b

		# self._y = numpy.where(self._u <= 0, 0.0, self._u)				# ReLU
		self._y = numpy.where(self._u <= 0, 0.2 * self._u, self._u)		# Leaky ReLU
		# self._y = 1 / (1+numpy.exp(-self._u))							# Sigmoid

	def backward(self, _grad_y):
		# delta = _grad_y * numpy.where(self._u <= 0, 0, 1)		# ReLU 미분
		delta = _grad_y * numpy.where(self._u <= 0, 0.2, 1)		# Leaky ReLU 미분
		# delta = (_grad_y										# Sigmoid 미분
		# 		* 1 / (1 + numpy.exp(-self._u))
		# 		* (1 - 1 / (1 + numpy.exp(-self.u))))

		self._grad_w = numpy.dot(self._x.T, delta)
		self._grad_b = numpy.sum(delta, axis=0)
		self._grad_x = numpy.dot(delta, self._w.T)

# ------------------------------------------------------------------------------
# Layer: OutputLayer
class OutputLayer(BaseLayer):
	def __init__(self, _opt_type, _init_type, _len_in, _len_out):
		super().__init__(_opt_type)
		if _init_type == "01":
			_w_b_width = 0.1
			self._w = _w_b_width * numpy.random.randn(_len_in, _len_out)	# 가중치
			self._b = _w_b_width * numpy.random.randn(_len_out)				# 편향
		elif _init_type == "02": # 자비에르 초기화 기반의 초기값
			self._w = numpy.random.randn(_len_in, _len_out) * numpy.sqrt(_len_in)	# 가중치
			self._b = numpy.zeros(_len_out)											# 편향

	def forward(self, _x):
		self._x = _x
		_u		= numpy.dot(_x, self._w) + self._b
		_eu		= numpy.exp(_u)
		self._y = _eu / numpy.sum(_eu, axis=1, keepdims=True)	# Softmax

	def backward(self, _t):
		delta = self._y - _t
		self._grad_w = numpy.dot(self._x.T, delta)
		self._grad_b = numpy.sum(delta, axis=0)
		self._grad_x = numpy.dot(delta, self._w.T)