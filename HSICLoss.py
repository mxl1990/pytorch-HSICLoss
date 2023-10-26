"""
pytorch implementation of Hilbert Schmidt Independence Criterion as HSIC Loss
Python 3 and pytorch 1.8.1 or pytorch 1.12.0

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B.,
& Smola, A. J. (2007). A kernel statistical test of independence.
In Advances in neural information processing systems (pp. 585-592).

XiaoLi (mxl1990 AT gmail.com)
10/26/2023

"""

from __future__ import division
import torch.nn as nn
import torch


def rbf_dot(pattern1, pattern2, deg):
	size1 = pattern1.shape
	size2 = pattern2.shape

	G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
	H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

	Q = np.tile(G, (1, size2[0]))
	R = np.tile(H.T, (size1[0], 1))

	H = Q + R - 2* np.dot(pattern1, pattern2.T)

	H = np.exp(-H/2/(deg**2))

	return H


def hsic_gam(X, Y, alph = 0.5):
	"""
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
	n = X.shape[0]

	# ----- width of X -----
	Xmed = X

	G = np.sum(Xmed*Xmed, 1).reshape(n,1)
	Q = np.tile(G, (1, n) )
	R = np.tile(G.T, (n, 1) )

	dists = Q + R - 2* np.dot(Xmed, Xmed.T)
	dists = dists - np.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
	# ----- -----

	# ----- width of X -----
	Ymed = Y

	G = np.sum(Ymed*Ymed, 1).reshape(n,1)
	Q = np.tile(G, (1, n) )
	R = np.tile(G.T, (n, 1) )

	dists = Q + R - 2* np.dot(Ymed, Ymed.T)
	dists = dists - np.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
	# ----- -----

	# bone = np.ones((n, 1), dtype = float)
	H = np.identity(n) - np.ones((n,n), dtype = float) / n

	K = rbf_dot(X, X, width_x)
	L = rbf_dot(Y, Y, width_y)

	Kc = np.dot(np.dot(H, K), H)
	Lc = np.dot(np.dot(H, L), H)

	testStat = np.sum(Kc.T * Lc) / n

	return testStat


def width_cal(x):
	n_x = x.shape[0]
	xmed = x.clone()
	G = torch.sum(xmed * xmed, 1).reshape(n_x, 1)
	Q = torch.tile(G, (1, n_x))
	R = torch.tile(G.T, (n_x, 1))

	dists = Q + R - 2 * xmed.mm(xmed.T)
	dists = dists - torch.tril(dists)
	dists = dists.reshape(n_x**2, 1)

	width_x = torch.sqrt(0.5 * torch.median(dists[dists > 0]))
	return width_x


def rbf_mm(pattern1, pattern2, deg):
	size1 = pattern1.shape
	size2 = pattern2.shape

	G = torch.sum(pattern1 * pattern1, 1).view(size1[0],1)
	H = torch.sum(pattern2 * pattern2, 1).view(size2[0],1)

	Q = torch.tile(G, (1, size2[0]))
	R = torch.tile(H.T, (size1[0], 1))

	H = Q + R - 2 * torch.mm(pattern1, pattern2.T)

	H = torch.exp(-H / 2 / (deg**2))

	return H


class HISCLoss(nn.Module):
	def __init__(self, alpha=0.5):
		super(HISCLoss, self).__init__()
		self.alpha = alpha

	def forward(self, x, y):
		n_x = x.shape[0]
		n_y = y.shape[0]

		# assert n_x == n_y  # may be nor right

		x = x.view(n_x, -1)
		y = y.view(n_y, -1)

		width_x = width_cal(x)
		width_y = width_cal(y)

		H = torch.eye(n_x) - torch.ones((n_x, n_x), dtype=torch.float32) / n_x
		H = H.cuda()

		K = rbf_mm(x, x, width_x)
		L = rbf_mm(y, y, width_y)

		Kc = torch.mm(torch.mm(H, K), H)
		Lc = torch.mm(torch.mm(H, L), H)

		testStat = torch.sum(Kc.T * Lc) / n_x

		return testStat


if __name__ == "__main__":
	# only use hsic_gam for test
	import numpy as np
	h_loss = HISCLoss()
	x = torch.randn(64, 3, 224, 224)
	y = torch.randn(64, 3, 224, 224)
	x_np = np.array(x).reshape(64, -1)
	y_np = np.array(y).reshape(64, -1)
	# x.require_grad = True
	t = h_loss(x, y)
	tt = hsic_gam(x_np, y_np)
	# the result is the same
	print(t)
	print(tt)
