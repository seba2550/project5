# Write your silhouette score unit tests here
from cluster import (KMeans, make_clusters, Silhouette)
import pytest
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

	def test_silhouette_range(self):
		"""
		Test that the silhouette scores are between -1 and 1.
		"""
		clusters, labels = make_clusters(k=4, scale=1)
		km = KMeans(k=4)
		km.fit(clusters)
		pred = km.predict(clusters)
		scores = Silhouette().score(clusters, pred)

		for i in scores:
			self.assertTrue(-1 <= i <= 1)

	def test_silhouette_n_scores(self):
		"""
		Test that we get a score for each observation
		"""
		clusters, labels = make_clusters(k=4, scale=1)
		km = KMeans(k=4)
		km.fit(clusters)
		pred = km.predict(clusters)
		scores = Silhouette().score(clusters, pred)

		n_observations = clusters.shape[0]
		n_scores = len(scores)

		assert n_observations == n_scores