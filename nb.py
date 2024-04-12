import numpy as np
RANDOM_SEED = 5


class NaiveBayes(object):

    def __init__(self):
        pass

    def likelihood_ratio(self, ratings_Sentiments):
        """		
		Args:
		    ratings_Sentiments: a python list of numpy arrays that is length <number of labels> x 1
		
		    Example rating_Sentiments for three-label NB model:
		
		    ratings_Sentiments = [ratings_1, ratings_2, ratings_3] -- length 3
		    ratings_1: N_ratings_1 x D
		        where N_ratings_1 is the number of negative news that we have,
		        and D is the number of features (we use the word count as the feature)
		    ratings_2: N_ratings_2 x D
		        where N_ratings_2 is the number of neutral news that we have,
		        and D is the number of features (we use the word count as the feature)
		    ratings_3: N_ratings_3 x D
		        where N_ratings_3 is the number of positive news that we have,
		        and D is the number of features (we use the word count as the feature)
		
		Return:
		    likelihood_ratio: (<number of labels>, D) numpy array, the likelihood ratio of different words for the different classes of sentiments.
		"""
        total_word_counts = [np.sum(ratings, axis=0) for ratings in ratings_Sentiments]
        class_totals = [np.sum(counts) for counts in total_word_counts]
        likelihoods = [counts / total for counts, total in zip(total_word_counts, class_totals)]
        reference_likelihood = likelihoods[0]
        likelihood_ratios = [likelihood / (reference_likelihood + 1e-9) for likelihood in likelihoods]
        likelihood_ratio_array = np.stack(likelihood_ratios, axis=0)
        return likelihood_ratio_array

    def priors_prob(self, ratings_Sentiments):
        """		
		Args:
		    ratings_Sentiments: a python list of numpy arrays that is length <number of labels> x 1
		
		    Example rating_Sentiments for Three-label NB model:
		
		    ratings_Sentiments = [ratings_1, ratings_2, ratings_3] -- length 3
		    ratings_1: N_ratings_1 x D
		        where N_ratings_1 is the number of negative news that we have,
		        and D is the number of features (we use the word count as the feature)
		    ratings_2: N_ratings_2 x D
		        where N_ratings_2 is the number of neutral news that we have,
		        D is the number of features (we use the word count as the feature)
		    ratings_3: N_ratings_3 x D
		        where N_ratings_3 is the number of positive news that we have,
		        and D is the number of features (we use the word count as the feature)
		
		Return:
		    priors_prob: (1, <number of labels>) numpy array, where each entry denotes the prior probability for each class
		"""
        class_counts = np.array([len(ratings) for ratings in ratings_Sentiments])
        total_count = class_counts.sum()
        priors_prob = class_counts / total_count
        priors_prob = priors_prob.reshape(1, -1)
        return priors_prob

    def analyze_sentiment(self, likelihood_ratio, priors_prob, X_test):
        """		
		Args:
		    likelihood_ratio: (<number of labels>, D) numpy array, the likelihood ratio of different words for different classes of ratings
		    priors_prob: (1, <number of labels>) numpy array, where each entry denotes the prior probability for each class
		    X_test: (N_test, D) numpy array, a bag of words representation of the N_test number of ratings that we need to analyze
		Return:
		    ratings: (N_test,) numpy array, where each entry is a class label specific for the Naïve Bayes model
		"""
        log_likelihoods = np.dot(X_test, np.log(likelihood_ratio).T)
        log_posteriors = log_likelihoods + np.log(priors_prob)
        ratings = np.argmax(log_posteriors, axis=1)
        return ratings
