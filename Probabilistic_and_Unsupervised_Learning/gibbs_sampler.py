# -*- coding: utf-8 -*-

"""
    File name: gibbs_sampler.py
    Description: a re-implementation of the Gibbs sampler for http://www.gatsby.ucl.ac.uk/teaching/courses/ml1
    Author: python: Roman Pogodin, MATLAB (original): Yee Whye Teh and Maneesh Sahani
    Date created: October 2018
    Python version: 3.6
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)


class GibbsSampler:
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          number of documents
        :param n_topics:        number of topics
        :param n_words:         number of words in vocabulary
        :param alpha:           dirichlet parameter on topic mixing proportions
        :param beta:            dirichlet parameter on topic word distributions
        :param random_seed:     random seed of the sampler
        """
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.n_words = n_words
        self.alpha = alpha
        self.beta = beta
        self.rand_gen = np.random.RandomState(random_seed)

        self.docs_words = np.zeros((self.n_docs, self.n_words))
        self.docs_words_test = None
        self.loglike = None
        self.loglike_test = None
        self.do_test = False

        self.A_dk = np.zeros((self.n_docs, self.n_topics))  # number of words in document d assigned to topic k
        self.B_kw = np.zeros((self.n_topics, self.n_words))  # number of occurrences of word w assigned to topic k
        self.A_dk_test = np.zeros((self.n_docs, self.n_topics))
        self.B_kw_test = np.zeros((self.n_topics, self.n_words))

        self.theta = np.ones((self.n_docs, self.n_topics)
                             ) / self.n_topics  # theta[d] is the distribution over topics in document d
        self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words  # phi[k] is the distribution words in topic k

        self.topics_space = np.arange(self.n_topics)
        self.topic_doc_words_distr = np.zeros((self.n_topics, self.n_docs, self.n_words))  # z_id|x_id, theta, phi

    def init_sampling(self, docs_words, docs_words_test=None,
                      theta=None, phi=None, n_iter=0, save_loglike=False):
        assert np.all(docs_words.shape == (self.n_docs, self.n_words)), "docs_words shape=%s must be (%d, %d)" % (
            docs_words.shape, self.n_docs, self.n_words)
        self.n_docs = docs_words.shape[0]

        self.docs_words = docs_words
        self.docs_words_test = docs_words_test

        self.do_test = (docs_words_test is not None)

        if save_loglike:
            self.loglike = np.zeros(n_iter)

            if self.do_test:
                self.loglike_test = np.zeros(n_iter)

        self.A_dk.fill(0.0)
        self.B_kw.fill(0.0)
        self.A_dk_test.fill(0.0)
        self.B_kw_test.fill(0.0)

        self.init_params(theta, phi)

    def init_params(self, theta=None, phi=None):
        if theta is None:
            self.theta = np.ones((self.n_docs, self.n_topics)) / self.n_topics
        else:
            self.theta = theta.copy()

        if phi is None:
            self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words
        else:
            self.phi = phi.copy()

        self.update_topic_doc_words()
        self.sample_counts()

    def run(self, docs_words, docs_words_test=None,
            n_iter=100, theta=None, phi=None, save_loglike=False):
        """
        docs_words is a matrix n_docs * n_words; each entry
        is a number of occurrences of a word in a document
        docs_words_test does not influence the updates and is used
        for validation
        """
        self.init_sampling(docs_words, docs_words_test,
                           theta, phi, n_iter, save_loglike)

        for iteration in range(n_iter):
            self.update_params()

            if save_loglike:
                self.update_loglike(iteration)

        return self.to_return_from_run()

    def to_return_from_run(self):
        return self.topic_doc_words_distr, self.theta, self.phi

    def update_params(self):
        """
        Samples theta and phi, then computes the distribution of
        z_id and samples counts A_dk, B_kw from it
        """
        # For each document, an element of theta is sampled from a dirichlet distribution
        self.theta = np.array([self.rand_gen.dirichlet(self.A_dk[d, :] + self.alpha) 
                       for d in range(self.n_docs)])
        
        # For each topic, an element of phi is sampled from a dirichlet distribution
        self.phi = np.array([self.rand_gen.dirichlet(self.B_kw[k, :] + self.beta) 
                       for k in range(self.n_topics)])

        self.update_topic_doc_words()
        self.sample_counts()

    def update_topic_doc_words(self):
        """
        Computes the distribution of z_id|x_id, theta, phi
        """
        self.topic_doc_words_distr = np.repeat(
            self.theta.T[:, :, None], self.n_words, axis=2) * self.phi[:, None, :]
        self.topic_doc_words_distr /= self.theta.dot(self.phi)[None, :, :]

    def sample_counts(self):
        """
        For each document and each word, samples from z_id|x_id, theta, phi
        and adds the results to the counts A_dk and B_kw
        """
        # Reset A_dk and B_kw
        self.A_dk.fill(0)
        self.B_kw.fill(0)

        if self.do_test:
            self.A_dk_test.fill(0)
            self.B_kw_test.fill(0)
        else:
            print('')
            pass

        for doc in range(self.n_docs):
            for word in range(self.n_words):
                # If there are occurences of this word in this document in the training data
                if self.docs_words[doc, word] > 0:
                    # Get the topic distribution
                    probabilities = self.topic_doc_words_distr[:, doc, word]

                    # Sample topics for all occurrences of this word in this document
                    sampled_topics = self.rand_gen.choice(self.topics_space,
                                                          size=self.docs_words[doc, word],
                                                          p=probabilities)

                    # Update counts A_dk and B_kw
                    sample, counts = np.unique(sampled_topics, return_counts=True)
                    self.A_dk[doc, sample] += counts
                    self.B_kw[sample, word] += counts

                # If there are occurences of this word in this document in the test data
                if self.docs_words_test[doc, word] > 0:
                    word_counts_test = self.docs_words_test[doc, word]

                    # Get the topic distribution
                    probabilities = self.topic_doc_words_distr[:, doc, word]

                    # Sample topics for all occurrences of this word in this document
                    sampled_topics_test = self.rand_gen.choice(self.topics_space,
                                                               size=word_counts_test,
                                                               p=probabilities)

                    # Update test counts A_dk_test and B_kw_test
                    sample_test, counts_test = np.unique(sampled_topics_test, return_counts=True)
                    self.A_dk_test[doc, sample_test] += counts_test
                    self.B_kw_test[sample_test, word] += counts_test

        pass

    def update_loglike(self, iteration): #----# Predictive
        """
        Updates loglike of the data, omitting the constant additive term
        with Gamma functions of hyperparameters
        """

        # Term 1 and 2
        term_1 = np.sum((self.B_kw + self.beta - 1) * np.log(self.phi)) # sum across both axes
        term_2 = np.sum((self.A_dk + self.alpha - 1) * np.log(self.theta)) # sum across both axes

        # Term 3 and 4: constant terms (we can omit?)
        term_3 = self.n_docs*(gammaln(self.n_topics*self.alpha) - self.n_topics*gammaln(self.alpha))
        term_4 = self.n_topics*(gammaln(self.n_words*self.beta) - self.n_words*gammaln(self.beta))

        self.loglike[iteration] = (term_1 + term_2)
        
        if self.do_test:
            # Taken from MATLAB template code
            pred_prob = self.theta @ self.phi
            log_pred = np.multiply(self.docs_words_test, np.log(pred_prob))
            self.loglike_test[iteration] = np.sum(log_pred)
        pass

    def get_loglike(self):
        """Returns log-likelihood at each iteration."""
        if self.do_test:
            return self.loglike, self.loglike_test
        else:
            return self.loglike


class GibbsSamplerCollapsed(GibbsSampler):
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          number of documents
        :param n_topics:        number of topics
        :param n_words:         number of words in vocabulary
        :param alpha:           dirichlet parameter on topic mixing proportions
        :param beta:            dirichlet parameter on topic word distributions
        :param random_seed:     random seed of the sampler
        """
        super().__init__(n_docs, n_topics, n_words, alpha, beta, random_seed)

        # topics assigned to each (doc, word)
        self.doc_word_samples = np.ndarray((self.n_docs, self.n_words), dtype=object)
        self.doc_word_samples_test = self.doc_word_samples.copy()

    def init_params(self, theta=None, phi=None):
        # z_id are initialized uniformly
        for doc in range(self.n_docs):
            for word in range(self.n_words):
                if self.do_test:
                    additional_samples = self.docs_words_test[doc, word]
                else:
                    additional_samples = 0

                sampled_topics = self.rand_gen.choice(self.topics_space,
                                                      size=self.docs_words[doc, word] + additional_samples)

                sampled_topics_train = sampled_topics[:self.docs_words[doc, word]]
                self.doc_word_samples[doc, word] = sampled_topics_train.copy()  # now each cell is an np.array

                sample, counts = np.unique(sampled_topics_train, return_counts=True)

                self.A_dk[doc, sample] += counts
                self.B_kw[sample, word] += counts

                if self.do_test:
                    sampled_topics_test = sampled_topics[self.docs_words[doc, word]:]
                    self.doc_word_samples_test[doc, word] = sampled_topics_test.copy()

                    sample, counts = np.unique(sampled_topics_test, return_counts=True)

                    self.A_dk_test[doc, sample] += counts
                    self.B_kw_test[sample, word] += counts

    def update_params(self):
        """
        Computes the distribution of z_id.
        Sampling of A_dk, B_kw is done automatically as
        each new z_id updates these counters
        """

        for doc in range(self.n_docs):
            for word in range(self.n_words):
                
                current_topics = self.doc_word_samples[doc, word]
                num_occurrences = len(current_topics)
                
                for i in range(num_occurrences):
                    # Get the current topic assignment for this occurrence
                    k_old = current_topics[i]
                    
                    # 'Un-assign' the word from its current topic: 
                    self.A_dk[doc, k_old] -= 1
                    self.B_kw[k_old, word] -= 1
                    
                    # Calculate the marginal probability 
                    
                    # M_k_minus_id is the total word count for topic k before re-assignment
                    M_k_minus_id = np.sum(self.B_kw, axis=1)

                    # Term 1
                    term_topic = self.A_dk[doc, :] + self.alpha
                    
                    # Term 2
                    term_word = (self.B_kw[:, word] + self.beta) / (M_k_minus_id + self.n_words * self.beta)
                    
                    # Compute the probability array for all k topics
                    probabilities = term_topic * term_word
                    
                    # Normalize the probabilities
                    probabilities /= np.sum(probabilities)
                    
                    # Sample the new topic k_new
                    k_new = self.rand_gen.choice(self.topics_space, p=probabilities)
                    
                    # 'Re-assign' the word to the new topic k_new
                    self.A_dk[doc, k_new] += 1
                    self.B_kw[k_new, word] += 1
                    
                    # Update the stored topic assignment
                    current_topics[i] = k_new

                # Update the stored array with the resampled topics
                self.doc_word_samples[doc, word] = current_topics
        
        # Likewise for test
        if self.do_test:

            for doc in range(self.n_docs):
                for word in range(self.n_words):

                    current_topics = self.doc_word_samples_test[doc, word]
                    num_occurrences = len(current_topics)

                    for i in range(num_occurrences):
                        k_old = current_topics[i]

                        self.A_dk_test[doc, k_old] -= 1
                        self.B_kw_test[k_old, word] -= 1

                        # M_k is the total word count for topic k from training
                        M_k = np.sum(self.B_kw, axis=1)

                        term_topic = self.A_dk_test[doc, :] + self.alpha
                        
                        # This term uses M_k and B_kw from the main training counts
                        term_word = (self.B_kw[:, word] + self.beta) / (M_k + self.n_words * self.beta)

                        probabilities = term_topic * term_word
                        probabilities /= np.sum(probabilities)

                        k_new = self.rand_gen.choice(self.topics_space, p=probabilities)

                        self.A_dk_test[doc, k_new] += 1
                        self.B_kw_test[k_new, word] += 1
                        
                        current_topics[i] = k_new

                    self.doc_word_samples_test[doc, word] = current_topics
        pass

    def update_loglike(self, iteration):
        """
        Updates loglike of the data, omitting the constant additive term
        with Gamma functions of hyperparameters
        """

        # Term 1 and 2: constant terms (we can omit?)
        log_prob_const_alpha = self.n_docs*(gammaln(self.n_topics*self.alpha) - self.n_topics*gammaln(self.alpha))
        log_prob_const_beta = self.n_topics*(gammaln(self.n_words*self.beta) - self.n_words*gammaln(self.beta))

        M_k = np.sum(self.B_kw, axis=1)
        N_d = np.sum(self.A_dk, axis=1)

        # Term 3 and 4
        term_3 = np.sum(np.sum(gammaln(self.B_kw + self.beta), axis = 1) - gammaln(M_k[:] + self.n_words * self.beta))

        term_4 = np.sum(np.sum(gammaln(self.A_dk + self.alpha), axis = 1) - gammaln(N_d[:] + self.n_topics * self.alpha))

        
        self.loglike[iteration] = (term_3 + term_4)
        
        if self.do_test:
            # Term 1 and 2: constant terms (we can omit?)
            log_prob_const_alpha = self.n_docs*(gammaln(self.n_topics*self.alpha) - self.n_topics*gammaln(self.alpha))
            log_prob_const_beta = self.n_topics*(gammaln(self.n_words*self.beta) - self.n_words*gammaln(self.beta))

            M_k = np.sum(self.B_kw_test , axis=1)
            N_d = np.sum(self.A_dk_test, axis=1)

            # Term 3 and 4
            term_3 = np.sum(np.sum(gammaln(self.B_kw_test + self.beta), axis = 1) - gammaln(M_k[:] + self.n_words * self.beta).T)

            term_4 = np.sum(np.sum(gammaln(self.A_dk_test + self.alpha), axis = 1) - gammaln(N_d[:] + self.n_topics * self.alpha).T)

            
            self.loglike_test[iteration] = (term_3 + term_4)
        pass

    def to_return_from_run(self):
        return self.doc_word_samples


def read_data(filename):
    """
    Reads the text data and splits into train/test.
    Examples:
    docs_words_train, docs_words_test = read_data('./code/toyexample.data')
    nips_train, nips_test = read_data('./code/nips.data')
    :param filename:    path to the file
    :return:
    docs_words_train:   training data, [n_docs, n_words] numpy array
    docs_words_test:    test data, [n_docs, n_words] numpy array
    """
    data = pd.read_csv(filename, dtype=int, sep=' ', names=['doc', 'word', 'train', 'test'])

    n_docs = np.amax(data.loc[:, 'doc'])
    n_words = np.amax(data.loc[:, 'word'])

    docs_words_train = np.zeros((n_docs, n_words), dtype=int)
    docs_words_test = np.zeros((n_docs, n_words), dtype=int)

    docs_words_train[data.loc[:, 'doc'] - 1, data.loc[:, 'word'] - 1] = data.loc[:, 'train']
    docs_words_test[data.loc[:, 'doc'] - 1, data.loc[:, 'word'] - 1] = data.loc[:, 'test']

    return docs_words_train, docs_words_test


def main():

    docs_words_train, docs_words_test = read_data('./toyexample.data')
    n_docs, n_words = docs_words_train.shape
    n_topics = 2
    alpha = 1
    beta = 1
    random_seed = 0
    n_iter = 500
    burn_in = 25  # Based on inspection
    max_lag = 50


    print('Running toyexample.data with the standard sampler')

    sampler = GibbsSampler(n_docs=n_docs, n_topics=n_topics, n_words=n_words,
                           alpha=alpha, beta=beta, random_seed=random_seed)

    topic_doc_words_distr, theta, phi = sampler.run(docs_words_train, docs_words_test,
                                                    n_iter=n_iter, save_loglike=True)

    print(phi * [phi > 1e-2])

    like_train, like_test = sampler.get_loglike()

    plt.subplots(figsize=(15, 8))
    plt.plot(like_train, label='train')
    plt.plot(like_test, label='test')
    plt.title('Joint (for train) and Predictive (for test) probabilities Standard Gibbs Sampler')
    plt.ylabel('loglike')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

    # Plotting autocorrelation

    post_burn_in_loglikelihood_train = like_train[burn_in:]
    post_burn_in_loglikelihood_test = like_test[burn_in:]
    loglike_train_series = pd.Series(post_burn_in_loglikelihood_train)
    loglike_test_series = pd.Series(post_burn_in_loglikelihood_test)
    autocorrelations_train = [loglike_train_series.autocorr(lag=k) for k in range(1, max_lag + 1)]
    autocorrelations_test = [loglike_test_series.autocorr(lag=k) for k in range(1, max_lag + 1)]
    
    plt.subplots(figsize=(15, 8))
    plt.plot(range(1, max_lag + 1), autocorrelations_train, marker='o', linestyle='-', label='train')
    plt.plot(range(1, max_lag + 1), autocorrelations_test, marker='o', linestyle='-', label='test')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Autocorrelation Function for Standard Gibbs Sampler Log Likelihood')
    plt.xlabel(f'Lag (Iterations, after burn-in={burn_in})')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.show()

    print('Running toyexample.data with the collapsed sampler')

    sampler_collapsed = GibbsSamplerCollapsed(n_docs=n_docs, n_topics=n_topics, n_words=n_words,
                                              alpha=alpha, beta=beta, random_seed=random_seed)

    doc_word_samples = sampler_collapsed.run(docs_words_train, docs_words_test,
                                             n_iter=n_iter, save_loglike=True)
    topic_counts = np.zeros((n_topics, 6))
    for doc in range(doc_word_samples.shape[0]):
        for word in range(doc_word_samples.shape[1]):
            for topic in doc_word_samples[doc, word]:
                topic_counts[topic, word] += 1

    print(topic_counts)

    like_train, like_test = sampler_collapsed.get_loglike()

    plt.subplots(figsize=(15, 8))
    plt.plot(like_train, label='train')
    plt.plot(like_test, label='test')
    plt.title('Joint (for train) and Predictive (for test) probabilities for Collapsed Gibbs Sampler')
    plt.ylabel('loglike')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

    # Plotting autocorrelation

    post_burn_in_loglikelihood_train = like_train[burn_in:]
    post_burn_in_loglikelihood_test = like_test[burn_in:]
    loglike_train_series = pd.Series(post_burn_in_loglikelihood_train)
    loglike_test_series = pd.Series(post_burn_in_loglikelihood_test)
    autocorrelations_train = [loglike_train_series.autocorr(lag=k) for k in range(1, max_lag + 1)]
    autocorrelations_test = [loglike_test_series.autocorr(lag=k) for k in range(1, max_lag + 1)]
    
    plt.subplots(figsize=(15, 8))
    plt.plot(range(1, max_lag + 1), autocorrelations_train, marker='o', linestyle='-', label='train')
    plt.plot(range(1, max_lag + 1), autocorrelations_test, marker='o', linestyle='-', label='test')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Autocorrelation Function for Collapsed Gibbs Sampler Log Likelihood')
    plt.xlabel(f'Lag (Iterations, after burn-in={burn_in})')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
