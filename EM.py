from __future__ import division
import collections
import numpy as np
from numpy import random

class MarkovException(Exception):
    """Exception raised when there is a problem with a markov chain
    """
    pass

class MarkovChain:
    """Markov Chain modeling

    Args:
        pi (list) : A vector containing the probability for each start to be the initial state
        transition : The matrix holding the transition probabilities

    Attributes:
            Each attribute starting with an underscore (_)
            is semi private (usable but not recommended)
            Each attribute starting with a double underscore (__)
            are meant to be private and not callable from outside the class.

            pi (list) : A vector containing the probability for each start to be the initial state
            transition (np.array) : The matrix holding the transition probabilities

            _cannonical (np.array) : Holds the cannonical representation of the transition matrix
            _Q (np.array) : Holds the submatrix for transient states (t * t, with t size of transient states)
            _R (np.array) : Holds the submatrix for absorbant states (t * r, with r size of absorbant states)
            _N (np.array) : Holds the fundamental matrix of the markov chain
            _B (np.array) : Holds the probability for each transient state to end up in each absorbant states

            __index_mapping (dictionary) : Map every index to their new index for _B.
    """
    def __init__(self, pi, transition):
        self.pi = np.copy(pi)
        self.transition = np.copy(transition)
        self.__index_mapping = {}
        self._cannonical, self._Q, self._R = self._compute_cannonical()
        self._N = self._compute_fondamental()
        self._B = np.dot(self._N, self._R)
        if( not self.__check_markovian()):
            raise MarkovException("The tansition matrix does not correspond to a markov chain")

    @staticmethod
    def check_markovian(transition):
        """Checks wether a given matrix is a markov transition matrix or not.
        It checks if the matrix is square and if the sum of each row equals 1.

        Args:
            transition (np.array) : The matrix to test

        Returns:
            (boolean) : True if the matrix is a markov transition matrix, False otherwise.
        """
        #If the matrix is not square then it's not a markov chain.
        if(transition.shape[0] != transition.shape[1]):
            return False
        for i, row in enumerate(transition):
            total = sum(row)
            #Using absolute difference with 10e-05 error to check if total == 1
            #Solve float arithmetic error
            if(abs(1-total) > 0.00001):
                return False
        return True

    def __check_markovian(self):
        """Check if the matrix of this markov chain is correct.

        Args:
              None

        Returns:
              (boolean) : True if the matrix is a markov transition matrix, False otherwise.
        """
        return MarkovChain.check_markovian(self.transition)

    def find_absorbant_state(self):
        """Return a list containing the index of every absorbant states

        Args:
            None

        Returns:
            indexes (list) : list containing the index of every absorbant states
        """
        diag = np.diagonal(self.transition)
        indexes = np.where(diag == 1)
        return indexes[0]

    def _compute_cannonical(self):
        """Computes the cannonical form of the transition matrix
        and extracts useful submatrices of this cannonical form

        Args:
             None

        Returns:
              cannonical (np.array) : The cannonical matrix of the transition matrix
              Q (np.array) : The t*t upper left submatrix.
              R (np.array) : The t*r upper right submatrix
        """
        #Get absorbant state indexes
        absorbants_indexes = self.find_absorbant_state()
        #Deduce transient indexes by taking every other index
        transient_indexes = [x for x in range(len(self.transition)) if x not in absorbants_indexes]
        r = len(absorbants_indexes)
        t = len(transient_indexes)

        #Create a mapping for indexes (since columns and rows will be interchanged)
        for i, item in enumerate(absorbants_indexes):
            self.__index_mapping[item] = i
        for i, item in enumerate(transient_indexes):
            self.__index_mapping[item] = i

        #If absorbant states are the last indexes, then the matrix is already cannonical
        last = range(len(self.transition))[-len(absorbants_indexes):]
        if(sorted(last) == sorted(absorbants_indexes)):
            cannonical = np.copy(self.transition)
        else:
            #Copy columns for readability
            absorbants_col = self.transition[:, absorbants_indexes]
            transient_col = self.transition[:, transient_indexes]
            #Reconcatenate (execute the swap)
            cannonical = np.concatenate((transient_col, absorbants_col), axis=1)
            #Without temp var :
            #self.transition = np.concatenate((self.transition[:, transient_indexes], self.transition[:, absorbants_indexes]), axis=1)

            #Same for rows
            absorbants_row = cannonical[absorbants_indexes]
            transient_row = cannonical[transient_indexes]
            cannonical = np.concatenate((transient_row, absorbants_row), axis=0)
            #Without temp var :
            #self.transition = np.concatenate((elf.transition[transient_indexes], self.transition[absorbants_indexes]), axis=0)
        Q = cannonical[np.ix_(range(t), range(t))]
        new_absor_index = [x for x in range(t+r) if x not in range(t)]
        R = cannonical[np.ix_(range(t), new_absor_index)]
        return cannonical, Q, R

    def _compute_fondamental(self):
        """Computes the fundamental matrix corresponding to the cannonical form
        Requires the cannonical form to be computed beforehand (see self._compute_cannonical())
        Its computed by taking the submatrix Q and performing these operations :
        N = (I-Q)^-1

        Args:
            None

        Returns:
            N (np.array) : The fundamental matrix computed
        """
        I = np.identity(self._Q.shape[1])
        N = np.linalg.inv(I-self._Q)
        return N

    def absorbing_probability(self, current_state, reaching_state):
        """Computes the probability to end in a given absorbant state based on the current state.

        Args:
            current_state (int) : The index of the current state
            reaching_state (int) : The index of the absorbant state where we want to end up

        Returns:
            (float) : probability to reach 'reaching_state' from 'current_state'

        """
        if(current_state in self.find_absorbant_state()):
            return 0
        i = self.__index_mapping[current_state]
        j = self.__index_mapping[reaching_state]
        return self._B[i][j]

class ExpectationMaximisation:
    """EM Algorithm applied to Markov Chain.

    Args:
        sequences (list): The sequences to clusterize.
        states (list): The exhaustive list of states.
        nb_clusters (int): he number of cluster to create.

    Attributes:
        Each attribute starting with an underscore (_)
        is semi private (usable but not recommended)
        Each attribute starting with a double underscore (__)
        are meant to be private and not callable from outside the class.

        clusters (list): A list holding the different clusters, with their associated
                  markov chain.
        weighted_compatibility (np.array): A matrix (size m*k) holding the probability
                                for each sequence to be in each cluster.

        _sequences (list): The sequences to clusterize.
        _states (list): The exhaustive list of states.
        _weights (np.array): The probability to be in each cluster, P(c) for 0<=c<=k.

        __m (int): The number of sequences used.
        __k (int): The number of cluster to create.
    """
    def __init__(self, sequences, states, nb_clusters):
        self._sequences = sequences
        self.__m = len(sequences)
        self._states = states
        self.__k = nb_clusters
        self._weights = np.zeros(self.__k)
        self.weighted_compatibility = np.zeros(shape=(self.__m, self.__k))
        self.clusters = []
        self.__initialize()

    def __initialize(self):
        """Randomly associate sequence to a cluster.

        Args:
            None

        Retuns:
            None
        """
        #Variable aliases to improve readability
        m = self.__m
        k = self.__k
        for i, s in enumerate(self._sequences):
            #Random association to a cluster
            c = random.randint(0, k)
            #The probability is certain so its 1
            self.weighted_compatibility[i][c] = 1
        for c in range(k):
            #@todo fix empty cluster exception
            if(sum(self.weighted_compatibility[:,c]) == 0):
                raise MarkovException("Empty cluster")

            #pi is the probability for each state to start on this markov chain
            #a is the matrix holding the probabilities for each state to go to each other state
            markov = self._compute_markov_chain(c)
            #pi, a = compute_markov_chain(sequences, weighted_compatibility, states, c)
            #Weights is the probability to be in cluster c, P(c)
            self._weights[c] = sum(self.weighted_compatibility[:,c])/m
            self.clusters.append(markov)

        #return weighted_compatibility, clusters, weights

    def _compute_markov_chain(self, c):
        """Compute a markov chain for a specific cluster

        Args:
            c (int): The index of the cluster for which the markov chain
                     will be computed.

        Returns:
            (phi): A markov chain (containing the initial states pi and the matrix a).
        """
        #Initialize values
        pi = []
        n = len(self._states)
        a = np.zeros(shape=(n, n))

        #We first calculate pi_c(state) with c beeing the cluster and state, well the state..
        #For this, we compute two terms.
        #The first one is the sum of the probability of sequence Si to belong to cluster c, knowing Si and phi :
        #sum(P(ci = c|Si, phi))
        #With ci being the cluster of the sequence i, Si the sequence i, and phi the characteristiques of the cluster
        #(in this case phi is the markov chain)
        #P(ci = c | Si, phi) is computed in another step, and is holded in weighted.
        #P(ci = c | Si, phi) is actualy weighted[i][c]

        #The second term is the same sum, but we only keep terms where the sequence start with a certain state :
        #sum( P(ci = c | Si, phi) * delta(state, initial_state) )
        #where P(ci = c | Si, phi) is the same as above and
        #delta(state, initial_state) = state == initial_state ? 1 : 0
        #(Kronecker delta)

        #With this two terms, pi_c(state) = term1 / term2
        for t, state in enumerate(self._states):
            total_ponderated = 0

            #R state holds the number of transition from the current state
            r_state = []
            for i, s in enumerate(self._sequences):
                if(s[0] == state):
                    total_ponderated += self.weighted_compatibility[i][c]
                #We don't count the last element for transitions
                nb_transitions = s[:-1].count(state)
                r_state.append(nb_transitions)

            total = sum(self.weighted_compatibility[:,c])

            #We add the newly calculated pi to the array
            pi.append(total_ponderated/total)

            for _, state_prime in enumerate(self._states):

                #R state prime holds the number of transition from current state to state_prime
                r_state_prime = []
                for i, s in enumerate(self._sequences):
                    nb_transitions = 0
                    for j in range(len(s)-1):
                        if(s[j] == state and s[j+1] == state_prime):
                            nb_transitions += 1
                    r_state_prime.append(nb_transitions)

                #To calculate the transition probability from one state to another, we must calculate two terms first.
                #The first term is the sum of the probability of the sequence i beeing in cluster c, knowing the sequence and phi, ponderated with the number of transitions :
                #sum( P(ci = c | Si, phi) * r_state_prime )
                #The second one is the same sum but with a different ponderation :
                #sum (P(ci = c | Si, phi) * r_state)
                dividend = sum(self.weighted_compatibility[:, c]*r_state_prime)
                divisor = sum(self.weighted_compatibility[:, c]*r_state)

                #Handling ending point case
                if(divisor == 0):
                    #We set the diagonal at 1
                    a[state][state] = 1
                else:
                    a[state][state_prime] = dividend/divisor
        return MarkovChain(pi, a)

    def _compute_compatibilities(self, sequence):
        """Calculate the compatibility between a given sequence and each cluster.

        Args:
            sequence (list): A list of different states (each state is an int).

        Returns:
            compatibility (np.array): An array holding the compatibility for each cluster.

        """
        #Aliases for readability
        k = self.__k
        initial_state = 0
        compatibilities = np.zeros(k)
        for c, cluster in enumerate(self.clusters):
            proba = 1
            for state in range(len(sequence)-1):
                proba *= cluster.transition[sequence[state]][sequence[state+1]]
            #Proba now holds the probability of sequence i to have been generated by markov chain of cluster c
            compatibilities[c] = cluster.pi[sequence[initial_state]]*proba
            #compatibility = P(Si | ci = c, Phi) = P(Si|Phi) = pi(ei,1)* Product(ac(ei,t-1; ei,t))
            #With ac(ei,t-1; ei,t) beeing the path taken in the markov chain.
        #We multiply by P(c) for each compatibility.
        compatibilities = compatibilities * self._weights
        return compatibilities

    def _compute_sequence_in_cluster_probabilities(self, sequence):
        """Give the probabilities for a given sequence to belong to each cluster.

        Args:
            The sequence to clusterize

        Returns:
            (list) : List containing the probabilities for the sequence to be in each cluster
        """
        compatibilities = self._compute_compatibilities(sequence)
        #print(compatibilities)
        compatibilities /= sum(compatibilities)

        return compatibilities

    def _expectation(self):
        """The expectation part of EM algorithm.

        Args:
            None

        Returns:
            weighted_compatibility (np.array): The new matrix holding the probailities
                                            for each sequence to belong in each cluster.
        """
        #Aliases for readability
        m = self.__m
        k = self.__k
        #Temporal matrix used because delta is needed (difference with old one).
        weighted_compatibility = np.zeros(shape=(m, k))
        for i, s in enumerate(self._sequences):
            weighted_compatibility[i] = self._compute_sequence_in_cluster_probabilities(s)

        return weighted_compatibility

    def _maximisation(self):
        """The maximisation part of EM algorithm.

        Args:
            None

        Returns:
            None
        """
        #Aliases for readability
        m = self.__m
        #Temp variable
        clusters = []
        for c, cluster in enumerate(self.clusters):
            #We recompute P(c)
            self._weights[c] = (1/m)*sum(self.weighted_compatibility[:,c])
            #Recompute markov chain and add to the list
            clusters.append(self._compute_markov_chain(c))
        #@todo change current clusters instead of creating new one
        self.clusters = clusters

    def fit(self):
        """Clusterize the sequence passed at init with the number of cluster specified.

        Args:
            None

        Returns:
            None
        """
        #weighted_compatibility, clusters, weights = initialize_EM(sequences, states, k)
        delta = 1
        #We loop over until the difference between the probabilities varies a little (10e-4 error)
        while delta > 0.0001:
            new_weighted_compatibility = self._expectation()
            #Change the delta
            delta = np.mean(abs(new_weighted_compatibility - self.weighted_compatibility))
            print(delta )
            self.weighted_compatibility = new_weighted_compatibility
            self._maximisation()

        #Return not necessary, to see
        #return self.clusters, self.weighted_compatibility

    def predict_proba_hard(self, sequence, final_state):
        """Give the proba for a sequence to end in final state
        The sequence is categorized in the cluster where the probability of belonging is the highest

        Args:
            sequence (list) : A list of states to clusterize and predict
            final_state (int) : The reaching state wanted

        Returns:
            (float) : The probability for sequence to end in final state
        """
        probas = self._compute_sequence_in_cluster_probabilities(sequence)
        c = np.where(probas == max(probas))[0]
        #@todo add treshold for selecting probas
        cluster = self.clusters[c]
        return cluster.absorbing_probability(sequence[-1], final_state)

    def predict_proba_soft(self, sequence, final_state):
        """Give the proba for a sequence to end in final state
        The probas is ponderated by the probability of belonging to each cluster

        Args:
            sequence (list) : A list of states to clusterize and predict
            final_state (int) : The reaching state wanted

        Returns:
            (float) : The probability for sequence to end in final state)
        """
        probas = self._compute_sequence_in_cluster_probabilities(sequence)
        abs_proba = 0
        for c, cluster in enumerate(self.clusters):
            abs_proba += (probas[c]*cluster.absorbing_probability(sequence[-1], final_state))
        return abs_proba

    def __str__(self):
        """Overiding the __str__ magic method to display the clusters and their distribution properly

        Args:
            None

        Returns:
            A string representation of the EM object
        """
        value = "============================\nWeighted probability matrix : \n"
        temp = ""
        for i, row in enumerate(self.weighted_compatibility):
            temp = "\t[ "
            for j, col in enumerate(row):
                temp = "".join([temp, "{:.2f}  ".format(col)])
            temp = "".join([temp, "]\t"])
            temp = "".join([temp, "\t{0}".format(self._sequences[i])])
            temp = "".join([temp, "\n"])
            value = "".join([value, temp])
        value = "".join([value, "============================\n"])
        value = "".join([value, "\n============================\nClusters : \n"])
        for c, cluster in enumerate(self.clusters):
            temp = "\tCluster {0} : \n".format(c)
            temp = "".join([temp, "\t\t Transition Matrix : \n"])
            for i, row in enumerate(cluster.transition):
                temp = "".join([temp, "\t\t[ "])
                for j, col in enumerate(row):
                    temp = "".join([temp, "{:.2f} ".format(col)])
                temp = "".join([temp, "]\n"])
            temp = "".join([temp, "\n\t\t Absorbing Probability : \n"])
            for i, row in enumerate(cluster._B):
                temp = "".join([temp, "\t\t[ "])
                for j, col in enumerate(row):
                    #temp = "".join([temp, "\t\t[ "])
                    temp = "".join([temp, "{:.2f} ".format(col)])
                temp = "".join([temp, "]\n"])
            value = "".join([value, temp])
        value = "".join([value, "============================\n"])
        return value
