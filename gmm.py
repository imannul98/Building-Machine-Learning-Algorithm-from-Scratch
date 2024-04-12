import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = True  # Set False if the covariance matrix is a diagonal matrix


class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error.
        """
        #logit = np.where(logit == -np.inf, -np.finfo(np.float32).max, logit)
        logit = logit - np.max(logit, axis=1, keepdims=True)
        exp_logit=np.exp(logit)
        exp_logit_sum = np.sum(exp_logit, axis=1, keepdims=True)
        softmax=np.nan_to_num(exp_logit/exp_logit_sum)
        return softmax

    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        max_logit=np.max(logit, axis=1, keepdims=True)
        logit=logit-max_logit
        s=np.log(np.sum(np.exp(logit),axis=1,keepdims=True))+max_logit
        return s

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """

        raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. Note the value in self.D may be outdated and not correspond to the current dataset.
            3. You may wanna check if the matrix is singular before implementing calculation process.
        """
        det_sigma=np.linalg.det(sigma_i)
        inv_sigma=np.linalg.inv(sigma_i)
        try:
            det_sigma==0
        except:
            print("matrix is singular")
        inside_bracket=np.sum(np.dot(-1/2*(points-mu_i),inv_sigma).T*(points-mu_i).T,axis=0)
        #print(inside_bracket)
        outside_bracket=1/((2*np.pi)**(points.shape[1]/2)) *det_sigma**(-1/2)
        #print(outside_bracket)
        N=outside_bracket*np.exp(inside_bracket)
        #print(np.exp(np.sum(np.dot(-1/2*(points-mu_i),inv_sigma).T*(points-mu_i).T,axis=0)))
        return N
        
        
        
        


    def create_pi(self):
        """
        Initialize the prior probabilities
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        self.pi=np.full(self.K,1/self.K)
        return self.pi

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        indices = np.random.choice(self.N, self.K, replace=True)
        self.mu = self.points[indices]
        return self.mu

    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        self.sigma = np.array([np.eye(self.D) for _ in range(self.K)])
        return self.sigma

    def _init_components(self, **kwargs):
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5)  # Do Not Remove Seed
        pi=self.create_pi()
        mu=self.create_mu()
        sigma=self.create_sigma()
        return pi,mu,sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        # === graduate implementation
        # if full_matrix is True:
        # ...

        # === undergraduate implementation
        # if full_matrix is False:
        # ...
        ll=np.zeros((self.N,self.K))
        for j in range(self.K):
            if full_matrix:
                ll[:, j] = np.log(pi[j]+1e-32) + np.log(self.multinormalPDF(self.points, mu[j],sigma[j])+1e-32)
            else:
                ll[:, j] = np.log(pi[j]+1e-32) + np.log(self.multinormalPDF(self.points, mu[j], np.diag(sigma[j]))+1e-32)
        return ll

    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        # if full_matrix is True:
        # ...

        # === undergraduate implementation
        # if full_matrix is False:
        # ...
        gamma=np.zeros((self.N,self.K))
        if full_matrix is True:
            gamma=self.softmax(self._ll_joint(pi,mu,sigma,full_matrix=True))
        else:
            gamma=self.softmax(self._ll_joint(pi,mu,sigma,full_matrix=False))
        return gamma

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        # if full_matrix is True:
        # ...

        # === undergraduate implementation
        # if full_matrix is False:
        # ...
        Nk = np.sum(gamma, axis=0)
        mu = np.sum(gamma[:, :, None] * self.points[:, None, :], axis=0) / Nk[:, None]
        sigma=np.zeros((mu.shape[0], self.points.shape[1], self.points.shape[1]))
        for k in range(mu.shape[0]):
            inner_bracket=np.dot((gamma[:, k].reshape(-1, 1) * (self.points - mu[k])).T, (self.points - mu[k]))
            sigma[k] = inner_bracket/Nk[k]

        if not full_matrix:
            sigma = np.diag(np.diag(sigma))
            
        pi = Nk / self.N
        return pi, mu, sigma


    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the parameters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)
            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description("iter %d, loss: %.4f" % (it, loss))
        return gamma, (pi, mu, sigma)
