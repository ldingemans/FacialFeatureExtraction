import pymc3 as pm
import pandas as pd
import numpy as np
import theano as tt
import arviz as az

def logit_normal_CASS(prior_inclusion_prob, sigma_v, p, K):
    """
    Logit-normal continuous analogue of the spike-and-slab: https://doi.org/10.1098/rsif.2018.0572    
    Uses a logit-normal instead of the beta prior which is common in continuous S&S approximations, as those can be
    obtained via transformations of Gaussians and that makes sampling way more efficient.
    
    Parameters
    ----------
    prior_inclusion_prob: float
        Prior inclusion probabilities for the features
    sigma_v: int
        SD for logit normal distribution, higher values push the regression coefficents more to 0 and 1
    p: int
        Number of features
    K: int
        Number of different classes
   
    Returns
    -------
    beta_lncass: pymc3 distribution
        The calculated LN-CASS regression coefficient(s)
    """
    mu_v = pm.math.logit(prior_inclusion_prob)
    tau = pm.HalfNormal('tau', 2)
    if K == 2:    
        lamb_raw = pm.Normal('lambda', mu=mu_v, sd=sigma_v, shape=p)
        lamb_logit = pm.Deterministic('lamb_logit', pm.math.invlogit(lamb_raw))
        beta_raw = pm.Normal('beta_raw', mu=0, sd=1, shape=p)
        beta_lncass = pm.Deterministic('beta_lncass', beta_raw * pm.math.sqr(lamb_logit * tau))
    else:
        lamb_raw = pm.Normal('lambda', mu=mu_v, sd=sigma_v, shape=(p, 1))
        lamb_logit = pm.Deterministic('lamb_logit', pm.math.invlogit(lamb_raw))
        beta_raw = pm.Normal('beta_raw', mu=0, sd=1, shape=(p, K-1))
        beta_raw_stacked = pm.Deterministic('beta_raw_stacked', tt.tensor.horizontal_stack(beta_raw, tt.tensor.zeros(shape=(p, 1))))
        beta_lncass = pm.Deterministic('beta_lncass', beta_raw_stacked * pm.math.sqr(tt.tensor.repeat(lamb_logit, K, axis=1) * tau))
    return beta_lncass


def bayes_logistic_reg(X_train, y_train, X_test, advi=False, prior_inclusion_prob=0.1, N_CORES=8, tune_steps=2000, target_accept=0.8, LN_CASS=True):
    """
    Building the Bayesian logistic regression model using PyMC3
    Parameters
    ----------
    X_train : numpy array
        The training data
    y_train: numpy array
        The training labels
    X_test: numpy array
        The validation/testing data
    advi: boolean
        Whether to use variational inference. When false, use MCMC (NUTS)
    prior_inclusion_prob: float
        The prior inclusion probability
    N_CORES: int
        Number of cores to use when sampling
    tune_steps: int
        Number of tuning steps during sampling
    target_accept: float
        Target accept during sampling
    LN_CASS: boolean
        Whether to use the LN-CASS as prior for the regression coefficients
    
    Returns
    -------
    predictions: numpy array
        The predictions on the validation/test data
    predicted_classes: numpy array
        The predicted classes on the validation/test data
    trace: pymc3 trace
        The trace after sampling using pymc3
    summ_trace:  panda dataframe
        Summary of trace data
    """
    no_of_classes = len(np.unique(y_train))
    with pm.Model() as model: 
      if LN_CASS == True:  
          beta = logit_normal_CASS(prior_inclusion_prob, 10, X_train.shape[1], no_of_classes)
      elif LN_CASS == False:
          beta = pm.Normal('beta', 0, 2, shape=X_train.shape[1])

      X_shared = pm.Data('X_shared', X_train)
      if no_of_classes == 2:
          alpha = pm.Normal('alpha', mu=0, sd=3) 
          equations = alpha + pm.math.dot(X_shared, beta)
          θ = pm.Deterministic('θ', pm.invlogit(equations))      
          y_model = pm.Bernoulli('y_model', p=θ, observed=y_train) 
      else:
          equations = pm.math.dot(X_shared, beta)
          probabilities = pm.Deterministic('probabilities', tt.tensor.nnet.softmax(equations))
          y_model = pm.Categorical('y_model', p=probabilities, observed=y_train)
          
      if advi == False:
          trace = pm.sample(tune=tune_steps, target_accept=target_accept, chains=2, max_treedepth=15, cores=N_CORES)
      else:
          approx = pm.fit(500000, method='fullrank_advi')
          trace = approx.sample(draws=10000)
          
      summ_trace = az.summary(trace)
      if no_of_classes == 2:
          X_shared.set_value(X_test)
          ppc = pm.sample_posterior_predictive(trace, samples=5000, model=model, var_names=['y_model', 'θ'])
          predictions = ppc['θ'].mean(axis=0)
          predicted_classes = np.array(predictions > 0.5, dtype=int)
      else:
          X_shared.set_value(X_test)
          ppc = pm.sample_posterior_predictive(trace, samples=5000, model=model, var_names=['y_model', 'probabilities'])
          predictions = ppc['probabilities'].mean(axis=0)
          predicted_classes = np.argmax(np.exp(predictions).T / np.sum(np.exp(predictions), axis=1), axis=0)
          
      return predictions, predicted_classes, trace, summ_trace