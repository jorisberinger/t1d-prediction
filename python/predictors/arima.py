import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMAResults

from PredictionWindow import PredictionWindow
from predictors.predictor import Predictor
path = os.getenv('T1DPATH', '../')
logger = logging.getLogger(__name__)
sampleTime = 1


class Arima(Predictor):
    name: str = "Arima Predictor"
    pw: PredictionWindow
    prediction_values: [float]

    def __init__(self, pw):
        super().__init__()
        logger.info("init Arima")
        self.pw: PredictionWindow = pw
        self.sample_time = sampleTime
        self.window = window = self.pw.data.iloc[::sampleTime]['cgmValue']
        self.index_train = pd.timedelta_range(start = '0 hour',
                                              end = '{} hours'.format(self.pw.userData.train_length() / 60),
                                              freq = str(sampleTime) + 'min')
        self.index_test = pd.timedelta_range(start = '{} minutes'.format(self.pw.userData.train_length() + self.sample_time),
                                             end = '{} hours'.format(self.pw.userData.simlength),
                                             freq = str(sampleTime) + 'min')

    def calc_predictions(self, error_times: [int]) -> bool:
        logger.info("get Errors at {}".format(error_times))

        train = self.window[0:(self.pw.userData.simlength * 60 - self.pw.userData.predictionlength)]
        train.index = pd.to_datetime(self.index_train)

        test = self.window[(self.pw.userData.train_length() + self.sample_time):(self.pw.userData.simlength * 60)]
        test.index = pd.to_datetime(self.index_test)

        stepwise_fit = auto_arima(train, start_p = 1, start_q = 1, max_p = 10, max_q = 10, seasonal = False,
                                  trace = True, max_order = 100,
                                  error_action = 'warn', suppress_warnings = True, stepwise = True)

        if sum(stepwise_fit.order):
            preds, conf_int = stepwise_fit.predict(n_periods = len(test), return_conf_int = True)
            prediction = pd.Series(preds, index = test.index)
            index = np.arange(self.pw.userData.train_length() + sampleTime, self.pw.userData.simlength * 60 + 1,
                              self.sample_time)
            self.prediction_values_all = prediction
            self.prediction_values_all.index = index
            self.prediction_values = self.prediction_values_all.loc[error_times + self.pw.userData.train_length()]
            self.prediction_values = self.prediction_values.tolist()
            return True
        else:
            return False

    def get_graph(self) -> ({'label': str, 'values': [float]}):

        return {'label': self.name, 'values': self.prediction_values_all}






'''
Parameters
    ----------
    start_p : int, optional (default=2)
        The starting value of ``p``, the order (or number of time lags)
        of the auto-regressive ("AR") model. Must be a positive integer.
    max_p : int, optional (default=5)
        The maximum value of ``p``, inclusive. Must be a positive integer
        greater than or equal to ``start_p``.

    d : int, optional (default=None)
        The order of first-differencing. If None (by default), the value
        will automatically be selected based on the results of the ``test``
        (i.e., either the Kwiatkowski–Phillips–Schmidt–Shin, Augmented
        Dickey-Fuller or the Phillips–Perron test will be conducted to find
        the most probable value). Must be a positive integer or None. Note
        that if ``d`` is None, the runtime could be significantly longer.
    max_d : int, optional (default=2)
        The maximum value of ``d``, or the maximum number of non-seasonal
        differences. Must be a positive integer greater than or equal to ``d``.
        
    start_q : int, optional (default=2)
        The starting value of ``q``, the order of the moving-average
        ("MA") model. Must be a positive integer.
    max_q : int, optional (default=5)
        The maximum value of ``q``, inclusive. Must be a positive integer
        greater than ``start_q``.

    max_order : int, optional (default=10)
        If the sum of ``p`` and ``q`` is >= ``max_order``, a model will
        *not* be fit with those parameters, but will progress to the next
        combination. Default is 5. If ``max_order`` is None, it means there
        are no constraints on maximum order.


    seasonal : bool, optional (default=True)
        Whether to fit a seasonal ARIMA. Default is True. Note that if
        ``seasonal`` is True and ``m`` == 1, ``seasonal`` will be set to False.

    stationary : bool, optional (default=False)
        Whether the time-series is stationary and ``d`` should be set to zero.

    information_criterion : str, optional (default='aic')
        The information criterion used to select the best ARIMA model. One of
        ``pmdarima.arima.auto_arima.VALID_CRITERIA``, ('aic', 'bic', 'hqic',
        'oob').

    alpha : float, optional (default=0.05)
        Level of the test for testing significance.

    test : str, optional (default='kpss')
        Type of unit root test to use in order to detect stationarity if
        ``stationary`` is False and ``d`` is None. Default is 'kpss'
        (Kwiatkowski–Phillips–Schmidt–Shin).

    
    stepwise : bool, optional (default=True)
        Whether to use the stepwise algorithm outlined in Hyndman and Khandakar
        (2008) to identify the optimal model parameters. The stepwise algorithm
        can be significantly faster than fitting all (or a ``random`` subset
        of) hyper-parameter combinations and is less likely to over-fit
        the model.

    n_jobs : int, optional (default=1)
        The number of models to fit in parallel in the case of a grid search
        (``stepwise=False``). Default is 1, but -1 can be used to designate
        "as many as possible".


    method : str, one of {'css-mle','mle','css'}, optional (default=None)
        This is the loglikelihood to maximize.  If "css-mle", the
        conditional sum of squares likelihood is maximized and its values
        are used as starting values for the computation of the exact
        likelihood via the Kalman filter.  If "mle", the exact likelihood
        is maximized via the Kalman Filter.  If "css" the conditional sum
        of squares likelihood is maximized.  All three methods use
        `start_params` as starting parameters.  See above for more
        information. If fitting a seasonal ARIMA, the default is 'lbfgs'

    trend : str or None, optional (default=None)
        The trend parameter. If ``with_intercept`` is True, ``trend`` will be
        used. If ``with_intercept`` is False, the trend will be set to a no-
        intercept value.

    solver : str or None, optional (default='lbfgs')
        Solver to be used.  The default is 'lbfgs' (limited memory
        Broyden-Fletcher-Goldfarb-Shanno).  Other choices are 'bfgs',
        'newton' (Newton-Raphson), 'nm' (Nelder-Mead), 'cg' -
        (conjugate gradient), 'ncg' (non-conjugate gradient), and
        'powell'. By default, the limited memory BFGS uses m=12 to
        approximate the Hessian, projected gradient tolerance of 1e-8 and
        factr = 1e2. You can change these by using kwargs.

    maxiter : int, optional (default=50)
        The maximum number of function evaluations. Default is 50.


    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If
        ``suppress_warnings`` is True, all of the warnings coming from
        ``ARIMA`` will be squelched.

    error_action : str, optional (default='warn')
        If unable to fit an ``ARIMA`` due to stationarity issues, whether to
        warn ('warn'), raise the ``ValueError`` ('raise') or ignore ('ignore').
        Note that the default behavior is to warn, and fits that fail will be
        returned as None. This is the recommended behavior, as statsmodels
        ARIMA and SARIMAX models hit bugs periodically that can cause
        an otherwise healthy parameter combination to fail for reasons not
        related to pmdarima.

    trace : bool, optional (default=False)
        Whether to print status on the fits. Note that this can be
        very verbose...

    random : bool, optional (default=False)
        Similar to grid searches, ``auto_arima`` provides the capability to
        perform a "random search" over a hyper-parameter space. If ``random``
        is True, rather than perform an exhaustive search or ``stepwise``
        search, only ``n_fits`` ARIMA models will be fit (``stepwise`` must be
        False for this option to do anything).

    random_state : int, long or numpy ``RandomState``, optional (default=None)
        The PRNG for when ``random=True``. Ensures replicable testing and
        results.

    n_fits : int, optional (default=10)
        If ``random`` is True and a "random search" is going to be performed,
        ``n_iter`` is the number of ARIMA models to be fit.

    return_valid_fits : bool, optional (default=False)
        If True, will return all valid ARIMA fits in a list. If False (by
        default), will only return the best fit.

    out_of_sample_size : int, optional (default=0)
        The ``ARIMA`` class can fit only a portion of the data if specified,
        in order to retain an "out of bag" sample score. This is the
        number of examples from the tail of the time series to hold out
        and use as validation examples. The model will not be fit on these
        samples, but the observations will be added into the model's ``endog``
        and ``exog`` arrays so that future forecast values originate from the
        end of the endogenous vector.

        For instance::

            y = [0, 1, 2, 3, 4, 5, 6]
            out_of_sample_size = 2

            > Fit on: [0, 1, 2, 3, 4]
            > Score on: [5, 6]
            > Append [5, 6] to end of self.arima_res_.data.endog values

'''
