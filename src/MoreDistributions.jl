module MoreDistributions

using Distributions, LinearAlgebra, LazyArrays, SpecialFunctions, StatsBase,
      Random, NLsolve, UnPack, HypergeometricFunctions
using Distributions: @quantile_newton, quantile_newton, quantile_bisect

# functions to extend
import Base: convert, rand, maximum, minimum, extrema
import Statistics: mean, median, quantile, std, var, cov, cor
import StatsBase: kurtosis, skewness, entropy, mode, modes,
                  fit, kldivergence, loglikelihood, dof, span,
                  params, params!

import Distributions:
    @check_args,
    @distr_support,
    partype,
    location,
    scale,
    shape,
    pdf,
    logpdf,
    cdf,
    fit_mle,
    sampler,
    isleptokurtic,
    isplatykurtic,
    ismesokurtic

export
    GeneralizedNormal, GeneralizedNormalSampler,
    BodyTailGeneralizedNormal,

    mean, median, quantile, std, var, cov, cor,

    canonform,          # get canonical form of a distribution
    ccdf,               # complementary cdf, i.e. 1 - cdf
    cdf,                # cumulative distribution function
    cf,                 # characteristic function
    cquantile,          # complementary quantile (i.e. using prob in right hand tail)
    component,          # get the k-th component of a mixture model
    components,         # get components from a mixture model
    componentwise_pdf,      # component-wise pdf for mixture models
    componentwise_logpdf,   # component-wise logpdf for mixture models
    concentration,      # the concentration parameter
    convolve,           # convolve distributions of the same type
    dim,                # sample dimension of multivariate distribution
    dof,                # get the degree of freedom
    entropy,            # entropy of distribution in nats
    failprob,           # failing probability
    fit,                # fit a distribution to data (using default method)
    fit_mle,            # fit a distribution to data using MLE
    insupport,          # predicate, is x in the support of the distribution?
    invcov,             # get the inversed covariance
    invlogccdf,         # complementary quantile based on log probability
    invlogcdf,          # quantile based on log probability
    isplatykurtic,      # Is excess kurtosis > 0.0?
    isleptokurtic,      # Is excess kurtosis < 0.0?
    ismesokurtic,       # Is excess kurtosis = 0.0?
    isprobvec,          # Is a probability vector?
    isupperbounded,
    islowerbounded,
    isbounded,
    hasfinitesupport,
    kurtosis,           # kurtosis of the distribution
    logccdf,            # ccdf returning log-probability
    logcdf,             # cdf returning log-probability
    logdiffcdf,         # log of difference between cdf at two values
    logdetcov,          # log-determinant of covariance
    loglikelihood,      # log probability of array of IID draws
    logpdf,             # log probability density
    logpdf!,            # evaluate log pdf to provided storage

    invscale,           # Inverse scale parameter
    sqmahal,            # squared Mahalanobis distance to Gaussian center
    sqmahal!,           # inplace evaluation of sqmahal
    location,           # get the location parameter
    location!,          # provide storage for the location parameter (used in multivariate distribution mvlognormal)
    mean,               # mean of distribution
    meandir,            # mean direction (of a spherical distribution)
    meanform,           # convert a normal distribution from canonical form to mean form
    meanlogx,           # the mean of log(x)
    median,             # median of distribution
    mgf,                # moment generating function
    mode,               # the mode of a unimodal distribution
    modes,              # mode(s) of distribution as vector
    moment,             # moments of distribution
    nsamples,           # get the number of samples contained in an array
    ncategories,        # the number of categories in a Categorical distribution
    ncomponents,        # the number of components in a mixture model
    ntrials,            # the number of trials being performed in the experiment
    params,             # get the tuple of parameters
    params!,            # provide storage space to calculate the tuple of parameters for a multivariate distribution like mvlognormal
    partype,            # returns a type large enough to hold all of a distribution's parameters' element types
    pdf,                # probability density function (ContinuousDistribution)
    probs,              # Get the vector of probabilities
    probval,            # The pdf/pmf value for a uniform distribution
    product_distribution, # product of univariate distributions
    quantile,           # inverse of cdf (defined for p in (0,1))
    qqbuild,            # build a paired quantiles data structure for qqplots
    rate,               # get the rate parameter
    sampler,            # create a Sampler object for efficient samples
    scale,              # get the scale parameter
    scale!,             # provide storage for the scale parameter (used in multivariate distribution mvlognormal)
    shape,              # get the shape parameter
    skewness,           # skewness of the distribution
    span,               # the span of the support, e.g. maximum(d) - minimum(d)
    std,                # standard deviation of distribution
    stdlogx,            # standard deviation of log(x)
    suffstats,          # compute sufficient statistics
    succprob,           # the success probability
    support,            # the support of a distribution (or a distribution type)
    truncated,          # truncate a distribution with a lower and upper bound
    var,                # variance of distribution
    varlogx,            # variance of log(x)
    expected_logdet,    # expected logarithm of random matrix determinant
    gradlogpdf          # gradient (or derivative) of logpdf(d,x) wrt x

include("generalizednormal.jl")
include("bodytail.jl")

end # module
