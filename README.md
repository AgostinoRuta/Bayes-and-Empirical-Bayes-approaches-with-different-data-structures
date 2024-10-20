## Abstract

Empirical Bayes methods have played a significant role in Statistics over the past decades. Many techniques emerged around the central idea of a compromise between the Bayesian 
and frequentist approaches. This dual nature of Empirical Bayes estimation implies uncertainty from a theoretical point of view. As stated by Efron (2019):

> "Empirical Bayes methods, though of increasing use, still suffer from an uncertain theoretical basis, enjoying neither the safe haven of Bayes theorem nor the steady support of frequentist optimality."

The objective of this thesis is thus to explore the properties of various Empirical Bayes estimation techniques that have evolved. The central distinction that guides this work 
lies between classic Empirical Bayes (EB), which includes the $G$-modeling and $F$-modeling approaches, and the use of empirical Bayes ingredients in Bayesian learning, which 
will be referred to as "Empirical Bayes in Bayes" (EBIB). 
In this latter case, the data structure does not necessarily envisage large-scale parallel experiments, as in classic EB, and there is no true prior law. Although debatable, 
recent results prove that the EBIB posterior distribution may be a computationally convenient approximation of a genuine Bayesian posterior law. The original contribution of 
the thesis is to explore these results further and develop their use in sparse regression. An extensive simulation study is conducted to give a more concrete sense of 
higher-order asymptotic approximation properties of the EB posterior distribution and is used to perform shrinkage regression on both real and simulated datasets.
