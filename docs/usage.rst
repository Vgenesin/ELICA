Usage
=====

ELiCA likelihoods are used as external cobaya likelihoods. You can reference them
in a cobaya input dictionary or YAML file.

Minimal example
---------------

.. code-block:: yaml

   likelihood:
     elica:

   theory:
     camb:
       extra_args:
         lens_potential_accuracy: 1
         nnu: 3.044
         num_massive_neutrinos: 1

   params:
     tau:
       prior:
         min: 0.01
         max: 0.8
       proposal: 0.003
       ref:
         dist: norm
         loc: 0.060
         scale: 0.001
     logA:
       prior:
         min: 2.61
         max: 3.91
       proposal: 0.001
       ref:
         dist: norm
         loc: 3.054
         scale: 0.001
       drop: true
     As:
       value: "lambda logA: 1e-10*np.exp(logA)"
     H0: 67.32
     ombh2: 0.02237
     omch2: 0.1201
     ns: 0.9651
     mnu: 0.06

   sampler:
     mcmc:
       Rminus1_stop: 0.001

Python API
----------

.. code-block:: python

   from cobaya.model import get_model

   info = {
       "likelihood": {"elica": None},
       "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
       "params": {
           "tau": 0.060,
           "logA": 3.054,
           "As": {"value": "lambda logA: 1e-10*np.exp(logA)"},
           "H0": 67.32,
           "ombh2": 0.02237,
           "omch2": 0.1201,
           "ns": 0.9651,
           "mnu": 0.06,
       },
   }
   model = get_model(info)
   loglike = model.loglikes({})[0][0]

See ``examples/sample_tau.py`` for a full sampling script.
