Synthetic data without raw data
===============================

It is also possible to create a GMF file without any input dataset, or to add additional synthetic columns to those already present in a dataset. 

To do so, you need to fully specify each column (variable) you want to generate. You will also need to set the ``data_free`` parameter to true,
to indicate that the variable will be generated from scratch, instead of being based on existing data.
Finally, you will need to set the number of rows to generate.

For example, the following configuration file will generate a GMF file with 100 rows of synthetic data, with a unique key column named ``PassengerId``:

.. code-block:: toml

   n_rows = 100

   [[var]]

   name = "PassengerId"
   data_free = true
   prop_missing = 0.0
   description = "ID of the unfortunate passenger."
   var_type = "discrete"
   distribution = {implements = "core.unique_key", unique = true, parameters = {consecutive = true, low = 0}}

See :doc:`distribution page </api/metasyn.distribution>` for a list of distributions that can be chosen from.

Setting defaults
----------------

Writing the same things for every distribution can be tedious work, but you can also create defaults for
variables. The following can be set by default: ``data_free``, ``prop_missing``, ``distribution`` and ``privacy``.
Since the distribution depends on the type of the column, you can set the default distribution per column type.
Below is an example on how to set defaults:

.. code-block:: toml

   [defaults]

   data_free = true
   prop_missing = 0.1

   [defaults.distribution]
   discrete = {implements = "core.uniform", parameters = {lower = 1, upper = 30}}
   continuous = {implements = "core.normal", parameters = {mean = 0, stdev = 1}}
   string = {implements = "core.faker", parameters = {faker_type = "name", locale = "en_US"}}

With this block, you won't have to set the ``data_free`` parameter, and the default
proportion of missing values is set to 0.1. For discrete columns, the distribution
will be set to a uniform distribution between 1 and 30, etc.

With the defaults set as above, you only need to specify the ``name`` and ``var_type`` of
the columns:

.. code-block:: toml

   [[var]]
   name = "some discrete variable"
   var_type = "discrete"
