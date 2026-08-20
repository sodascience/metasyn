Synthetic data without raw data
===============================

In some cases you might want to generate synthetic data, without having access to an input dataset, or add additional columns to an existing dataset.
Metasyn is able to do this, when you fully specify the distributions for each of the columns. There are two ways to achieve this:
use the :class:`MetaFrameBuilder` or using a ``.toml`` configuration file. Below we will give an example for each.

For example, the following configuration file will generate a GMF file with 100 rows of synthetic data, with a unique key column named ``PassengerId``:

.. tab:: MetaFrameBuilder

   .. code-block:: python

      from metasyn.builder import MetaFrameBuilder
      from metasyn.distribution import UniqueKeyDistribution

      builder = MetaFrameBuilder(n_rows=100)
      builder.add_column("PassengerId")
      builder["PassengerId"].distribution = UniqueKeyDistribution(consecutive=True, low=0)
      builder["PassengerId"].description = "ID of the unfortunate passenger."
      builder["PassengerId"].prop_missing = 0.0

      mf = builder.fit()

.. tab:: Configuration file

   .. code-block:: shell

      metasyn create-meta --config your_config_file.toml -o gmf_file.json

   With the following ``your_config_file.toml``:


   .. code-block:: toml

      config_version = "1.2"
      n_rows = 100

      [[var]]

      name = "PassengerId"
      data_free = true  # Needed in the configuration file to signal it is not using an input dataframe.
      prop_missing = 0.0
      description = "ID of the unfortunate passenger."
      var_type = "discrete"
      distribution = {name = "core.unique_key", unique = true, parameters = {consecutive = true, low = 0}}

See :doc:`distribution page </api/metasyn.distribution>` for a list of distributions that can be chosen from.

Setting defaults
----------------

Writing the same things for every distribution can be tedious work, but you can also create defaults for
variables. The following can be set by default: ``data_free``, ``prop_missing``, ``distribution`` and ``privacy``.
Since the distribution depends on the type of the column, you can set the default distribution per column type.
Below is an example on how to set defaults:

.. tab:: MetaFrameBuilder

   .. code-block:: python

      from metasyn.distribution import DiscreteUniformDistribution, ContinuousNormalDistribution, FakerDistribution

      builder.defaults["prop_missing"] = 0.1
      builder.defaults["distribution"] = {
         "discrete": DiscreteUniformDistribution(1, 30),
         "continuous": ContinuousNormalDistribution(0, 1),
         "string": FakerDistribution(faker_type="name", locale="en_US")
      }

      builder.add_column("ID", var_type="discrete")
      builder.add_column("name", var_type="string")
      builder.add_column("Result", var_type="continuous")
      builder.fit().synthesize(5)

.. tab:: Configuration file

   .. code-block:: toml

      config_version = "1.2"
      n_rows = 100

      [defaults]

      data_free = true
      prop_missing = 0.1

      [defaults.distribution]

      discrete = {name = "core.uniform", parameters = {lower = 1, upper = 30}}
      continuous = {name = "core.normal", parameters = {mean = 0, sd = 1}}
      string = {name = "core.faker", parameters = {faker_type = "name", locale = "en_US"}}

   With this block, you won't have to set the ``data_free`` parameter, and the default
   proportion of missing values is set to 0.1. For discrete columns, the distribution
   will be set to a uniform distribution between 1 and 30, etc.

   With the defaults set as above, you only need to specify the ``name`` and ``var_type`` of
   the columns:

   .. code-block:: toml

      [[var]]
      name = "some discrete variable"
      var_type = "discrete"
