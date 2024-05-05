Configuration Files
===================

.. give toml primer
.. explain config file 
.. give example of config file
.. link to api to know what things are

.. note::

   The configuration file must be in the `.toml` format. For more information on the format, please refer to the `TOML documentation <https://toml.io/en/>`_.

.. simple example:
An example of a configuration file that specifies the ``PassengerId`` column to be unique and the ``Fare`` column to have a log-normal distribution is as follows:

   .. code-block:: toml

      [[var]]
      name = "PassengerId"
      distribution = {unique = true}  # Notice lower capitalization for .toml files.


      [[var]]
      name = "Fare"
      distribution = {implements = "core.log_normal"}

.. comprehensive example:

.. code-block:: toml

   dist_providers = ["builtin", "metasyn-disclosure"]

   [privacy]
   name = "disclosure"
   parameters = {n_avg = 11}


   [[var]]
   name = "PassengerId"
   distribution = {unique = true}  # Notice booleans are lower case in .toml files.

   [[var]]
   name = "Name"
   prop_missing = 0.1
   description = "Name of the unfortunate passenger of the titanic."
   distribution = {implements = "core.faker", parameters = {faker_type = "name", locale = "en_US"}}

   [[var]]
   name = "Fare"
   distribution = {implements = "core.exponential"}

   [[var]]
   name = "Age"
   distribution = {implements = "core.uniform", parameters = {lower = 20, upper = 40}}

   [[var]]
   name = "Cabin"
   distribution = {implements = "core.regex", parameters = {regex_data = "[A-F][0-9]{2,3}"}}
   privacy = {name = "disclosure", parameters = {n_avg = 21}}


Synthetic data without input file
---------------------------------
It is also possible to create a GMF file without any input dataset (or to add columns that were not present in the dataset). For this to work, you need to supply a configuration file (.toml) that fully specifies all wanted columns. You will need to tell ``metasyn`` in the configuration file that the column is ``data_free``.
data_free is a boolean that indicates whether a variable/column is to be generated from scratch or from an existing column in the DataFrame. set to True, it means that the variable will be generated from scratch, not based on any existing data. If set to False (or not specified), the variable will be based on an existing column in the DataFrame. 

 It is also required to set the number of rows under the `general` section, for example:

   .. code-block:: toml

      n_rows = 100

      [[var]]

      name = "PassengerId"
      data_free = true
      prop_missing = 0.0
      description = "ID of the unfortunate passenger."
      var_type = "discrete"
      distribution = {implements = "core.unique_key", unique = true, parameters = {consecutive = 1, low = 0}}

The example will generate a GMF file that can be used to generate new synthetic data with the ``synthesize``
subcommand described below.

