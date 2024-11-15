Synthetic data without raw data
-------------------------------

It is also possible to create a GMF file without inputting a dataset, or to add additional fictional columns to those already present in a dataset. 

To do so, you need to fully specify each column (variable) you want to generate. You will also need to set the data_free parameter to true, to indicate that the variable will be generated from scratch, instead of being based on existing data.
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


