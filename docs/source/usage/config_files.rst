Configuration Files
===================

Configuration files are designed to be used in conjunction with the :doc:`/usage/cli` (though they can be used with the Python API too) to help specify distribution behavior when creating generative metadata. 

In the configuration file, you can specify the variable specification, distribution providers, privacy level and number of rows of data to generate. 

The configuration file must be in the `.toml` format. 

.. admonition:: What is TOML?

   TOML, short for Tom's Obvious, Minimal Language, aims to be a minimal configuration file with easy readability. 

   A TOML file consists of key/value pairs, tables (groups of key/value pairs), and optionally, arrays of tables. Values can be strings, numbers, booleans, dates/times, or arrays. TOML is **case sensitive**, but does not care about indentation or whitespace. 

   Refer to the `TOML documentation <https://toml.io/en/>`_ to learn more about the syntax and structure of TOML files.


Configuration File Structure
----------------------------


Specifying rows
^^^^^^^^^^^^^^^

To specify the number of rows to generate, use the ``n_rows`` key.

For example:

.. code-block:: toml

   n_rows = 100

Variable specification
^^^^^^^^^^^^^^^^^^^^^^^

To set the variable spec for a variable, create an entry in the ``[[var]]`` array. Each entry should include the following:

#. ``name``: The name of the variable.
#. ``description`` (optional): A description of the variable.
#. ``distribution``: The distribution specification for the variable. This is a dictionary that includes the ``implements`` key specifying the distribution type, and optionally a ``parameters`` key specifying parameters for the distribution. To find distributions, and their parameters, refer to the :doc:`/api/metasyn.distribution` page.
#. ``privacy`` (optional): The privacy specification for the variable. This is a dictionary that includes the ``name`` of the privacy, and a ``parameters`` key for specifying its parameters.
#. ``prop_missing`` (optional): The proportion of missing values for the variable.

For example:

.. code-block:: toml

   [[var]]
   name = "Cabin"
   description = "Cabin number of the passenger."
   distribution = {implements = "core.regex", parameters = {regex_data = "[A-F][0-9]{2,3}"}}
   prop_missing = 0.1
   privacy = {name = "disclosure", parameters = {partition_size = 21}}


Distribution providers
^^^^^^^^^^^^^^^^^^^^^^

To specify the distribution providers, use the ``dist_providers`` key. This is an array of strings, where each string is the name of a distribution provider. Note that this is only necessary if you install distribution plugins.

For example:

.. code-block:: toml

   dist_providers = ["builtin", "metasyn-disclosure"]

Privacy level
^^^^^^^^^^^^^

To specify the privacy level, use the ``privacy`` key. This is a dictionary that includes the ``name`` key specifying the privacy level, and optionally a ``parameters`` key specifying parameters for the privacy level.

For example:

.. code-block:: toml

   [privacy]
   name = "disclosure"
   parameters = {partition_size = 11}


Example Configuration File
--------------------------

The following is an example which specifies the distribution providers, privacy level, variable specifications and number of rows of data to generate (for the :doc:`Titanic demo dataset </api/metasyn.demo>`):


.. code-block:: toml

   dist_providers = ["builtin", "metasyn-disclosure"]

   n_rows = 100

   [privacy]
   name = "disclosure"
   parameters = {partition_size = 11}


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
   privacy = {name = "disclosure", parameters = {partition_size = 21}}


Synthetic data without input file
---------------------------------
It is also possible to create a GMF file without inputting a dataset, or to add additional fictive columns to those already present in a dataset. 

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


