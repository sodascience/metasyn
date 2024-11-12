Improve your synthetic data
===========================

When you run metasyn on your dataframe without any directives, metasyn will attempt
to fit the best distribution to each of your columns. This could be sub-optimal:
for example, metasyn won't try see whether a column contains names of people for example.
The column can also be too privacy sensitive to fit with the default methods.


.. admonition:: What is the TOML file format?

   The `TOML <https://toml.io/en/>`_ file format can be read with any text editor, and natural to comprehend.
   You should be able to create your own TOML files from the examples below, but for more details reffer to the TOML 
   `Documentation <https://toml.io/en/>`_. One important thing to note is that the TOML format is case sensitive.


Metasyn provides two ways of improving the quality of your synthetic data: by
specifying directives directly in the API, or by providing a configuration file in the
TOML format. Examples for metasyn configuration files are available on our
`GitHub page <https://github.com/sodascience/metasyn/tree/develop/examples/config_files>`_.

General specifications
----------------------

There are three general specifications that can be set: ``dist_providers``, ``privacy`` and ``n_rows``.
Depending on whether you want to use a configuration file, examples are shown below:

.. tab:: Python

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         dist_providers=["builtin", "disclosure"],
         privacy="disclosure",
         n_rows=100,
      )

.. tab:: Configuration file

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         var_specs="your_configuration_file.toml"
      )

An example of a configuration file:

.. code-block:: toml

   # Put these at the start of the TOML file.
   dist_providers = ["builtin", "disclosure"]
   privacy = "disclosure"
   n_rows = 100


Distribution Providers: ``dist_providers``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extra distribution providers can be added using plugins. By default all installed distribution providers
will be used. It is good practice to set the distribution providers explicitly if you are using more than
the built-in distribution provider, so that other people using your configuration file understand which
plugins/providers were used. This can be done as follows:

.. tab:: Python

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         dist_providers=["builtin", "disclosure"],
      )

.. tab:: Configuration file

   .. code-block:: toml

      # Put this at the start of the TOML file.
      dist_providers = ["builtin", "disclosure"]

Privacy: ``privacy``
^^^^^^^^^^^^^^^^^^^^

To be extra careful with privacy, you can use a different privacy mechanism from the default one.
An example is ``disclosure`` privacy, which limits how the outliers are handled.

.. tab:: Python

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         privacy={"name": "disclosure", "parameters": {"partition_size": 11}}
      )

.. tab:: Configuration file

   .. code-block:: toml

      # Put this at the start of the TOML file.
      privacy = "disclosure"
      parameters = {partition_size = 11}


Number of Rows: ``n_rows``
^^^^^^^^^^^^^^^^^^^^^^^^^^

By default metasyn will set the number of rows to the number of rows of your dataframe. This can be disclosive
or undesirable. In this case you can specify it manually:

.. tab:: Python

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         n_rows=100
      )

.. tab:: Configuration file

   .. code-block:: toml

      # Put this at the start of the TOML file.
      n_rows = 100



Adding column specifications
----------------------------

In addition to specifications that apply to all columns, you can also specify the behavior for individual columns.
The most common use-case for this is to set the distribution type and/or parameters. 

.. tab:: Python

   .. code-block:: python

      # You have to put the specifications in the ...
      MetaFrame.fit_dataframe(df, var_specs=[...])

.. tab:: Configuration file

   .. code-block:: python

      # In this example you put the specifications in the toml file.
      MetaFrame.fit_dataframe(df, var_specs="your_config_file.toml")


In the following examples we will provide specifications for a column called "Cabin". Note that these specifications
can be combined, where only the name of the column is manditory to provide.

.. tab:: Python

   .. code-block:: python
   
      MetaFrame.fit_dataframe(df, var_specs=[{
         "name": "Cabin",
         "description": "Cabin number of the passenger.",
         "distribution": {"implements": "core.regex",
                          "parameters": {"regex_data": "[A-F][0-9]{2,3}"}},
         "prop_missing": 0.2,
      },
      {
         "name": "some_other_column",
         ...
      }])
      
.. tab:: Configuration file

   .. code-block:: toml

      [[var]]

      name = "Cabin"
      description = "Cabin number of the passenger."
      distribution = {implements = "core.regex", parameters = {regex_data = "[A-F][0-9]{2,3}"}}
      prop_missing = 0.1
      privacy = {name = "disclosure", parameters = {partition_size = 21}}

      # Repeat the [[var]] to add the specifications for another column.
      [[var]]

      name = "some_other_column"


Description: ``description``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can add a description about your column. This will not be used in the estimation phase of metasyn,
but it will be present in the resulting GMF file so that others can more easily understand what is
in the data.

.. tab:: Python

   .. code-block:: python

      # You have to put the specifications in the ...
      MetaFrame.fit_dataframe(df, var_specs=[{"name": "Cabin", "description": "some description"}])

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      description = "some_description"


Missing values: ``prop_missing``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default metasyn will estimate the proportion of missing values from the data, but you can
overwrite this with the ``prop_missing`` parameter:

.. tab:: Python

   .. code-block:: python

      # You have to put the specifications in the ...
      MetaFrame.fit_dataframe(df, var_specs=[{"name": "Cabin", "prop_missing": 0.2}])

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      prop_missing = 0.2


Privacy: ``privacy``
^^^^^^^^^^^^^^^^^^^^

You can set the privacy only for specific columns:

.. tab:: Python

   .. code-block:: python

      # You have to put the specifications in the ...
      MetaFrame.fit_dataframe(df, var_specs=[{"name": "Cabin", "privacy": "disclosure"}])

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      privacy = "disclosure"


Uniqueness: ``unique``
^^^^^^^^^^^^^^^^^^^^^^

Some distributions produce only values that are unique without any repeats (see distributions starting with ``Unique``
in :doc:`api/generated/metasyn.distribution`). By default, metasyn will not select any unique distributions. An exception
is the :class:`metasyn.distribution.UniqueKeyDistribution <UniqueKeyDistribution>`; if values in the column are sequentially
increasing. When the column represents a variable that is known to be unique (such as IDs), this can be represented with:

.. tab:: Python

   .. code-block:: python

      # You have to put the specifications in the ...
      MetaFrame.fit_dataframe(df, var_specs=[{"name": "Cabin", "unique": True}])

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      unique = true  # Notice the lower case for TOML



Distribution: ``distribution``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can specify the distribution for a column in two different ways: either specify only the type of distribution
and let metasyn find the parameters or specify both the type and parameters of the distribution.

.. tab:: API

   .. code-block:: python

      # You have to put the specifications in the ...
      MetaFrame.fit_dataframe(df, var_specs=[
         {"name": "Cabin",
          "distribution": {"parameters": {"regex_data": "[A-F][0-9]{2,3}"}}
         }])

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]

      name = "Cabin"
      distribution = {implements = "core.regex", parameters = {regex_data = "[A-F][0-9]{2,3}"}}

Ensure that the column type matches the type of the distribution, for example if the column has string values, use a distribution
that supports the string type. An overview of all distributions sorted by type can be found in the :doc:`API<api/metasyn.distribution>`
