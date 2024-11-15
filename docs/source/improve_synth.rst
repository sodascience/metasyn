Improve your synthetic data
===========================

When you run metasyn on your dataframe, by default it will attempt
to find the best distribution for each of your columns. This could be sub-optimal:
for example, metasyn won't know whether a column contains names of people.
The column can also be too privacy-sensitive to fit with default methods.

Metasyn provides two paths to improving the quality of your synthetic data: by further
specifying information directly in python, or by providing a configuration file in the
TOML format. For interactive use, we foresee using python directly, and for programmatic
use the configuration file is a more appropriate interface (see also our :doc:`cli`).


.. tab:: Python

   .. code-block:: python

      from metasyn import MetaFrame, VarSpec
      from metasyn.distribution import FakerDistribution
      from metasyncontrib.disclosure import DisclosurePrivacy

      specs = [
         VarSpec(name="Name", distribution=FakerDistribution(faker_type="name")),
      ]

      mf = MetaFrame.fit_dataframe(
         df,
         privacy=DisclosurePrivacy(),
         var_specs=specs,
      )

.. tab:: Configuration file

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         var_specs="your_config_file.toml"
      )

   This refers to a configuration file called ``your_config_file.toml``:

   .. code-block:: toml

      privacy = "disclosure"

      [[var]]
      name = "Name"
      description = "Name of the unfortunate passenger of the titanic."
      distribution = {implements = "core.faker", parameters = {faker_type = "name"}}

   More examples for metasyn configuration files are available on our
   `GitHub page <https://github.com/sodascience/metasyn/tree/develop/examples/config_files>`_.

.. admonition:: What is the TOML file format?

   The `TOML <https://toml.io/en/>`_ file format can be read with any text editor, and is natural to comprehend.
   You should be able to create your own TOML files from the examples below, but for more details refer to the TOML 
   `Documentation <https://toml.io/en/>`_. One important thing to note is that the TOML format is case sensitive.


The remainder of this page serves as a reference for the different options to improve synthetic data quality.


General specifications
----------------------

Three general options can be set: ``privacy``, ``n_rows``, and ``dist_providers``. 
In our python interface, these are arguments to :py:meth:`~MetaFrame.fit_dataframe()`; in the 
configuration file these are mentioned at the top of the file.

Privacy: ``privacy``
^^^^^^^^^^^^^^^^^^^^

Using privacy plug-ins (see :doc:`extensions`), metasyn can increase the level of privacy.
An example is ``disclosure`` privacy, which limits the influence of various problematic 
situations such as outliers.

.. tab:: Python

   .. code-block:: python

      from metasyncontrib.disclosure import DisclosurePrivacy
      MetaFrame.fit_dataframe(
         df,
         privacy=DisclosurePrivacy(partition_size=11)
      )

.. tab:: Configuration file

   .. code-block:: toml

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

      n_rows = 100


Distribution Providers: ``dist_providers``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extra distribution providers can be added using plugins. By default all installed distribution providers
will be used. For reproducibility, it is a good idea to set the distribution providers explicitly, so that 
other people using your configuration file understand which plugins/providers were used. This can be done 
as follows:

.. tab:: Python

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         dist_providers=["builtin", "disclosure"],
      )

.. tab:: Configuration file

   .. code-block:: toml

      dist_providers = ["builtin", "disclosure"]


Adding column specifications
----------------------------

In addition to specifications that apply to all columns, you can also specify the behavior for individual columns.
The most common use-case for this is to set the distribution type and/or parameters. 

.. tab:: Python

   .. code-block:: python

      # we suggest using the VarSpec object like so:
      from metasyn import MetaFrame, VarSpec
      from metasyn.distribution import RegexDistribution

      specs = [
         VarSpec(
            name="Cabin", 
            description="Cabin number of the passenger.", 
            distribution=RegexDistribution("[A-F][0-9]{2,3}"), 
            prop_missing=0.2,
         ),
         VarSpec(
            name=..., 
            description=..., 
            distribution=...,
         ),
         ...
      ]
      
      MetaFrame.fit_dataframe(df, var_specs=specs)

.. tab:: Configuration file

   .. code-block:: python

      # In this example you put the specifications in the toml file.
      MetaFrame.fit_dataframe(df, var_specs="your_config_file.toml")

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      description = "Cabin number of the passenger."
      distribution = {implements = "core.regex", parameters = {regex_data = "[A-F][0-9]{2,3}"}}
      prop_missing = 0.2

      [[var]]
      name = ...
      description = ...
      distribution = ...


Description: ``description``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can add a description about your column. This will not be used in the estimation phase of metasyn,
but it will be present in the resulting GMF file so that others can more easily understand what is
in the data.

.. tab:: Python

   .. code-block:: python

      specs = [ VarSpec(name="Cabin", description="Cabin number of the passenger.") ]
      MetaFrame.fit_dataframe(df, var_specs=specs)

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      description = "Cabin number of the passenger."


Missing values: ``prop_missing``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default metasyn will estimate the proportion of missing values from the data, but you can
overwrite this with the ``prop_missing`` parameter:

.. tab:: Python

   .. code-block:: python

      specs = [ VarSpec(name="Cabin", prop_missing=0.2) ]
      MetaFrame.fit_dataframe(df, var_specs=specs)

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

      from metasyncontrib.disclosure import DisclosurePrivacy

      specs = [ VarSpec(name="Cabin", privacy=DisclosurePrivacy()) ]
      MetaFrame.fit_dataframe(df, var_specs=specs)

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      privacy = "disclosure"


Uniqueness: ``unique``
^^^^^^^^^^^^^^^^^^^^^^

Some distributions produce only values that are unique without any repeats (see distributions starting with ``Unique``
in :doc:`api/metasyn.distribution`). By default, metasyn will not select any unique distributions. An exception
is the :class:`metasyn.distribution.UniqueKeyDistribution <UniqueKeyDistribution>`; if values in the column are sequentially
increasing. When the column represents a variable that is known to be unique (such as IDs), this can be represented with:

.. tab:: Python

   .. code-block:: python

      specs = [ VarSpec(name="Cabin", unique=True) ]
      MetaFrame.fit_dataframe(df, var_specs=specs)

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

      from metasyn.distribution import RegexDistribution

      cabin_dist = RegexDistribution("[A-F][0-9]{2,3}")
      specs = [ VarSpec(name="Cabin", distribution=cabin_dist) ]
      MetaFrame.fit_dataframe(df, var_specs=specs)

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]

      name = "Cabin"
      distribution = {implements = "core.regex", parameters = {regex_data = "[A-F][0-9]{2,3}"}}

Ensure that the column type matches the type of the distribution, for example if the column has string values, use a distribution
that supports the string type. An overview of all distributions sorted by type can be found in the :doc:`API<api/metasyn.distribution>`
