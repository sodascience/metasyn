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


The MetaFrameBuilder
--------------------

In the :doc:`quick_start` you have seen how to create a :class:`MetaFrame` directly using the
:meth:`MetaFrame.fit_dataframe` method. This method builds the :class:`MetaFrame` at once from
the arguments that are supplied. Especially, when using metasyn interactively, this can be a bit cumbersome.
A more convenient interface is the :class:`MetaFrameBuilder` class, with which you can build the
:class:`MetaFrame` step by step.

.. tab:: Python (MetaFrameBuilder)

   .. code-block:: python

      from metasyn import MetaFrameBuilder
      from metasyn.distribution import FakerDistribution
      from metasyncontrib.disclosure import DisclosurePrivacy

      builder = MetaFrameBuilder()
      builder.add_dataframe(df)
      builder["Name"].distribution = FakerDistribution(faker_type="name")
      builder.privacy = DisclosurePrivacy()
      mf = builder.fit()

.. tab:: Python (fit_dataframe)

   .. code-block:: python

      from metasyn import MetaFrame
      from metasyn.distribution import FakerDistribution
      from metasyncontrib.disclosure import DisclosurePrivacy

      specs = [{"name": "Name", "distribution": FakerDistribution(faker_type="name")}]

      mf = MetaFrame.fit_dataframe(
         df,
         privacy=DisclosurePrivacy(),
         var_specs=specs,
      )

.. tab:: Configuration file

   .. code-block:: python

      from metasyn import MetaFrame

      MetaFrame.fit_dataframe(
         df,
         config="your_config_file.toml"
      )

   This refers to a configuration file called ``your_config_file.toml``:

   .. code-block:: toml

      config_version = "1.2"

      [defaults.privacy]
      name = "disclosure"

      [[var]]
      name = "Name"
      description = "Name of the unfortunate passenger of the titanic."
      distribution = {name = "core.faker", parameters = {faker_type = "name"}}

   More examples for metasyn configuration files are available on our
   `GitHub page <https://github.com/sodascience/metasyn/tree/develop/examples/config_files>`_.

.. admonition:: What is the TOML file format?

   The `TOML <https://toml.io/en/>`_ file format can be read with any text editor, and is human and machine-readable.
   You should be able to create your own TOML files from the examples below, but for more details refer to the TOML 
   `Documentation <https://toml.io/en/>`_. One important thing to note is that the TOML format is case sensitive.


The remainder of this page serves as a reference for the different options to improve synthetic data quality.


General specifications
----------------------

Three general options can be set: ``privacy``, ``n_rows``, and ``plugins``. 
In our python interface, these are arguments to :py:meth:`~MetaFrame.fit_dataframe()` or attributes of :class:`MetaFrameBuilder`; in the 
configuration file these are mentioned at the top of the file.

Privacy: ``privacy``
^^^^^^^^^^^^^^^^^^^^

Using privacy plug-ins (see :doc:`plugins`), metasyn can increase the level of privacy.
An example is ``disclosure`` privacy, which limits the influence of various disclosive values such as outliers on the fitted distributions.

.. tab:: Python (MetaFrameBuilder)

   .. code-block:: python

      from metasyncontrib.disclosure import DisclosurePrivacy

      builder.privacy = DisclosurePrivacy(partition_size=11)

.. tab:: Python (fit_dataframe)

   .. code-block:: python

      from metasyncontrib.disclosure import DisclosurePrivacy

      MetaFrame.fit_dataframe(
         df,
         privacy=DisclosurePrivacy(partition_size=11)
      )

.. tab:: Configuration file

   .. code-block:: toml

      [defaults.privacy]
      name = "disclosure"
      parameters = {partition_size = 11}


Number of rows: ``n_rows``
^^^^^^^^^^^^^^^^^^^^^^^^^^

By default metasyn will set the number of rows to the number of rows of your dataframe. This can be disclosive
or undesirable. In this case you can specify it manually:

.. tab:: Python (MetaFrameBuilder)

   .. code-block:: python

      builder.n_rows = 100

.. tab:: Python (fit_dataframe)

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         n_rows=100
      )

.. tab:: Configuration file

   .. code-block:: toml

      n_rows = 100


Distribution registry: ``plugins``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extra distributions and fitters can be added using plugins. By default all installed plugins
will be used. For reproducibility, it is a good idea to set the plugins explicitly, so that 
other people using your configuration file understand which plugins were used. This can be done 
as follows:


.. tab:: Python (MetaFrameBuilder)

   builder.plugins = ["builtin", "disclosure"]

.. tab:: Python (fit_dataframe)

   .. code-block:: python

      MetaFrame.fit_dataframe(
         df,
         plugins=["builtin", "disclosure"],
      )

.. tab:: Configuration file

   .. code-block:: toml

      plugins = ["builtin", "disclosure"]


Column specifications
---------------------

In addition to specifications that apply to all columns, you can also specify the behavior for individual columns.
The most common use-case for this is to set the distribution type and/or parameters. 

.. tab:: Python (MetaFrameBuilder)

   .. code:: python

      from metasyn.distribution import RegexDistribution

      builder["Cabin"].description = "Cabin number of the passenger."
      builder["Cabin"].distribution = RegexDistribution("[A-F][0-9]{2,3}")
      builder["Cabin"].prop_missing = 0.2


.. tab:: Python (fit_dataframe)

   .. code-block:: python

      from metasyn.distribution import RegexDistribution

      specs = [
         {
            "name": "Cabin", 
            "description": "Cabin number of the passenger.", 
            "distribution": RegexDistribution("[A-F][0-9]{2,3}"), 
            "prop_missing": 0.2,
         }
      ]
      
      MetaFrame.fit_dataframe(df, var_specs=specs)

.. tab:: Configuration file

   .. code-block:: python

      # In this example you put the specifications in the toml file.
      MetaFrame.fit_dataframe(df, config="your_config_file.toml")

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      description = "Cabin number of the passenger."
      distribution = {implements = "core.regex", parameters = {regex_data = "[A-F][0-9]{2,3}"}}
      prop_missing = 0.2

      [[var]]
      name = "Another column name"
      description = "With descriptions."
      # And more specifications for that column after this.


Description: ``description``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can add a description about your column. This will not be used in the estimation phase of metasyn,
but it will be present in the resulting GMF file so that others can more easily understand what is
in the data.


.. tab:: Python (MetaFrameBuilder)

   .. code-block:: python

      builder["Cabin"].description = "Cabin number of the passenger."


.. tab:: Python (fit_dataframe)

   .. code-block:: python

      specs = [ {"name": "Cabin", "description": "Cabin number of the passenger."} ]
      MetaFrame.fit_dataframe(df, var_specs=specs)

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      description = "Cabin number of the passenger."


Missing values: ``prop_missing``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default metasyn will estimate the proportion of missing values from the data, but you can
overwrite this with the ``prop_missing`` parameter (between 0 and 1, inclusive):

.. tab:: Python (MetaFrameBuilder)

   .. code-block:: python

      builder["Cabin"].prop_missing = 0.2

.. tab:: Python (fit_dataframe)

   .. code-block:: python

      specs = [ {"name": "Cabin", "prop_missing": 0.2} ]
      MetaFrame.fit_dataframe(df, var_specs=specs)

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      prop_missing = 0.2


Privacy: ``privacy``
^^^^^^^^^^^^^^^^^^^^

You can override the privacy level for specific columns:

.. tab:: Python (MetaFrameBuilder)

   .. code-block:: python

      from metasyncontrib.disclosure import DisclosurePrivacy
   
      builder["Cabin"].privacy = DisclosurePrivacy()

.. tab:: Python (fit_dataframe)

   .. code-block:: python

      from metasyncontrib.disclosure import DisclosurePrivacy

      specs = [ {"name": "Cabin", "privacy": DisclosurePrivacy()} ]
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
increasing. When the column represents a variable that is known to be unique (such as IDs or other key variables), this uniqueness can be enforced with:

.. tab:: Python (MetaFrameBuilder)

   .. code-block:: python

      builder["Cabin"].distribution = {"unique": True}

.. tab:: Python (fit_dataframe)

   .. code-block:: python

      specs = [ {"name": "Cabin", "distribution": {"unique": True}} ]
      MetaFrame.fit_dataframe(df, var_specs=specs)

.. tab:: Configuration file

   .. code-block:: toml

      [[var]]
      name = "Cabin"
      distribution = {unique = true}  # Notice the lower case for TOML



Distribution: ``distribution``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can specify the distribution for a column in two different ways: either specify only the type of distribution
and let metasyn find the parameters or specify both the type and parameters of the distribution.

.. tab:: Python (MetaFrameBuilder)

   .. code-block:: python

      builder["Cabin"].distribution = RegexDistribution  # Set distribution type, but metasyn infers parameters
      builder["Cabin"].distribution = RegexDistribution("[A-F][0-9]{2,3}")

.. tab:: Python (fit_dataframe)

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
