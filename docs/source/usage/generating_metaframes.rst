Generating MetaFrames
=====================

One of the main features of ``metasyn`` is to create a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>`, an object which captures the essential aspects of the dataset, including variable names, types, data types, the percentage of missing values, and distribution attributes. :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` objects essentially capture all the information needed to generate a synthetic dataset that aligns with the original dataset.

.. image:: /images/pipeline_estimation_simple.png
   :alt: MetaFrame Generation Flow
   :align: center


.. admonition:: Want to learn more?
    
   This page focuses on using ``metasyn`` to generate MetaFrames. If you're interested in learning more about how MetaFrames are generated behind the scenes and the assumptions involved, see the :doc:`/metasyn_in_detail` page for details.

Preparing the Dataset
---------------------

Before we can pass a dataset into metasyn, we need to convert it to a `Polars <https://pola.rs>`__ DataFrame. In doing so, we can indicate which columns contain categorical values. We can also tell Polars to find columns that may contain dates or timestamps. Metasyn later uses this information to generate categorical or date-like values where appropriate. For more information on how to use Polars, check out the `Polars documentation <https://docs.pola.rs/>`__.

For example, if we want to load a dataset named 'dataset.csv' into a Polars DataFrame, set the columns ``Color`` and ``Fruit`` to be categorical and parse dates in the DataFrame. We can use the following code:

.. code:: python

   # Create a Polars DataFrame
   df = pl.read_csv(
       source="dataset.csv",
       schema_overrides={"Color": pl.Categorical, "Fruit": pl.Categorical},
       try_parse_dates=True,
   )

.. admonition:: Note on Pandas and Polars DataFrames

    Internally, ``metasyn`` uses Polars (instead of Pandas) mainly because typing and the handling of non-existing data is more consistent. It is possible to supply a Pandas DataFrame instead of a Polars DataFrame to the ``MetaFrame.fit_dataframe`` method. However, this uses the automatic Polars conversion functionality, which for some edge cases results in problems. Therefore, we recommend users to create Polars DataFrames. The resulting synthetic dataset is always a Polars DataFrame, but this can be easily converted back to a Pandas DataFrame by using ``df_pandas = df_polars.to_pandas()``.

Generating a MetaFrame
----------------------
With the DataFrame in place, we can now generate a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` object using the :meth:`metasyn.MetaFrame.fit_dataframe(df) <metasyn.metaframe.MetaFrame.fit_dataframe>` class method, passing in a DataFrame as a parameter.

.. image:: /images/pipeline_estimation_code.png
   :alt: MetaFrame Generation With Code Snippet
   :align: center

The following code returns a :obj:`MetaFrame<metasyn.metaframe.MetaFrame>` object named :obj:`mf`, based on a DataFrame named :obj:`df`.

.. code-block:: python
    
   mf = metasyn.MetaFrame.fit_dataframe(df)



It is possible to print the statistical metadata contained in the :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` to the console/output log. This can simply be done by calling the Python built-in `print` function on a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>`:

.. code-block:: python

    print(mf)


.. _OptionalParams:

Optional Parameters
----------------------
The :meth:`metasyn.MetaFrame.fit_dataframe() <metasyn.metaframe.MetaFrame.fit_dataframe>` class method
allows you to have more control over how your synthetic dataset is generated with additional (optional)
parameters:
    
Besides the required `df` parameter, :meth:`metasyn.MetaFrame.fit_dataframe() <metasyn.metaframe.MetaFrame.fit_dataframe>`
accepts three parameters: ``var_specs``, ``dist_providers`` and ``privacy``.

Let's take a look at each optional parameter individually:

var_specs
^^^^^^^^^
**var_specs** is an optional list that outlines specific directives for columns (variables) in the DataFrame.
This list can also be generated from a .toml file. In that case you have to provide a string of path instead of
a list.
The potential directives include:

    - ``name``: This specifies the column name and is mandatory.

    - ``distribution``: Allows you to specify the statistical distribution of each column. To see what distributions are available refer to the :doc:`distribution package API reference</api/metasyn.distribution>`.
    
    - ``unique``: Declare whether the column in the synthetic dataset should contain unique values. By default no column is set to unique.
    
    .. admonition:: Detection of unique variables

        When generating a MetaFrame, ``metasyn`` will automatically analyze the columns of the input DataFrame to detect ones that contain only unique values.
        If such a column is found, and it has not manually been set to unique in the ``var_specs`` list, the user will be notified with the following warning:
        ``Variable '[variable name]' was detected to be unique, but has not explicitly been set to unique. To generate only unique values for column 'PassengerId', set unique to True. To dismiss this warning, set unique to False."``
        
        It is safe to ignore this warning - however, be aware that without setting the column as unique, ``metasyn`` may generate duplicate values for that column when synthesizing data.
        
        To remove the warning and ensure the values in the synthesized column are unique, set the column to be unique (``unique = True``) in the ``var_specs`` list.    
    
    - ``description``: Includes a description for each column in the DataFrame.


    - ``privacy``: Set the privacy level for each column to bypass potential privacy concerns.

    
    - ``prop_missing``: Set the intended proportion of missing values in the synthetic data for each column.


.. admonition:: Example use of the ``var_specs`` parameter

    - For the column ``PassengerId``, we want unique values. 
    - The ``Name`` column should be populated with realistic fake names using the `Faker <https://faker.readthedocs.io/en/master/>`_ library.
    - In the ``Fare`` column, we aim for an exponential distribution.
    - Age values in the ``Age`` column should follow a discrete uniform distribution, ranging between 20 and 40.

    The following code to achieve this would look like:

    .. code-block:: python

        from metasyn.distribution import FakerDistribution, DiscreteUniformDistribution
        from metasyn.config import VarSpec

        # Create a specification list for generating synthetic data
        var_specs = [
            # Ensure unique values for the `PassengerId` column
            VarSpec("PassengerId", unique=True),

            # Utilize the Faker library to synthesize realistic names for the `Name` column
            VarSpec("Name", distribution=FakerDistribution("name")),

            # Fit `Fare` to an log-normal distribution, but base the parameters on the data
            VarSpec("Name", distribution="lognormal"),

            # Set the `Age` column to a discrete uniform distribution ranging from 20 to 40
            VarSpec("Age", distribution=DiscreteUniformDistribution(20, 40)),
        ]

        mf = MetaFrame.fit_dataframe(df, var_specs=var_specs)

   
dist_providers
^^^^^^^^^^^^^^^^

The parameter **dist_providers** determines which plug-ins should be loaded and in which order. By default all plug-ins will be loaded and available for fitting, which
is what most users will probably want. It can be helpful for reproducibility to specify which providers were used. The distributions that are available through the `metasyn`
package itself (without installing any plug-ins) is called "builtin".
   
privacy
^^^^^^^^^
**privacy** allows you to set the global privacy level for synthetic data generation. If it's not provided, the function defaults it to ``None``.

.. warning::
    Privacy features (such as differential privacy or other forms of disclosure control) are currently under active development. More information on currently available extensions can be found in the :doc:`/usage/extensions` section.


Config Files 
^^^^^^^^^^^^
It is also possible specify variable specifications, distribution providers and privacy levels through a .toml config file. This is mostly intended for working with the :doc:`/usage/cli`, but can also be used in the Python API. Information on how to use config files can be found in the :doc:`/usage/config_files` section.
