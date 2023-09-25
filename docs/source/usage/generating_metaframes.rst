Generating MetaFrames
=====================

One of the main features of metasyn is to create a :obj:`MetaFrame <metasyn.metaframe.MetaFrame>`, an object which captures the essential aspects of the dataset, including variable names, types, data types, the percentage of missing values, and distribution attributes. :obj:`MetaFrame <metasyn.metaframe.MetaFrame>` objects essentially capture all the information needed to generate a synthetic dataset that aligns with the original dataset.

.. image:: /images/pipeline_estimation_simple.png
   :alt: MetaFrame Generation Flow
   :align: center


.. admonition:: Want to learn more?
    
   This page focuses on using metasyn to generate MetaFrames. If you're interested in learning more about how MetaFrames are generated behind the scenes and the assumptions involved, see the :doc:`/about/metasyn_in_detail` page for details.
   
Basics
------

Metasyn can generate metadata from any given dataset (provided as Polars or Pandas DataFrame), using the :meth:`metasyn.MetaFrame.fit_dataframe(df) <metasyn.metaframe.MetaFrame.fit_dataframe>` classmethod.

.. image:: /images/pipeline_estimation_code.png
   :alt: MetaFrame Generation With Code Snippet
   :align: center

This function requires a :obj:`DataFrame` to be specified as parameter. The following code returns a :obj:`MetaFrame<metasyn.metaframe.MetaFrame>` object named :obj:`mf`, based on a DataFrame named :obj:`df`.

.. code-block:: python
    
   mf = metasyn.MetaFrame.from_dataframe(df)

.. admonition:: Note on Pandas and Polars DataFrames

    Internally, metasyn uses Polars (instead of Pandas) mainly because typing and the handling of non-existing data is more consistent. It is possible to supply a Pandas DataFrame instead of a Polars DataFrame to the ``MetaFrame.from_dataframe`` method. However, this uses the automatic Polars conversion functionality, which for some edge cases result in problems. Therefore, we advise users to create Polars DataFrames. The resulting synthetic dataset is always a Polars DataFrame, but this can be easily converted back to a Pandas DataFrame by using ``df_pandas = df_polars.to_pandas()``.

.. _OptionalParams:

Optional Parameters
----------------------
The :meth:`metasyn.MetaFrame.fit_dataframe() <metasyn.metaframe.MetaFrame.fit_dataframe>` class method allows you to have more control over how your synthetic dataset is generated with additional (optional) parameters:
    
Besides the required `df` parameter, :meth:`metasyn.MetaFrame.fit_dataframe() <metasyn.metaframe.MetaFrame.fit_dataframe>` accepts three parameters: ``spec``, ``dist_providers`` and ``privacy``.

Let's take a look at each optional parameter individually:

spec
^^^^
**spec** is an optional dictionary that outlines specific directives for each DataFrame column (variable). The potential directives include:
   
    - ``distribution``: Allows you to specify the statistical distribution of each column. To see what distributions are available refer to the :doc:`distribution package API reference</api/metasyn.distribution>`.
    
    - ``unique``: Declare whether the column in the synthetic dataset should contain unique values. By default no column is set to unique.
    
    .. admonition:: Detection of unique variables

        When generating a MetaFrame, metasyn will automatically analyze the columns of the input DataFrame to detect ones that contain only unique values.
        If such a column is found, and it has not manually been set to unique in the ``spec`` dictionary, the user will be notified with the following warning:
        ``Warning: Variable [column_name] seems unique, but not set to be unique. Set the variable to be either unique or not unique to remove this warning``
        
        It is safe to ignore this warning - however, be aware that without setting the column as unique, metasyn may generate duplicate values for that column when synthesizing data.
        
        To remove the warning and ensure the column remains unique, set the column to be unique (``"column" = {"unique": True}``) in the ``spec`` dictionary.    
    
    - ``description``: Includes a description for each column in the DataFrame.


    - ``privacy``: Set the privacy level for each column to bypass potential privacy concerns.

    
    - ``prop_missing``: Set the intended proportion of missing values in the synthetic data for each column.


.. admonition:: Example use of the ``spec`` parameter

    - For the column ``PassengerId``, we want unique values. 
    - The ``Name`` column should be populated with realistic fake names using the `Faker <https://faker.readthedocs.io/en/master/>`_ library.
    - In the ``Fare`` column, we aim for an exponential distribution.
    - Age values in the ``Age`` column should follow a discrete uniform distribution, ranging between 20 and 40.
    - The ``Cabin`` column should adhere to a predefined structure: a letter between A and F, followed by 2 to 3 digits (e.g., A40, B721).

    The following code to achieve this would look like:

    .. code-block:: python
        
        from metasyn.distribution import FakerDistribution, DiscreteUniformDistribution, RegexDistribution

        # Create a specification dictionary for generating synthetic data
        var_spec = {

            # Ensure unique values for the `PassengerId` column
            "PassengerId": {"unique": True},

            # Utilize the Faker library to synthesize realistic names for the `Name` column
            "Name": {"distribution": FakerDistribution("name")},

            # Fit `Fare` to an exponential distribution based on the data
            "Fare": {"distribution": "ExponentialDistribution"},

            # Fit `Age` to a discrete uniform distribution ranging from 20 to 40
            "Age": {"distribution": DiscreteUniformDistribution(20, 40)},

            # Use a regex-based distribution to generate `Cabin` values following [ABCDEF]\d{2,3}
            "Cabin": {"distribution": RegexDistribution(r"[ABCDEF][0-9]{2,3}")}

        }

        mf = MetaFrame.fit_dataframe(df, spec=var_spec)

   
dist_providers
^^^^^^^^^^^^^^^^
**dist_providers** allows you to specify distribution providers (as strings or actual provider objects) to use when fitting distributions to the column data.

   
privacy
^^^^^^^^^
**privacy** allows you to set the global privacy level for synthetic data generation. If it's not provided, the function defaults it to ``None``.
For more on privacy modules available refer to :mod:`Privacy Features (experimental) <metasyn.privacy>`.

.. warning::
    Privacy features (such as differential privacy or other forms of disclosure control) are currently under active development. More information on currently available extensions can be found in the :doc:`/usage/extensions` section.
