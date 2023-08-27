Generating MetaFrames
=====================

One of the main functionalities of MetaSynth is the functionality to create a :obj:`MetaFrame <metasynth.dataset.MetaFrame>`, an object which captures the essential aspects of the dataset, including variable names, types, data types, the percentage of missing values, and distribution attributes. :obj:`MetaFrame <metasynth.dataset.MetaFrame>` objects essentially capture all the information needed to generate a synthetic dataset that aligns with the original dataset. MetaSynth can later use this information to generate a synthetic dataset that aligns with the original dataset.

.. image:: /images/pipeline_estimation_simple.png
   :alt: MetaFrame Generation Flow
   :align: center


Basics
------

MetaSynth can generate metadata from any given dataset (provided as Polars or Pandas DataFrame), using the :meth:`metasynth.MetaFrame.fit_dataframe(df) <metasynth.dataset.MetaFrame.fit_dataframe>` classmethod.

.. image:: /images/pipeline_estimation_code.png
   :alt: MetaFrame Generation With Code Snippet
   :align: center

This function requires a :obj:`DataFrame` to be specified as parameter. The following code returns a :obj:`MetaFrame<metasynth.dataset.MetaFrame>` object named :obj:`mf`, based on a DataFrame named :obj:`df`.

.. code-block:: python
    
   mf = metasynth.MetaFrame.from_dataframe(df)

.. note:: 
    Internally, MetaSynth uses Polars (instead of Pandas) mainly because typing and the handling of non-existing data is more consistent. It is possible to supply a Pandas DataFrame instead of a Polars DataFrame to ``MetaDataset.from_dataframe``. However, this uses the automatic Polars conversion functionality, which for some edge cases result in problems. Therefore, we advise users to create Polars DataFrames. The resulting synthetic dataset is always a Polars DataFrame, but this can be easily converted back to a Pandas DataFrame by using ``df_pandas = df_polars.to_pandas()``.


Optional Parameters
----------------------
The :meth:`metasynth.MetaFrame.fit_dataframe() <metasynth.dataset.MetaFrame.fit_dataframe>` class method allows you to have more control over how your synthetic dataset is generated with additional (optional) parameters:
    
Besides the required `df` parameter, :meth:`metasynth.MetaFrame.fit_dataframe() <metasynth.dataset.MetaFrame.fit_dataframe>` accepts three parameters: ``spec``, ``dist_providers`` and ``privacy``.

Let's take a look at each optional parameter individually:

spec
^^^^
**spec** (``spec (Optional[dict[str, dict]] = None)``) is an optional dictionary that outlines specific directives for each DataFrame column. The potential directives include:
   
    - ``distribution``: Allows you to specify the statistical distribution of each column. To see what distributions are available refer to the :doc:`distribution package API reference</api/metasynth.distribution>`.
    
    - ``unique``: Declare whether the column in the synthetic dataset should contain unique values.
    
    
    - ``description``: Includes a description for each column in the DataFrame.


    - ``privacy``: Set the privacy level for each column to bypass potential privacy concerns.

    
    - ``prop_missing``: Set the intended proportion of missing values in the synthetic data for each column.
     

Here's an example use of the ``spec`` parameter, where:

  - For the column ``PassengerId``, we want unique values.
  - The ``Name`` column should be populated with realistic fake names using the `Faker <https://faker.readthedocs.io/en/master/>`_ library.
  - In the ``Fare`` column, we aim for an exponential distribution.
  - Age values in the ``Age`` column should follow a discrete uniform distribution, ranging between 20 and 40.
  - The ``Cabin`` column should adhere to a predefined structure: a letter between A and F, followed by 2 to 3 digits (e.g., A40, B721).

The following code to achieve this would look like:

.. code-block:: python
    
    from metasynth.distribution import FakerDistribution, DiscreteUniformDistribution, RegexDistribution

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
        "Cabin": {"distribution": RegexDistribution(r"[ABCDEF]\d{2,3}")}

    }

    mf = MetaFrame.fit_dataframe(df, spec=var_spec)

dist_providers
^^^^^^^^^^^^^^^^
**dist_providers** (``dist_providers (Union[str, list[str], BaseDistributionProvider, list[BaseDistributionProvider]] = "builtin")``) allows you to specify distribution providers (as strings or actual provider objects) to use when fitting distributions to the column data.

   
privacy
^^^^^^^^^
**privacy** (``privacy (Optional[BasePrivacy] = None)``) allows you to set the global privacy level for synthetic data generation. If it's not provided, the function defaults it to ``None``.
For more on privacy modules available refer to :mod:`Privacy Features (experimental) <metasynth.privacy>`.

.. warning::
    Privacy features (such as differential privacy or other forms of disclosure control) are currently unfinished and under active development.