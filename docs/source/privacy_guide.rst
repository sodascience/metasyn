Privacy guide
=============

This is a non-technical guide on how to create synthetic data with metasyn: it gives general recommendations on how to minimize the
risk of disclosing privacy sensitive information. To understand how to technically implement those recommendations or how to generate synthetic data
with metasyn is explained elsewhere in the documentation. To get started, you can read the :doc:`quick_start` section, and
for more advanced usage the :doc:`improve_synth` section.

Preface
-------

Metasyn is a tool that tries to minimize the risk of diclosing privacy sensitive information at the cost of
some utility of the output. Assessing the risk of releasing synthetic data (or any privacy sensitive information for that matter) is context dependent,
context, which a tool such as metasyn doesn't have. Therefore, you as the creator of synthetic data are ultimately responsible
for mitigating the risks of diclosure.

.. admonition:: Disclaimer

    The software metasyn and this privacy guide are provided as is and any responsibility for disclosing privacy sensitive
    information is disclaimed, whether this is due to misuse, omission, bugs or any other reason.

Luckily, metasyn gives you the opportunity to audit the exact information that will be released with your
synthetic data: the :ref:`GMF file<metaframes and GMF>`. It is highly recommended to look through this file manually
(you can use a text editor or an internet browser such as firefox).

Privacy plugin
--------------

To reduce the disclosure risk without manual intervention, we have created a disclosure control `plugin <https://github.com/sodascience/metasyn-disclosure-control>`_.
This plugin is based on the SDC (statistical disclosure control) rules that the CBS uses, and attempts to adhere to
most of the guidelines, such as the minimum number of observations, group disclosure and dominance. For example,
in the base metasyn implementation the uniform distribution might output the minimum and maximum of a particular column.
In many cases this might not be an issue, but if this column happens to be someone's date of birth, it might be.
The disclosure control plugin prevents this by aggregating first over multiple rows, which means the minimum might
instead become the average of the 10 lowest values, which is a lot less disclosive.

We therefore recommend to install and use the disclosure plugin for privacy sensitive datasets, even though it comes at
a small cost to the utility of the synthetic dataset.

Identifying columns
-------------------

Columns can be `classified <https://sdcpractice.readthedocs.io/en/latest/measure_risk.html#classification-of-variables>`_
into different levels of how easy it is to identify someone using data in that column.  The most risky columns are "direct identifiers",
such as social security numbers, addresses, phone numbers, credit card numbers. These are columns that require extra care
and which you should either remove or manually set the distribution for. Distributions such as the :class:`metasyn.distribution.faker.FakerDistribution`
and :class:`metasyn.distribution.regex.RegexDistribution` can be particular useful for this. Then there are quasi-identifiers
such as age, sex or postal codes. These generally don't need removal, but depending on the context some care might
be warranted. The final category contains non-identifying variables, which are for example the output results of your
a test. Given that metasyn does not keep relations between columns, these are of a lesser concern.

Categorical columns
-------------------

Categorical columns often require a little more attention, since automatic modelling has a harder time dealing with them.

Sometimes categories have (hierarchical) relationships between them. For example, cervical cancer and lung cancer are
both types of cancer. Metasyn cannot know this of course, and the disclosure control plugin might fail to notice
that more than 90% percent falls into this implicit category, violating the general SDC rules as implemented by the CBS.

Another pitfall is if the categories of the column is an identifying variable. For example, if the column represents names
and individuals are many times present in the same table, then these names might become the labels of a multinoulli distribution.
The solution is the same as with other identifying variables, which is to manually set the distribution without the names
or remove the column altogether.

Column names
------------

In theory, column names could be disclosive to individuals or groups. This might be a rare occasion, but metasyn does
not provide any protection against this case for obvious reasons.

File metadata
-------------

Some files such as .sav and .dta files have the ability to store metadata. This metadata can contain privacy sensitive information,
which can end up in the GMF file, and thus the synthetic file. With these kinds of file formats, always look through the
metadata in the GMF file to prevent this.

Fit log
-------

Since version 3.0, metasyn can create a report on the fitting procedure. This will give information on the different
distributions that were fitted, which privacy parameters where used, and any particularities that were encounted
during the fit. The fit log can be accessed using the :class:`MetaFrameBuilder` class:

.. code:: python

    from metasyn.builder import MetaFrameBuilder

    builder = MetaFrameBuilder()
    builder.add_dataframe(df)
    mf = builder.fit()
    builder.fit_log.save_md("fitlog.md")

.. admonition:: FitLog

    Do **not** publish the fit log! The fit can generally contain information that is more privacy sensitive
    than the metaframe or synthetic dataset. It shows the creator of the synthetic data a more complete
    picture on what metasyn has done, and allows the creator to adjust the procedure if necessary.
