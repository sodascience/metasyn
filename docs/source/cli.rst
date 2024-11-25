Command-Line Interface
======================
Metasyn provides a command-line interface (CLI) for accessing core functionality without needing to write any Python code.

The CLI currently has three subcommands:

* The ``create-meta`` subcommand, which allows you to **create generative metadata** from a ``.csv file`` and/or a ``.toml`` configuration file.
* The ``synthesize`` subcommand, which allows you to **generate synthetic data** from a ``GMF file``
* The ``schema`` subcommand, which allows you to **create validation schemas** for GMF files.


At any point, you can also use the help command to get more information about the CLI and its subcommands:

.. code-block:: console

   metasyn --help


.. admonition:: When should I (not) use the CLI

   The CLI is mostly feature complete from creating synthetic data from ``.csv files``, but there are some limitations. The main limitation is that it needs to read the
   csv file, for which it tries to use sensible default settings. However, this can result in errors. If you encounter errors, might want to either preprocess your data
   so that it can be read with the default settings, or use the python API instead.
   
   Another limitation is that you cannot specify which columns should be categorical.

Accessing the CLI
-----------------
If you have installed the main ``metasyn`` package, the CLI should be available automatically. You can find instructions on how to install ``metasyn`` in the :doc:`installation` section of the documentation.

Alternatively, the CLI can be accessed through a Docker container, allowing you to run ``metasyn`` in an isolated environment without installing the package on your system.

Here's how you can use Docker to access Metasyn's CLI:

1. **Install Docker:** If Docker isn't already set up on your machine, please follow the instructions on `Docker's official website <https://docs.docker.com/get-docker/>`_.

2. **Pull the ``metasyn`` Docker Image:** After successfully installing Docker, you can download the ``metasyn`` Docker image from Docker Hub using the following command.

   .. code-block:: console

      docker pull sodateam/metasyn

3. **Run metasyn's CLI via Docker:** Once the Docker image is downloaded, you can use the following command to run a ``metasyn`` CLI command within a Docker container (in this case ``--help``), and simultaneously set the working directory (which we denote as `wd` in this case).

   .. tab:: Windows

      .. code-block:: console

         docker run -v %cd%:/wd sodateam/metasyn --help

   .. tab:: Unix or MacOS:

      .. code-block:: console

         docker run -v $(pwd):/wd sodateam/metasyn --help


The ``metasyn`` CLI should now be up and running within the Docker container and ready for use!

.. note:: 
   You can also specify which ``metasyn`` version to use in docker, by adding a tag to the docker image. For example, to use version 1.1.0, you can use the following command:

   .. tab:: Installing a specific version
      
      .. code-block:: console

         docker pull sodateam/metasyn:v1.1.0

   .. tab:: Using a command on a specific version

      .. tab:: Windows

         .. code-block:: console

            docker run -v %cd%:/wd sodateam/metasyn:v1.1.0 --help

      .. tab:: Unix or MacOS:

         .. code-block:: console

            docker run -v $(pwd):/wd sodateam/metasyn:v1.1.0 --help


Creating generative metadata
----------------------------
The ``create-meta`` subcommand combines the estimation and serialization steps in the pipeline into one, this allows you to generate generative metadata for a tabular dataset (in CSV format), and store it in a GMF (Generative Metadata Format) file.

.. image:: /images/pipeline_cli_create_meta.png
   :alt: Creating Generative Metadata using the CLI
   :align: center

The ``create-meta`` command can be used as follows:

.. code-block:: bash

   metasyn create-meta [input] --output [output]

This will:

1. Read the CSV file from the `[input]` filepath
2. Estimate the metadata from the data
3. Serialize the metadata into a GMF file and save it at the `[output]` filepath

The ``create-meta`` command takes two positional arguments:

* ``[input]``: The filepath and name of the CSV file from which the metadata will be generated.
* ``[output]``: The filepath and name of the output JSON file that will contain the generative metadata.

An example of how to use the ``create-meta`` subcommand is as follows:

.. tab:: Local Installation

   .. code-block:: console

      metasyn create-meta wd/my_dataset.csv wd/my_gmf.json

.. tab:: Docker Container

   .. tab:: Windows

      .. code-block:: console

         docker run -v %cd%:/wd sodateam/metasyn create-meta wd/my_dataset.csv wd/my_gmf.json

   .. tab:: Unix or MacOS:

      .. code-block:: console

         docker run -v $(pwd):/wd sodateam/metasyn create-meta wd/my_dataset.csv wd/my_gmf.json

The ``create-meta`` command also takes one optional argument:

* ``--config [config-file]``: The filepath and name of a .toml configuration file that specifies distribution behavior. For example, if we want to set a column to be unique or to have a specific distribution, we can do so by specifying it in the configuration file. Information on how to use these files can be found in the :doc:`improve_synth` section.

.. admonition:: Generating a GMF file without a dataset

   See our :doc:`section<datafree>` for how to create a configuration file without using a dataset. In this case, you will not need to supply any ``[input]`` to the ``create-meta`` command.


Generating synthetic data
-------------------------
The ``synthesize`` subcommand combines the deserialization and generation steps in the pipeline into one, and allows you to generate a synthetic dataset from a previously saved MetaFrame (stored as GMF file). 

.. image:: /images/pipeline_cli_synthesize.png
   :alt: Creating Synthetic Data from a GMF file using the CLI
   :align: center

The ``synthesize`` command can be used as follows:

.. code-block:: bash

   metasyn synthesize [input] --output [output]

This will:

1. Read the GMF file from the `[input]` filepath
2. Deserialize it into a MetaFrame 
3. Generate synthetic data based on the metadata
4. Save the output data to a file at the `[output]` filepath

The ``synthesize`` command takes two positional arguments:

* ``[input]``: The filepath and name of the GMF file.
* ``[output]``: The Filepath and name of the desired synthetic data output file. The file extension determines the output format. Currently supported file types are ``.csv``, ``.feather``, ``.parquet``, ``.pkl`` and ``.xlsx``.

An example of how to use the ``synthesize`` subcommand is as follows:

.. tab:: Local Installation

   .. code-block:: console

      metasyn synthesize wd/my_gmf.json wd/my_synthetic_data.csv

.. tab:: Docker Container

   .. tab:: Windows

      .. code-block:: console

         docker run -v %cd%:/wd sodateam/metasyn synthesize wd/my_gmf.json wd/my_synthetic_data.csv

   .. tab:: Unix or MacOS:

      .. code-block:: console

         docker run -v $(pwd):/wd sodateam/metasyn synthesize wd/my_gmf.json wd/my_synthetic_data.csv



The ``synthesize`` command also takes two optional arguments:

- ``-n [rows]`` or ``--num_rows [rows]``: To generate a specific number of data rows.
- ``-p`` or ``--preview``: To preview the first six rows of synthesized data. This can be extremely useful for quick data validation without saving it to a file.

.. note::

   The ``output`` is required unless ``--preview`` is used.




Creating validation schemas
---------------------------

The ``schema`` subcommand generates a schema that describes the expected format of the GMF files. These can be used to validate GMF files before importing and loading them into a :obj:`MetaFrame<metasyn.metaframe.MetaFrame>`.

.. code-block:: console
   
   metasyn schema

It's also possible to include additional plugins in the validation schema, this can be done by passing in their names as space-seperated arguments:

.. code-block:: console
   
   metasyn schema plugin1 plugin2

To retrieve a list of all available plugins, you can use the ``--list`` or ``-l`` argument. This displays the available plugins:

.. code-block:: console
   
   metasyn schema --list

