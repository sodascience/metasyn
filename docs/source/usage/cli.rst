Command-line Interface
======================
Metasyn provides a command-line interface (CLI) for accessing core functionality without needing to write any Python code.

The CLI currently has two subcommands. The ``synthesize`` subcommand, which allows you to **generate synthetic data** from any ``GMF file``, and the ``schema`` subcommand, which allows you to **create validation schemas** for GMF files.


At any point, you can also use the help command to get more information about the CLI and its subcommands:

.. code-block:: console

   metasyn --help


Accessing the CLI
-----------------
If you have installed the main metasyn package, the CLI should be available automatically. You can find instructions on how to install metasyn in the :doc:`installation` section of the documentation.

Alternatively, the CLI can be accessed through a Docker container, allowing you to run metasyn in an isolated environment without installing the package on your system. This can be useful, for example, when trying out metasyn without affecting your existing Python environment.

Here's how you can use Docker to access Metasyn's CLI:

1. **Install Docker:** If Docker isn't already set up on your machine, please follow the instructions on `Docker's official website <https://docs.docker.com/get-docker/>`_.

2. **Pull the metasyn Docker Image:** After successfully installing Docker, you can download the metasyn Docker image from Docker Hub using the following command.

   .. code-block:: console

      docker pull sodateam/metasyn

3. **Run metasyn's CLI via Docker:** Once the Docker image is downloaded, you can use the following command to run a metasyn CLI command within a Docker container (in this case ``--help``), and simultaneously set the working directory (which we denote as `wd` in this case).

   .. tab:: Windows

      .. code-block:: console

         docker run -v %cd%:/wd sodateam/metasyn --help

   .. tab:: Unix or MacOS:

      .. code-block:: console

         docker run -v $(pwd):/wd sodateam/metasyn --help


The Metasyn CLI should now be up and running within the Docker container and ready for use!

.. note:: 
   You can also specify which metasyn version to use in docker, by adding a tag to the docker image. For example, to use version 0.6, you can use the following command:

   .. tab:: Installing a specific version
      
      .. code-block:: console

         docker pull sodateam/metasyn:v0.6

   .. tab:: Using a command on a specific version

      .. tab:: Windows

         .. code-block:: console

            docker run -v %cd%:/wd sodateam/metasyn:v0.6 --help

      .. tab:: Unix or MacOS:

         .. code-block:: console

            docker run -v $(pwd):/wd sodateam/metasyn:v0.6 --help


Generating Synthetic Data
-------------------------
The ``synthesize`` subcommand combines the :doc:`deserialization </usage/exporting_metaframes>` and :doc:`generation </usage/generating_synthetic_data>` steps in the pipeline into one, and allows you to generate a synthetic dataset from a previously exported MetaFrame (stored as GMF file). 

.. image:: /images/pipeline_cli.png
   :alt: CLI in the metasyn pipeline
   :align: center


The ``synthesize`` command can be used as follows:

.. code-block:: bash

   metasyn synthesize [input] [output]

This will:

1. Read the GMF file
2. Deserialize it into a MetaFrame
3. Generate synthetic data based on the metadata
4. Save the output data to a file

The ``synthesize`` command takes two positional arguments:

* ``[input]``: The filepath and name of the GMF file.
* ``[output]``: The Filepath and name of the desired synthetic data output file. The file extension determines the output format. Currently supported file types are ``.csv``, ``.feather``, ``.parquet``, ``.pkl`` and ``.xlsx``.

For example: 

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


Creating Validation schemas
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





