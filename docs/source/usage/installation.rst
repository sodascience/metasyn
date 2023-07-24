Installation Guide
==================

This document will guide you through the process of installing the MetaSynth package. 

.. note:: 
	If you wish to try MetaSynth out in an online Notebook, before installing it locally, check out our :doc:`/usage/interactive_tutorials`.


Step 1: Python Installation
---------------------------

MetaSynth requires Python 3.7 or higher. If you don't have Python installed or your Python version is lower than 3.7, you will need to install or upgrade Python.

If you're not familiar with Python or you've never installed it before, refer to the `Python Guide <https://docs.python-guide.org/starting/installation/>`_ for detailed instructions. 

Step 2: Installing PIP
----------------------

MetaSynth uses `PyPI <https://pypi.org/project/metasynth/>`_ for distribution, which requires pip, the Python package installer, to install. If you haven't installed pip, refer to the `Pip Installation Guide <https://pip.pypa.io/en/stable/installation/>`_ for instructions.

Step 3: Installing MetaSynth
----------------------------

There are two ways to install MetaSynth, you can either download the latest official version from PyPI or directly from our GitHub repository.

.. tab:: PyPI

	.. code-block:: console

		pip install metasynth

.. tab:: GitHub

	.. code-block:: console

		pip install git+https://github.com/sodascience/meta-synth.git
		
Choose the method that best suits your needs. If you're unsure, using PyPI is the simplest and most straightforward method.

Step 4: Verifying Installation
-------------------------------

To ensure MetaSynth has been successfully installed, run the following command in a Python console:

.. code-block:: python

	import metasynth

If the command runs without any errors, you have successfully installed MetaSynth.

Optional: Creating a Virtual Environment
----------------------------------------

Although not necessary, it's a good practice to create a virtual environment for each Python project to manage dependencies. You can create a virtual environment using either venv (built into Python) or a similar tool like `Conda <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_. If you're unfamiliar with this concept, refer to the `Python Virtual Environments Guide <https://docs.python-guide.org/dev/virtualenvs/>`_.

To create a virtual environment using venv:

.. code-block:: console

	python3 -m venv metasynth-env

To activate the environment: 

.. tab:: Windows

	.. code-block:: console

		metasynth-env\Scripts\activate

.. tab:: Unix or MacOS:

	.. code-block:: console

		source metasynth-env/bin/activate

With the virtual environment activated, you can then install MetaSynth as described in Step 3. To exit the virtual environment, simply type `deactivate` in your console.

