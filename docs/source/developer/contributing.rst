Contributing to metasyn
=========================

Thank you for your interest in contributing to metasyn! We greatly appreciate any contributions that can help improve the package and make it more useful for the community. Whether it's bug fixes, new features, or documentation enhancements, your efforts will be highly valued.

Feedback, suggestions and issues:
---------------------------------
If you encounter a bug or have a feature request, you can report it in the `issue tracker <https://github.com/sodascience/metasyn/issues>`_. Detailed bug reports and well-defined feature requests are highly appreciated. Additionally, you can help us by leaving suggestions or feedback on how to enhance ``metasyn`` on the project's `GitHub repository <https://github.com/sodascience/metasyn>`_. More information on getting in touch with us can be found on our :doc:`contact page </about/contact>`.

.. image:: https://img.shields.io/badge/GitHub-blue?logo=github&link=https%3A%2F%2Fgithub.com%2Fsodascience%2Fmetasyn
   :alt: GitHub Repository Button
   :target: https://github.com/sodascience/metasyn

.. image:: https://img.shields.io/badge/GitHub-Issue_Tracker-blue?logo=github&link=https%3A%2F%2Fgithub.com%2Fsodascience%2Fmetasyn%2Fissues   
   :alt: GitHub Issue Tracker Button
   :target: https://github.com/sodascience/metasyn/issues

Contribute directly to the code or documentation:
--------------------------------------------------
You can contribute to the project by writing code to fix bugs, implement new features, or improve existing functionality. Alternatively, you can help improve the documentation by fixing errors, adding examples, or clarifying instructions.
The following shows a basic overview of the steps needed to do so.

.. warning::
    1. The code examples provided below are for illustrative purposes. Make sure to replace them with the actual commands relevant to your local environment. 
    2. This is just a brief overview of the steps needed to get started, for specifics on Git or GitHub, check out the `Git documentation <https://git-scm.com/doc>`_ and the `Github documentation <https://docs.github.com/en>`_.  

1. **Fork the Project**: Start by forking the `official metasyn repository <https://github.com/sodascience/metasyn>`_ to your GitHub account. This will create a copy of the repository under your account, allowing you to freely make changes.
2. **Create a Branch**: Before making changes, create a new branch with a descriptive name. This helps maintain clarity and organization throughout the contribution process:

   .. code-block:: shell

       git checkout -b feature/AmazingFeature

   Replace ``AmazingFeature`` with a short and descriptive name of your feature or fix.

3. **Make Your Changes**: Implement the changes and improvements you wish to contribute. Ensure that your code adheres to the project's coding style and guidelines.

4. **Commit Your Changes**: Commit your changes with a clear and concise message describing the modifications you made:

   .. code-block:: shell

       git add .
       git commit -m 'Add some AmazingFeature'
5. **Push to Your Branch**: Once your branch is up-to-date, push your changes to your GitHub repository:

   .. code-block:: shell

       git push origin feature/AmazingFeature
6.  **Open a Pull Request**: After pushing your changes to your GitHub repository, you can proceed to open a pull request (PR) to propose your changes for inclusion in the official ``metasyn`` repository. Navigate to your forked repository on GitHub and click on the "Compare & pull request" button for the branch you want to merge. Provide a clear title and description for your PR, detailing the changes you made and any relevant context.
7.  **Review Process**: The project maintainers will review your PR, providing feedback and suggestions if needed. Please be patient during the review process, as it may take some time to thoroughly evaluate your contribution. Once your PR is approved and any requested changes are addressed, the project maintainers will merge your changes into the official repository.
8.  **Congratulations**! You have successfully contributed to metasyn!


Running local tests
-------------------
When a pull request is created, GitHub automatically runs `a series of tests <https://github.com/sodascience/metasyn/actions>`_ on the code to ensure it meets the projects standards and does not introduce any errors. You can run these tests locally to ensure your code passes before opening a pull request, using the `Tox <https://tox.wiki/>`_ package. 

To do so, first install Tox to your environment following the `Tox installation guide <https://tox.wiki/en/4.11.3/installation.html>`_. Then simply run the ``tox`` on the ``metasyn`` root directory to run all the tests. If you want to run a specific test, you can do so by specifying a (list of) test environment(s), e.g. ``tox -e ruff`` or ``tox -e ruff,pylint,pydocstyle,mypy,sphinx,pytest,nbval``. The available test environments can be found in the `pyproject.toml <https://github.com/sodascience/metasyn/blob/main/pyproject.toml>`_ file, under the ``[tool.tox]`` section.




Maintaining the package
-----------------------
Our GitHub Wiki contains a guide on how to maintain the package. You can find it `here <https://github.com/sodascience/metasyn/wiki>`_.

Code of Conduct
---------------
We expect all contributors to adhere to the Code of Conduct found on our `Github page <https://github.com/sodascience/metasyn/blob/main/.github/CODE_OF_CONDUCT.md>`_.