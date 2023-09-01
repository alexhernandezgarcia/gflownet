##########################
How to write documentation
##########################

In this document you will learn how to write good, informative, pretty and actionable documentation.

It's not hard !

Overview
--------

There are two major types of documentation:

1. **docstrings**: your code's docstrings will be automatically parsed by the documentation sofware (`Sphinx <https://www.sphinx-doc.org>`_, more in `generating the documentation`_).
2. **Manual** documentation such as this document.

**Both** are written in `ReStructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ (``.rst``) format.

Some of the great features of using Sphinx is to be able to automatically generate documentation from your code's docstrings, and to be able to link to other parts of the documentation.

For instance: :meth:`~gflownet.gflownet.GFlowNetAgent.trajectorybalance_loss` or to an external function :func:`torch.cuda.synchronize()`.

Learn by example
^^^^^^^^^^^^^^^^

The next section will introduce many of the cool features of ``.rst`` + Sphinx + plugins.

Click on "*Code for the example*" to look at the ``.rst`` code that generated what you are reading.

.. tab-set::

    .. tab-item:: Full-fleshed ``.rst`` example

        .. include:: example.rst

    .. tab-item:: Code for the example

        .. literalinclude:: example.rst
            :language: rst

FAQ
---

.. dropdown:: How do I create new manual documentation files.

    - Create a new ``.rst`` file in the ``docs/`` folder
    - List it in ``docs/index.rst`` file under the ``.. toctree::`` directive
    - **Or** create a subfolder in ``docs/`` with an ``index.rst`` file.
        - This is useful for grouping documentation files together.
        - ``docs/{your_subfolder}/index.rst`` should contain a ``.. toctree::`` directive listing the files in the subfolder.
        - It should also be listed in the ``docs/index.rst`` under the ``.. toctree::`` directive to appear on the left handside of the documentation.

    You can look at the |contributors|_ folder for an example.

.. dropdown:: How do I document a sub-package like :py:mod:`gflownet.proxy.crystals`?

    Just add a docstring at the top of the ``__init__.py`` file of the sub-package:

    .. code-block:: python

        """
        This is the docstring of the sub-package.

        It can contain any kind of ``.rst`` syntax.

        And refer to its members: :meth:`~gflownet.proxy.crystals.crystal.Stage`

        .. note::
            This is a note admonition.

        """

    You can similarly document a **module** by adding a docstring at the top of the file

.. dropdown:: How do I document a module varaible?

    Add a docstring **below** the variable to document like

    .. code-block:: python

        MY_VARIABLE = 42
        """
        This is the docstring of the variable.

        Again, It can contain any kind of ``.rst`` syntax.
        """

.. dropdown:: How do I document a class?

    Currently, ``autoapi`` is setup to consider the documention of a class to be the same as the documentation for the ``__init__`` method of the class.

    This can be changed by changing the ``autoapi_python_class_content = "init"`` configuration variable in ``docs/conf.py``. See `AutoAPI <https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_python_class_content>`_ for more details.

.. dropdown:: Where is the documentation for those advanced features? (tabs, dropdowns etc.)

    - `Sphinx-Design <https://sphinx-design.readthedocs.io/en/furo-theme/>`_ contains many components you can re-use
    - We use the `Furo <https://pradyunsg.me/furo/reference/admonitions/>`_ theme, you'll find the list of available *admonitions* there

.. dropdown:: What plugins are used to make the documentation?

    - `Todo <https://www.sphinx-doc.org/en/master/usage/extensions/todo.html>`_ enables the ``.. todo::`` admonition
    - `Intersphinx mapping <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ enables linking to external documentation like in the ``torch.cuda.synchronize()`` example above
    - `AutoAPI <https://autoapi.readthedocs.io/>`_ enables the automatic generation of documentation from docstrings & package structure
    - `Sphinx Math Dollar <https://www.sympy.org/sphinx-math-dollar/>`_ enables the ``$...$`` math syntax
    - `Sphinx autodoc type ints <https://github.com/tox-dev/sphinx-autodoc-typehints>`_ enables more fine-grained control on how types are displayed in the docs
    - `MyST <https://myst-parser.readthedocs.io/en/latest/intro.html>`_ enables the parsing of enhanced Markdown syntax in the ``.rst`` documentation.
    - `Hover X Ref <https://sphinx-hoverxref.readthedocs.io/en/latest/index.html>`_ Enables tooltips to display contents on the hover of links
    - `Napoleon <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ enables the parsing of Google-style docstrings

Generating the documentation
----------------------------


..
    This is a comment.

    LINKS SECTION ⬇️

.. |contributors| replace::  ``docs/contributors/``
.. _contributors: https://github.com/alexhernandezgarcia/gflownet/tree/master/docs/contributors
