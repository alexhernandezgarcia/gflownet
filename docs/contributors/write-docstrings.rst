.. _write docstrings:

#######################
How to write docstrings
#######################

.. note::

    This section is still under construction. It will evolve as the code base matures

Basics
------

In the :doc:`previous section </contributors/write-documentation>`, we learned about the syntax of ``rst`` to write documentation and docstrings. In this section, we'll look at how to write the content of a good docstring.

.. caution::

    This section is still under construction. It will be finished soon.

.. note::

    There are two main conventions for writing docstrings: `Numpy <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy>`_ and `Google <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google>`_. The latter (Google) is more popular in the Python community, but the former (Numpy) is more popular in the scientific Python community.

We use the |numpy-docs|_ **convention** for the ``gflownet`` project.

First, start with a one-line summary that gives an overview of the functionality, in Numpy docstring style:

.. caution::

     *Unlike* Markdown, code rendering like ``this`` in docstrics uses **two** backticks ``````, not one `````.

.. code-block:: python

    def function(arg1, arg2):
        """Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        bool
            Description of return value
        """
        return True

The one-line summary should be short and concise. It should not mention the function's arguments or return values, since they are described in more detail in the ``Parameters`` and ``Returns`` sections.

The first line should be followed by a blank line. The docstring should describe the function's behavior, not its implementation, *i.e.* **what** it does, not **how**.

The ``Parameters`` and ``Returns`` sections should list each argument and return value, along with a description of its type and purpose.

The ``Returns`` section should also describe the type of the return value, if applicable.

If there are any exceptions that the function can raise, they should be listed in a ``Raises`` section, along with a description of the circumstances under which they are raised. For example:

.. code-block:: python

    def function(arg1, arg2):
        """Summary line.

        Extended description of function.

        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        bool
            Description of return value

        Raises
        ------
        ValueError
            If arg1 is equal to arg2
        """
        if arg1 == arg2:
            raise ValueError('arg1 must not be equal to arg2')
        return True

If there are any examples of how to use the function, they should be listed in an ``Examples`` section. Separate groups of examples with empty lines. For instance:

.. code-block:: python

    def function(arg1, arg2):
        """Summary line.

        Extended description of function.

        Examples
        --------

        >>> function(1, 'a')
        True
        >>> function(1, 2)
        True

        >>> function(1, 1)
        Traceback (most recent call last):
            ...


        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        bool
            Description of return value

        Raises
        ------
        ValueError
            If arg1 is equal to arg2
        """
        if arg1 == arg2:
            raise ValueError('arg1 must not be equal to arg2')
        return True

The ``Examples`` section should contain code that can be executed by the user to demonstrate how to use the function.

Importantly, if you need maths in your docstrings, you can use LaTeX to write equations between single `$` for inline equations and between double `$$` for block equations.

.. important::

    If you want to use LaTeX in your docstrings, you need to use raw strings ``r"..."`` for ``\`` to be appropriately interpreted. Alternatively you must double them ``\\``. For example:

    .. code-block:: python

        r"""
        Summary line with inline $1+1=3$ math.

        $$
        \int_0^1 x^2 dx = \frac{1}{3}
        $$

        ...
        """"

    Or

    .. code-block:: python

        """
        Summary line with inline $1+1=3$ math.

        $$
        \\int_0^1 x^2 dx = \\frac{1}{3}
        $$

        ...
        """

    This is because the `r` before the triple quotes tells Python that the string is a raw string, which means that backslashes are treated as literal backslashes and not as escape characters.

.. _write docstrings-extended:

Full Example
------------

The following code renders as: :py:func:`gflownet.utils.common.example_documented_function`.

.. code-block:: python

    def example_documented_function(arg1, arg2):
        r"""Summary line: this function is not used anywhere, it's just an example.

        Extended description of function from the docstrings tutorial :ref:`write
        docstrings-extended`.

        Refer to

        * functions with :py:func:`gflownet.utils.common.set_device`
        * classes with :py:class:`gflownet.gflownet.GFlowNetAgent`
        * methods with :py:meth:`gflownet.envs.base.GFlowNetEnv.get_action_space`
        * constants with :py:const:`gflownet.envs.base.CMAP`

        Prepenend with ``~`` to refer to the name of the object only instead of the full
        path -> :py:func:`~gflownet.utils.common.set_device` will display as ``set_device``
        instead of the full path.

        Great maths:

        .. math::

            \int_0^1 x^2 dx = \frac{1}{3}

        .. important::

            A docstring with **math** MUST be a raw Python string (a string prepended with
            an ``r``: ``r"raw"``) to avoid backslashes being treated as escape characters.

            Alternatively, you can use double backslashes.

        .. warning::

            Display a warning. See :ref:`learn by example`. (<-- this is a cross reference,
            learn about it `here
            <https://www.sphinx-doc.org/en/master/usage/referencing.html#ref-rolel>`_)


        Examples
        --------
        >>> function(1, 'a')
        True
        >>> function(1, 2)
        True

        >>> function(1, 1)
        Traceback (most recent call last):
            ...

        Notes
        -----
        This block uses ``$ ... $`` for inline maths -> $e^{\frac{x}{2}}$.

        Or ``$$ ... $$`` for block math instead of the ``.. math:`` directive above.

        $$\int_0^1 x^2 dx = \frac{1}{3}$$


        Parameters
        ----------
        arg1 : int
            Description of arg1
        arg2 : str
            Description of arg2

        Returns
        -------
        bool
            Description of return value
        """
        if arg1 == arg2:
            raise ValueError("arg1 must not be equal to arg2")
        return True


So many rules, how can I check?
-------------------------------

There are many rules to follow when writing docstrings. How can you check that you are following them all?

There's an easy way to check: use a tool called `pydocstyle <https://www.pydocstyle.org/en/stable/>`_.

.. code-block:: bash

    $ pip install pydocstyle

``pydocstyle`` checks that your docstrings follow the `Numpy docstring style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

.. code-block:: bash

    $ pydocstyle --convention=numpy --add-ignore=D212 gflownet/my_module.py
    $ pydocstyle --convention=numpy --add-ignore=D212 gflownet/

..
    This is a comment.

    LINKS SECTION ⬇️

.. |numpy-docs| replace:: **Numpy**
.. _numpy-docs: https://numpydoc.readthedocs.io/en/latest/format.html
