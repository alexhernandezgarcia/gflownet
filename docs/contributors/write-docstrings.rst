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

    There are two main conventions for writing docstrings: `Numpy <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy>`_ and `Google <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google>`_. The latter (Google) is more popular in the Python community, but the former (Numpy) is more popular in the scientific Python community. We recommend the Google style as it is more readable, easier to write and `used by PyTorch <https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#PReLU>`_, but both are fine.

First, start with a one-line summary that gives an overview of the functionality, in Google docstring style:

.. code-block:: python

    def function(arg1, arg2):
        """Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value

        """
        return True

The one-line summary should be short and concise. It should not mention the function's arguments or return values, since they are described in more detail in the ``Args`` and ``Returns`` sections.

The first line should be followed by a blank line. The docstring should describe the function's behavior, not its implementation, *i.e.* **what** it does, not **how**.

The ``Args`` and ``Returns`` sections should list each argument and return value, along with a description of its type and purpose.

The ``Returns`` section should also describe the type of the return value, if applicable.

For example:

.. code-block:: python

    def function(arg1, arg2):
        """Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value

        """
        return True

If there are any exceptions that the function can raise, they should be listed in a ``Raises`` section, along with a description of the circumstances under which they are raised. For example:

.. code-block:: python

    def function(arg1, arg2):
        """Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value

        Raises:
            ValueError: If arg1 is equal to arg2

        """
        if arg1 == arg2:
            raise ValueError('arg1 must not be equal to arg2')
        return True

If there are any examples of how to use the function, they should be listed in an ``Examples`` section. For example:

.. code-block:: python

    def function(arg1, arg2):
        """Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value

        Raises:
            ValueError: If arg1 is equal to arg2

        Notes:
            Do not use this function if arg1 is equal to arg2.

        Examples:
            >>> function(1, 'a')
            True

        """
        if arg1 == arg2:
            raise ValueError('arg1 must not be equal to arg2')
        return True

The ``Examples`` section should contain code that can be executed by the user to demonstrate how to use the function. The code should be indented by four spaces.

If there are any references that the user should be aware of, they should be listed in a ``References`` section. For example:

.. code-block:: python

    def function(arg1, arg2):
        """Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value

        Raises:
            ValueError: If arg1 is equal to arg2

        Notes:
            Do not use this function if arg1 is equal to arg2.

        Examples:
            >>> function(1, 'a')
            True

        References:
            - https://example.com
            - https://example.com

        """
        if arg1 == arg2:
            raise ValueError('arg1 must not be equal to arg2')


So many rules, how can I check?
-------------------------------

There are many rules to follow when writing docstrings. How can you check that you are following them all?

There's an easy way to check: use a tool called `pydocstyle <https://www.pydocstyle.org/en/stable/>`_.

.. code-block:: bash

    $ pip install pydocstyle

``pydocstyle`` checks that your docstrings follow the `Google docstring style <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google>`_.

.. code-block:: bash

    $ pydocstyle --convention=google --add-ignore=D212 gflownet/my_module.py
    $ pydocstyle --convention=google --add-ignore=D212 gflownet/
