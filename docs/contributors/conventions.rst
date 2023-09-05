.. _coding conventions:

##################
Coding conventions
##################

Python
------

.. dropdown:: TL;DR

    We use ``black`` to format our code, ``flake8`` to lint it, and ``isort`` to sort our imports.

    .. important::

        ``flake8`` and ``isort`` are not mandatory, but ``black`` is.

    .. note::

        For ``flake8`` to play nicely with ``black``, you need to use a couple extra flags:

        .. code-block:: bash

            flake8 --max-line-length=88 --ignore=E203 path/to/file.py

        ``--max-line-length=88`` is to match ``black``'s default line length.

        ``--ignore=E203`` is to avoid conflicts between ``black`` and ``flake8``. ``black`` will add a space before ``:`` in slices, ``flake8`` will complain about it. This flag tells ``flake8`` to ignore this particular error.

Black
~~~~~

Black is a code formatter. It ensures we all write code in the same style.

For instance, this is valid Python code:

.. code-block:: python

    a=32000
    b = {
    'a': 1, "b":2,}
    function (a=1,b = [1,2,3],c=3,)

Black will reformat it to:

.. code-block:: python

    a = 32000
    b = {"a": 1, "b": 2}
    function(a=1, b=[1, 2, 3], c=3)

Some choices are optimized for readability (for instance single quotes are replaced by double quotes). Others are just a matter of subjective taste (for instance the number of spaces around the equal sign).

That being said, what matters is the consistency. Black is a tool that ensures we all write code in the same style. This will also make PRs clearer, easier to review, and less prone to conflicts. Plus, it will save you time: you don't have to think about formatting anymore, Black does it for you.

There are a number of ways to work with Black. It can be an extension in your IDE (PyCharm, VSCode, etc.), a pre-commit hook, or a command line tool. The latter is the simplest to set up and use:

.. code-block:: bash

    pip install black
    black path/to/file.py

Black will reformat the file in place. If you want to see the changes before applying them, use the ``--diff`` flag.

.. caution::

    Your file needs to be valid Python for ``black`` to run. If you have a Syntax Error in your code, ``black`` will fail and it may look like your IDE extension "is not working". It is trying to, but it cannot. Fix the Syntax Error first, then run ``black`` again.

Check out the `Black documentation <https://black.readthedocs.io/en/stable/>`_ for more information.

Flake8
~~~~~~

Flake8 is a code linter. It will help you, just like ``black``, with writing good, consistent code. It will also help you avoid common pitfalls and mistakes like undefined variables, unused imports, etc.

Flake8 is a command line tool. It can be installed with:

.. code-block:: bash

    pip install flake8
    flake8 --max-line-length=88 --ignore=E203 path/to/file.py


.. note::

        For ``flake8`` to play nicely with ``black``, you need to use a couple extra flags:

        .. code-block:: bash

            flake8 --max-line-length=88 --ignore=E203 path/to/file.py

        ``--max-line-length=88`` is to match ``black``'s default line length.

        ``--ignore=E203`` is to avoid conflicts between ``black`` and ``flake8``. ``black`` will add a space before ``:`` in slices, ``flake8`` will complain about it. This flag tells ``flake8`` to ignore this particular error.

Most IDEs will also let you use Flake8 as an extension to have feedback as you code. Ask Google about your particular IDE, you're very likely not the first one.

Check out the `Flake8 documentation <https://flake8.pycqa.org/en/latest/>`_ for more information.

Isort
~~~~~

Isort is a tool that sorts your imports. It will make sure that:

-   standard library imports are on top
-   third-party imports are in the middle
-   local imports are at the bottom

It will also sort the imports alphabetically, and group them by package.

Isort can be installed with:

.. code-block:: bash

    pip install isort
    isort path/to/file.py

Again, this is all configurable. You can read more about it in the `isort documentation <https://pycqa.github.io/isort/>`_.