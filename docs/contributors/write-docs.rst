##########################
How to write documentation
##########################

In this document you will learn how to write good, informative, pretty and actionable documentation.

It's not hard !

First, let's look at an example:


``.rst`` Example
----------------

This is a short example on how to use the library.

.. code-block:: python

    import gflownet

    # Create a new instance of the class
    my_class = gflownet.MyClass()

    # Call a method of the class
    my_class.do_something()

    # Get an attribute of the class
    my_class.attribute

.. note::

    This is a note. You can use it to add notes to your documentation.

.. warning::

    This is a warning. You can use it to add warnings to your documentation.

Cool features:

Reference to a class: :class:`gflownet.proxy.crystals.dave.DAVE` (long), or another :class:`~gflownet.gflownet.GFlowNetAgent` or to a method: :meth:`~gflownet.gflownet.GFlowNetAgent.trajectorybalance_loss` or to an external function :func:`torch.cuda.synchronize()` (this <- needs to be listed in ``docs/conf.py:intersphinx_mapping``).

An actual tutorial on ``.rst``: `ReStructured Text for those who know Markdown <https://docs.open-mpi.org/en/v5.0.x/developers/rst-for-markdown-expats.html#hyperlinks-to-urls>`_

.. important::

    Check out this documentation for more on the specific so-called *admonitions* like
    the "note", "warning", "important", etc. coloured boxes in this document:
    `Furo theme documentation <https://pradyunsg.me/furo/reference/admonitions/#supported-types>`_

.. attention::

    ReStructured Text is a bit more complicated than Markdown, but it's worth it. **One common mistake** is to forget that spaces and new lines matter in ``.rst``. For example, the following will not work:

    .. code-block::

        .. note::
        This is a note.

    But this will

    .. code-block::

        .. note::

            This is a note.

    Same goes for whitespaces: ``.. code-block::`` ✅ ``..code-block::`` ❌.


.. todo::

    Improving the documentation: `Recommendations for Sphinx plugins <https://pradyunsg.me/furo/recommendations/>`_.
