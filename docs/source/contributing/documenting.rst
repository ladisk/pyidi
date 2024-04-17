.. _documenting-label:

Documenting the code
====================

Requirements
^^^^^^^^^^^^

* *Sphinx* ::

    pip install sphinx


Automatic code documentation with autodoc
-----------------------------------------

The Sphinx autodoc_ extension automatically includes our Python modules documentation in the generated Sphinx documentation. 

.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html


Addind a new module documentaiton to the build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is most likely not practical to iclude all of our apps Python modules in the build. Only the modules and classes with the most extensive documentation (docstrings), and the ones where most developer updates are expected should be included.

To add another module's documentation to the build, add a new entry to the ``doc/code/modules.rst`` file, with the correct relative Python path to the module. For example, the ``views.py`` documentation is included by::

        Tools
        -----
        .. automodule:: pyidi.tools
            :members:

For more information, see the autodoc_ documentation.


Docstring style: reStructuredText
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* default docstring style in PyCharm_
* ``"autoDocstring.docstringFormat": "sphinx"`` in `VSCode Python Docstring extenion`_

.. _PyCharm: https://www.jetbrains.com/help/pycharm/python-integrated-tools.html
.. _`VSCode Python Docstring extenion`: https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring

Example_:

.. code-block:: python

  def function1(self, arg1, arg2, arg3):
    """returns (arg1 / arg2) + arg3

    This is a longer explanation, which may include math with latex syntax
    :math:`\\alpha`.
    Then, you need to provide optional subsection in this order (just to be
    consistent and have a uniform documentation. Nothing prevent you to
    switch the order):

      - parameters using ``:param <name>: <description>``
      - type of the parameters ``:type <name>: <description>``
      - returns using ``:returns: <description>``
      - examples (doctest)
      - seealso using ``.. seealso:: text``
      - notes using ``.. note:: text``
      - warning using ``.. warning:: text``
      - todo ``.. todo:: text``

    **Advantages**:
      - Uses sphinx markups, which will certainly be improved in future
        version
      - Nice HTML output with the See Also, Note, Warnings directives


    **Drawbacks**:
      - Just looking at the docstring, the parameter, type and  return
        sections do not appear nicely

    :param arg1: the first value
    :param arg2: the first value
    :param arg3: the first value
    :type arg1: int, float,...
    :type arg2: int, float,...
    :type arg3: int, float,...
    :returns: arg1/arg2 +arg3
    :rtype: int, float

    :Example:

    >>> import template
    >>> a = template.MainClass1()
    >>> a.function1(1,1,1)
    2

    .. note:: can be useful to emphasize
        important feature
    .. seealso:: :class:`MainClass2`
    .. warning:: arg2 must be non-zero.
    .. todo:: check that arg2 is non zero.
    """

    return arg1/arg2 + arg3

.. _Example: https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html