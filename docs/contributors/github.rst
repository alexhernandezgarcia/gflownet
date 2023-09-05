################
Github etiquette
################

How we work together.

Where to learn
--------------

If you are new to ``git`` or Github, here are some resources to get you started:

- [Git tutorial](https://www.atlassian.com/git)
- [Github help](https://help.github.com/)
- [Github tutorial](https://docs.github.com/en/get-started/quickstart/hello-world)
- [How to contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [A successful branching mode](https://nvie.com/posts/a-successful-git-branching-model/)

Typical workflow
----------------

1. Create a branch with a meaningful name like ``fix-xyz-victor`` or ``feature-abc-victor``.
2. Commit to this branch as often as you want.
    1. Commit messages should be meaningful and concise.
    2. Try to commit in a granular way, not too many changes at once.
3. As soon as you have a couple commits, push your branch to Github.
4. As soon as you have pushed, create a **draft** pull request.

   a. The PR's title should be self-evident
   b. As long as the PR is a draft, you can also add ``[WIP]`` to the title (yes it's redundant, but it's also more visible).
   c. The *draft* / *WIP* status means everyone knows this is work in progress, they won't expect it to work or to be finished. But it means **they can see what you're working on**.

5. Make sure that you have written all the :doc:`appropriate docstrings </contributors/write-docstrings>`.
6. Make sure the PR's comment is complete. Emphasize major changes, especially breaking ones like new dependencies and provide examples of how to use a new feature.
7. Before you are done, run ``black``, ``isort`` and ``flake8`` on your code (see our :doc:`conventions </contributors/conventions>`).
8. When you are done, remove the ``[WIP]`` from the PR's title and the *Draft* status.
9. Wait for the CI to pass.
10. Ask for reviewers


.. image:: /_static/images/github-pr.png
   :align: center
   :alt: Github PR
