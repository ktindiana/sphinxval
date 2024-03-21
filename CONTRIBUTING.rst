Developer Guidelines
====================

We are collaboratively developing SPHINX using a workflow inspired by
SunPy, which in turn was inspired by AstroPy, which in turn is
following git-based collaborative development practices used in many
projects.  The goal is make some effort to make a clean, intelligable
code history for the main branch sphinxval repository, and to leave
the messy reality of developing to the feature branches, which are
deleted when the contribution is merged into the main code base.  The
rough synopsis is:

* In GitHub, fork the main sphinxval repository, owned by Katie
* Clone your fork to your workstation
* Do not use your `main` branch for anything.
* Create a "feature branch" for your contribution, with an
  appropriate, descriptive name
* Make frequent commits, and always include a commit message
  following the commit message guidelines. Each commit should
  represent one logical set of changes. 
* Never merge (using git pull or git merge) changes from
  sphinxval/main into your feature branch.  This produces an
  undesireable merge commit.
* Keep your feature branch current with `git rebase`.  Do this
  frequently; the longer you wait the greater the chances that you
  have large conflicts to resolve.
* Test your new feature before making a pull request
* When your new feature looks good, push your feature branch to your
  personal repository and make a pull request
* The code maintainer will review your pull request, potentially
  requesting additional changes
* The code maintainer will potentially squash your commits to create
  a more compact history for the project.  Alternatively, you can do
  this yourself before making the pull request, for example using
  `git rebase -i` (interactive rebase)
* Once the pull request for your feature branch is approved, you
  should delete the feature branch.  Because of the potential squash
  done above, any further work on your feature branch could result in
  attempting to reintrodce your previous commits to the main
  repository, which is undesireable.  Better to start fresh.
* When you are ready to make a new contribution, use `git pull
  upstream main` to synchronize your main branch with the offical
  repository main, then make a new feature branch and repeat the
  above procedure.

For the sphinxval maintainters, follow the following guidlelines:

* Approve pull requests using the GitHub web interface
* Read the pull request commits.  Make a review and make comments on
  any line of code you do not understand or do not like.
* Request any needed changes or additions using pull request comments
* Ask contributer to rebase and push again if any merge conficts are
  found
* If the pull request contains only one or only a few straightforward
  and logical commits, use the normal Merge button
* If the pull request contains several non-essential commits, with
  messages like "bugfix" or "fixed typo", or if the several commits
  achieve a single logical goal, use the Squash and merge option
* The Rebase and merge option should not be used.  Indeed, it should
  not be necessary if the above guidelines are followed.
* If the pull request completes a new feature or fixes a bug reported
  in Issues, include a cross-reference in the pull request message.
  Close the relevant Issue when the pull request is merged.

Git Commands Used Infrequently
------------------------------

.. code-block:: sh

  # Set a reference for upstream (the source of your GitHub fork)
  git remote add upstream https://github.com/ktindiana/sphinxval

  # View remotes
  git remote -v

Git Commands Used Routinely
---------------------------

.. code-block:: sh
		
  # Create a new feature branch and switch to it
  # 1. Make sure the trunk is current
  git fetch upstream --tags

  # 2. Create the branch
  git branch my-new-feature

  # 3. Switch to the branch
  git switch my-new-feature

  # 2+3. Or Do the above two in one line
  git switch -c my-new-feature

  # Check what branch you are on at the moment
  git status
  On branch my-new-feature
  ...

  # Commit new code
  git commit

  # Update your personal GitHub with your new branch
  git push --set-upstream origin my-new-feature

  # At this point, a pull request on GitHub.com is possible
  # But let's assume you're not done yet, and meanwhile
  # progress has been made on the trunk

  # Get up-to-date with the upstream trunk
  # 1. Verify that all your work is committed
  git status

  # 2. fetch the new stuff from upstream
  git fetch upstream main

  # 3. Rebase your current work to the head of upstream main
  git rebase upstream/main
  # if there were conflicts, resolve them

  # Push your updates to your personal repository.
  # The rebase operation neccessitates force, since you have
  # rewritten your history to include the changes from upstream
  git push --force

  # Since the pull request was accepted you can see your changes in
  # the trunk
  git switch main
  git pull upstream main
  git log

  # After the pull request is accepted you should delete your branch.
  # The commits you offered in the pull request may be squashed to a #
  # simpler commit, and in that case may never use this branch again as
  # they have different histories.
  git branch -D my-new-feature
  
  # If for some reason you want to examine the history of your
  # deleted branch:

  # 1. Find the SHA1 of your old branch, identified by your last commit
  #    message on it
  git reflog --no-abbrev

  # 2. Copy/paste into a git log command
  git log <SHA1>


References
==========

* `SunPy newcomers`_
* `SunPy maintainer`_
* `AstroPy development`_
* `AstroPy git example`_
* `Altassian git rebase`_
* `Altassian merge vs. rebase`_
* `GitHub merge pull requests`_
* `git branching and rebasing`_

.. _SunPy newcomers: https://docs.sunpy.org/en/latest/dev_guide/contents/newcomers.html#newcomers
.. _SunPy maintainer: https://docs.sunpy.org/en/latest/dev_guide/contents/maintainer_workflow.html
.. _AstroPy development: https://docs.astropy.org/en/latest/development/workflow/development_workflow.html
.. _AstroPy git example: https://docs.astropy.org/en/latest/development/workflow/git_edit_workflow_examples.html#astropy-fix-example
.. _Altassian git rebase: https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase
.. _Altassian merge vs. rebase: https://www.atlassian.com/git/tutorials/merging-vs-rebasing
.. _GitHub merge pull requests: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges
.. _git branching and rebasing: https://git-scm.com/book/en/v2/Git-Branching-Rebasing
