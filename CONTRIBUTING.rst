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



References
==========

 * SunPy_newcomers_
 * SunPy_maintainer_
 * AstroPy_development_
 * AstroPy_git_example_
 * Altassian_git_rebase_
 * Altassian_merge_vs_rebase_
 * GitHub_merge_pull_requests_
 * git_branching_rebasing_

.. _SunPy_newcomers: https://docs.sunpy.org/en/latest/dev_guide/contents/newcomers.html#newcomers
.. _SunPy_maintainer: https://docs.sunpy.org/en/latest/dev_guide/contents/maintainer_workflow.html
.. _AstroPy_development: https://docs.astropy.org/en/latest/development/workflow/development_workflow.html
.. _AstroPy_git_example: https://docs.astropy.org/en/latest/development/workflow/git_edit_workflow_examples.html#astropy-fix-example
.. _Altassian_git_rebase: https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase
.. _Altassian_merge_vs_rebase: https://www.atlassian.com/git/tutorials/merging-vs-rebasing
.. _GitHub_merge_pull_requests: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges
.. _git_branching_rebasing: https://git-scm.com/book/en/v2/Git-Branching-Rebasing
