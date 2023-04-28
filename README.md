# OpenSCM-Calibration-examples

Long-running examples using OpenSCM-Calibration

To do:

- tidy up the below
- Makefile/install/run instructionss

## Development

Comments on the below welcome, it is a work in progress

In this repository, we don't currently store outputs in the notebooks (more on this below). The reason is that outputs can quickly become bloated and storing the outputs discourages re-running i.e. testing.

Having looked around, we haven't found a good solution that allows us to run our notebooks once as part of a test, then use the run output directly in a docs build without rebuilding. nbmake claims to support this use case, but in our experience jupyter-book didn't recognise the nbmake output and tried to re-run the outputs anyway. Perhaps we were just using the combination of tools incorrectly (one thing to keep in mind if trying to make this work is that we want the execution time of the notebook to appear in our docs too).

Instead, we combine the docs building and testing steps. We build the docs using jupyter-book. In our notebooks, we include assertion cells (which we make togglable in the built docs using jupyter-book's support for this) to check that the notebooks have run correctly. This makes it clear to devs where this assertion is and allows users to look at it if they wish without forcing them to deal with that detail if they don't want to. As a result of the assertions in the notebook, the docs build won't pass unless the assertions pass i.e. our docs build is also our test. This is a bit yuck as we'd ideally have these two steps be separate, but we couldn't work out how to make existing tools do this so have gone with this plan b instead (PRs or issues to discuss solutions are very welcome, our suggestion would be to try to build any solution around a combination of nbmake, jupyter-cache and jupyter-book).

We do check the notebook formatting as a separate step. This is easy to do and very cheap.

If we had super long-running notebooks, we could start to track the outputs of our notebook cache i.e. store their outputs in the repository too. This would make the notebook untested in most cases (because the cache would be used by default) but this might be an ok compromise in some circumstances. We don't have such a use case yet so we haven't implemented it but we think the current solution at least provides a path for that so is a good choice for now.
