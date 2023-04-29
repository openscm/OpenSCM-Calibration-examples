# OpenSCM-Calibration-examples

Long-running examples using [OpenSCM-Calibration](https://github.com/openscm/OpenSCM-Calibration).

## Installation

After cloning the repository, we recommend installing with [poetry](https://python-poetry.org/).

```bash
poetry install
```

## Running the examples

All the examples can be run with (and the docs built with)

```bash
poetry run jupyter-book build book
```

## Rationale

As a user of [OpenSCM-Calibration](https://github.com/openscm/OpenSCM-Calibration)
(or indeed any repository), it is helpful to have examples of how it is used
in production and the outputs it produces. However, including such examples in
the core development repository comes with a problem: it adds a very slow step
to any continuous integration, which quickly becomes really annoying for
development. A secondary problem is that you also end up with output in the
repository, which quickly bloats it (even notebooks can be megabytes in size
if they contain many plots).

The solution we use here is to house our production-style examples in a
separate repository. We don't run these examples every time we change the code
base, however we do run them regularly and check/update them when the core
repository makes new releases.

### Further details

Even in this repository, we don't currently store outputs in the notebooks.
The reason is that outputs can quickly become bloated and storing the outputs
discourages re-running i.e. testing the notebooks.

Having looked around, we haven't found a good solution that allows us to run
our notebooks once as part of a test, then use the run output directly in a
docs build without rebuilding. [nbmake](https://github.com/treebeardtech/nbmake)
claims to support this use case, but in our experience jupyter-book didn't
recognise the nbmake output and tried to re-run the notebooks anyway. Perhaps
we were just using the combination of tools like [nbmake](https://github.com/treebeardtech/nbmake),
[jupyter-cache](https://github.com/executablebooks/jupyter-cache) and
[jupyter-book](https://github.com/executablebooks/jupyter-book) incorrectly
(one thing to keep in mind if trying to make this work is that we want the
execution time of the notebook to appear in our docs too).

Instead, we combine the docs building and testing steps. We build the docs
using jupyter-book, and include, in our notebooks, assertion cells that act
effectively as tests. We avoid polluting the entire notebook with these
assertions by making them hidden by default (and use jupyter-book's support
for showing and hiding cells to give users and readers the chance to check
them if they wish). We like this solution because it makes clear to developers
where the assertion is (other solutions hide the assertions in the cell's
JSON, which feels like a hack to us) and allows users to look at it if they
wish while making clear that they aren't actually necessary for the example to
run.

We do check the notebook formatting as a separate step. This is easy to do and
very cheap using [blacken-docs](https://github.com/adamchainz/blacken-docs).

### Really long-running notebooks

If we want to add super long-running notebooks to this repository, one
possible problem is that they are too long-running to reasonably run in CI
(perhaps they take 3 days to run). In this case, one solution could be to
start tracking some of the outputs of our notebook cache. This would make the
notebook untested in most cases (because the cache would be used instead of
running the notebook) but this might be the best compromise where running the
notebook is truly not an option. We don't have such a use case yet so we
haven't implemented this, but we think the current solution doesn't shut the
door on really long-running notebooks so is a good choice for now.

## For developers

For development, we rely on [poetry](https://python-poetry.org) for all our
dependency management. For all of work, we use our `Makefile`.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone and doesn't
update if dependencies change (e.g. the environment is updated).
In order to create your environment, run `make virtual-environment -B`.

If there are any issues, the messages from the `Makefile` should guide you
through. If not, please raise an issue in the
[issue tracker](https://github.com/openscm/OpenSCM-Calibration_examples/issues).
