# Contributing to InVEST

## Asking Questions
Do you have a question about using InVEST or need help troubleshooting a specific issue? Please visit the [NatCap Community Forum](https://community.naturalcapitalproject.org/).

## Reporting Bugs
Did you find a bug in the InVEST software?
1. Search the [InVEST Open Issues on GitHub](https://github.com/natcap/invest/issues) to see if it has already been reported.
2. If you don’t find an existing issue, create a new issue. Select “Bug report”, and provide as much detail as you can.
3. Do you know how to fix the bug? Learn how we handle code contributions in the [Contributing Code](#contributing-code) section of this guide.

## Requesting Features
A feature is any addition or modification that affects users of the InVEST software but does not directly address a specific bug. A feature may be very small (e.g., minor styling updates in a small component of the Workbench), very large (e.g., a new InVEST model), or somewhere in between.

If you want to suggest an InVEST feature:
1. Search the [InVEST Open Issues on GitHub](https://github.com/natcap/invest/issues) to see if it has already been suggested.
2. If you don’t find an existing issue, create a new issue. Select “Feature request”, and provide as much detail as you can.

Every proposed feature, large or small, will be weighed in terms of its potential impact on InVEST users and the capacity of the NatCap team to implement and maintain the feature. Please note that major features, including new InVEST models, must go through a peer-review process with our Platform Steering Committee. In addition, requests for science-related changes to existing InVEST models, such as the modification of equations, or other changes to the way the model functions, will require a NatCap science review.

## Contributing Code

### Contributor License Agreement
Any code you contribute to InVEST will fall under the same license as InVEST. Please read the [InVEST Contributor License Agreement](https://natcap.github.io/invest-cla) and, if you agree to the terms, sign it. If you do not sign the Contributor License Agreement, you will not be able to contribute code to InVEST.

### Finding Open Issues
1. Browse the [InVEST Open Issues on GitHub](https://github.com/natcap/invest/issues). You may wish to search by keyword and/or filter by label, such as the `good first issue` label.
2. When you find something you want to work on, comment on the issue to let us know you’re interested in working on it.
3. Wait for a member of the NatCap team to respond. One of the following will happen:
    1. If it’s OK for you to work on the issue, a member of the NatCap team will let you know (in a comment on the issue) and assign the issue to you. Proceed to step 4.
    2. If there’s some reason you shouldn’t work on the issue (because the issue is invalid, or we are intentionally delaying work on it, or—oops—someone else is already working on it), a member of the NatCap team will let you know by leaving a comment on the issue. Return to step 1!
4. Review the [InVEST Contributor License Agreement](https://natcap.github.io/invest-cla), if you haven’t already. If you do not sign the Contributor License Agreement, you will not be able to contribute code to InVEST.

### Forking the Repository and Creating a Branch
1. Fork the `natcap/invest` repository. Check [GitHub’s “Fork a repository” guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) for details, if needed.
2. On your fork, create a new branch off `main`. Give your branch a descriptive name with the form: `<category>/<issue-number>-<short-description>`. For example:
    - `bugfix/12345-remove-moth-from-mainframe`,
    - `feature/23456-new-ui-theme`, or
    - `docs/34567-update-readme`.

### Working on Code Changes
1. Make sure you have followed the instructions in [Forking the Repository and Creating a Branch](#forking-the-repository-and-creating-a-branch).
2. Refer to the [InVEST README](./README.rst) for guidance in setting up your local development environment.
3. As you work, use commit messages that are descriptive and include the issue number (e.g., “Validate input LULC raster against biophysical table (#1234)”) rather than generic (e.g., “update code”). This will make it easier to distinguish between commits in the commit history, and it will make it easier for you to write a PR description.
4. Update automated tests as needed. If you are adding anything new, such as a new input parameter, a new output, more robust validation, or a larger feature, you should add tests to ensure the new behavior is covered.
5. Manually test your code changes to catch any issues the automated tests might have missed. Manual tests may include interacting with the Workbench UI to verify visual appearance as well as behavior, running a model (via the CLI or the Workbench) and verifying the outputs in a GIS, and/or other related activities.
6. Consistent code style keeps our codebase readable and helps us focus on the content of code changes. We follow the [PEP 8 Style Guide for Python Code](https://peps.python.org/pep-0008/) and [PEP 257 Python Docstring Conventions](https://peps.python.org/pep-0257/). When in doubt about how to format your code, please refer to existing InVEST code for examples.

### Submitting a Pull Request
1. Make sure you have followed the instructions in [Working on Code Changes](#working-on-code-changes).
2. Once your code is ready to be reviewed, open a draft PR.
    1. Select the appropriate target branch on the `natcap` fork. This should be the same branch you branched from when starting work on your PR. (For example: if you branched from `main`, the target branch should be `main`.) If you’re unsure what the target branch should be, ask us in a comment on the relevant issue.
    2. Double-check that your PR includes all the changes you want (and none that you don’t).
    3. Add a description of the changes you made. Be sure to reference the issue your PR is addressing.
    4. If you have made any UI changes, include relevant screenshots and/or screen recordings.
    5. Follow the checklist in the PR template. If you’ve completed an item, check its box.
    6. Select “Create draft pull request”.
3. As soon as your (draft) PR is open, a series of automated checks (test runners, build processes, etc.) will begin. Wait for these checks to complete.
4. After you’ve completed all relevant items on the checklist, and all the automated checks are passing, mark your PR “Ready for review”.
5. Wait for someone to review your PR. One or more members of the NatCap team will review your PR. They may request some additional changes and/or ask clarifying questions about the changes you’ve made. You may browse our closed PRs for examples of our PR review process, including conversations about PR details.
6. After you receive a review on your PR, you may need to do one or more of the following:
    1. Respond to reviewers’ questions or comments.
    2. Make additional changes, then request a “re-review” from your reviewer(s).
    3. Repeat steps 5 and 6 as needed.
7. Once the NatCap team has determined your PR is ready to be merged, they will approve and merge your PR. Thanks for contributing to InVEST! 🎉
