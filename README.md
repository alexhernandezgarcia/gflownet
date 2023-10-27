# Private sister repository of gflownet

This repository (`gflownet-dev`) is private. It is meant to be used to develop research ideas and projects before making them public in the original [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet) repository (`gflownet`).

As of October 2023, it is uncertain whether we will stick to this plan in the long term, but the idea is the following:

- Develop ideas and projects in `gflownet-dev`.
- Upon publication or whenever the authors feel comfortable, transfer the relevant code to `gflownet`. 
- Relevant code improvements and development that does not compromise research projects should be transferred to `gflownet` as early as possible.

This involves extra complexity, so we will re-evaluate or refine this plan after a test period.

## Short guide to work multiple remote repositories

Here is a basic guide about how to work with multiple repositories, in our case the "public" repository ([alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet)) and the "dev" repository ([alexhernandezgarcia/gflownet-dev](https://github.com/alexhernandezgarcia/gflownet-dev)).

### Listing the remote repositories

First, you can list the current remote repositories with:

```bash
git remote -v
```

If you have not made any changes to the default configuration, you will likely get something like:

```bash
origin  git@github.com:alexhernandezgarcia/gflownet-dev.git (fetch)
origin  git@github.com:alexhernandezgarcia/gflownet-dev.git (push)
```

### Adding a new remote repository

Now, it may be useful to add to the list of remote repositories the "public" repository. If we want to call it `public`, we can use this command:

```bash
git remote add public git@github.com:alexhernandezgarcia/gflownet.git
```

The updated list shown by `git remote -v` should now be:

```
origin  git@github.com:alexhernandezgarcia/gflownet-dev.git (fetch)
origin  git@github.com:alexhernandezgarcia/gflownet-dev.git (push)
public  git@github.com:alexhernandezgarcia/gflownet.git (fetch)
public  git@github.com:alexhernandezgarcia/gflownet.git (push)
```

### Renaming a remote repository

It may also be helpful to rename the remote `origin` to `dev`, to avoid confusion. The command is `git remote rename <old-name> <new-name>`. So after running `git remote rename origin dev`, `git remote -v` should return:

```bash
dev     git@github.com:alexhernandezgarcia/gflownet-dev.git (fetch)
dev     git@github.com:alexhernandezgarcia/gflownet-dev.git (push)
public  git@github.com:alexhernandezgarcia/gflownet.git (fetch)
public  git@github.com:alexhernandezgarcia/gflownet.git (push)
```

### Fetching a branch from the public repository

A common action we may want to often do is fetching changes in the public repository, for instance after a PR is merged into `main`. These are some steps we can follow, assuming we have added the public repository to the list of remotes (see above).

1. Fetch the branch from the public repository that we are interested in. For instance, `main`:

```bash
git fetch public main
```

2. Create a new branch for the new changes. For instance, `main-public`:

```bash
git checkout -b main-public
```

Note that the order of the first two steps above does not matter.

3. Merge the main branch of the public repository into the new branch:

```bash
git merge public/main main-public
```

4. Push the changes to a new remote branch (for example, `main-public`) on the dev repository:

```bash
git push origin main-public
```
Finally, it is also possible to set the default upstream branch to the desired repository and branch. For example, to set it to the main branch of the public repository:

```bash
git branch --set-upstream-to=public/main
```
