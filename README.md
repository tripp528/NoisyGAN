# NoisyGAN

## Git Workflow

Summary: Make sure master is up to date on your local machine, branch off of it, then merge it back in.

[Shortcuts for the commands](https://github.com/ohmyzsh/ohmyzsh/wiki/Cheatsheet)

### 1. Pull from master
(make sure you're on master branch)
```
git pull origin master
```

### 2. Create a new branch
```
git branch yourNameAndFeature
```

### 3. Check out a branch (switch branches)

```
git checkout yourNameAndFeature
```

### 4. Push your new branch that you just made to the remote
```
git push -u origin yourNameAndFeature
```

### 5. Do all the work in your branch, commit and push frequently

```
git add --all
git commit -m "commit message"
git push origin yourNameAndFeature
```
Shortcuts
```
gaa, gcmsg "", ggp
```

### 6. Create pull request when it's stable (on github website)
:)

### If you want to pull my remote branch and check it out
```
git checkout --track origin/branchYouWant
```


## Setting up DeepGreen

### zsh/ohmyzsh

1. [zsh without root access](https://stackoverflow.com/questions/15293406/install-zsh-without-root-access)

2. Put ```exec ~/bin/zsh -l``` in a .sh file and run it on login.

3. [Oh my zsh](https://github.com/ohmyzsh/ohmyzsh)
