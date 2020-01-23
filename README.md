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

4. neovim

```
curl -LO https://github.com/neovim/neovim/releases/download/stable/nvim.appimage
chmod u+x nvim.appimage
./nvim.appimage

# then add alias for ./nvim.appimage to .zshrc
```

## Use DeepGreen

Submit
```
sbatch submit.sh
```

check on it
```
squeue -u {username}
```

Cancel it

```
scancel <job_id>
```

Watch the output as it runs
```
# go to output folder that you set in submit.sh

tail -f <output_file_name>
```

Use srun to just normally run shit

```
srun -N 1 -p dggpu --gres=gpu:2 --accel-bind=g --ntasks=4 --pty bash


```

Check gpus (tensorflow v1; for v2 it's a different command)
```
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```
