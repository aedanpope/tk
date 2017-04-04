# Manual things to do:

# Run
# $echo "source ~/aedan.bashrc" >> ~/.bashrc
# Install sublime text
# Set syntax highlighting for .bashrc files to "View > Syntax > Shell Script"


# Custom BashRC commands

# Add a new line at the end of the command prompt
#PS1=${PS1}\\n

PS1=${PS1%?}
PS1=${PS1%?}\n'$ '

alias rb="source ~/.bashrc"
alias s=subl
alias p=python

# Wine
WINE_DIR=${WINEPREFIX:-~/.wine}

# StarCraft

#export STARCRAFT=/media/aedanpope/Tee/StarCraft
#alias sc="wine explorer /desktop=foo,1024x768 /media/aedanpope/Tee/StarCraft/StarCraft.exe"
#alias sc="wine /media/aedanpope/Tee/StarCraft/StarCraft.exe"
#alias chaoslauncher="wine /media/aedanpope/Tee/BWAPI/Chaoslauncher/Chaoslauncher.exe"
#alias chaoslauncher="wine /media/aedanpope/Tee/StarCraft/BWAPI/Chaoslauncher/Chaoslauncher.exe"

# TorchCraft

export STARCRAFT=~/.wine/drive_c/StarCraft
alias sc="wine explorer /desktop=foo,1024x768 $STARCRAFT/StarCraft.exe"
alias sc_server="cd $STARCRAFT && wine bwheadless.exe -e $STARCRAFT/StarCraft.exe -l $STARCRAFT/bwapi-data/BWAPI.dll --headful"
# Doesn't work:
# alias sc_server_headless="wine $STARCRAFT/bwheadless.exe -e $STARCRAFT/StarCraft.exe"
# Waits for config before starting.
alias sc_server_cfg="wine $STARCRAFT/BWEnv.exe"
# alias sc_client="th ~/TorchCraft/examples/simple_exe.lua -t 127.0.0.1"
alias sc_client="p ~/sc_dev/tc_client/exercise.py"
alias tb="tensorboard --logdir=/tmp/tfgraph"

# Torch

# REENABLE All 5 lines if w need torch.
# . /home/aedanpope/torch/install/bin/torch-activate
# export PATH=/home/user/torch/install/bin:$PATH
# export LD_LIBRARY_PATH=/home/user/torch/install/lib:$LD_LIBRARY_PATH
# export DYLD_LIBRARY_PATH=/home/user/torch/install/lib:$DYLD_LIBRARY_PATH
# eval $(luarocks path --bin)

# sudo chown -R $(whoami) ~/.cache
#$ cat /etc/environment
#PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games"
#sudo echo "~/torch/install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games" > /etc/environment
# http://stackoverflow.com/questions/14637979/how-to-permanently-set-path-on-linux
# https://github.com/torch/nngraph/issues/52

# Git

git config --global user.email "aedanpope@gmail.com"
git config --global user.name "Aedan Pope"
# git pwd AgeBitSameSympRule5ghb

# Set the cache to timeout after 1 days (setting is in seconds)
git config --global credential.helper 'cache --timeout=86400'

gpush () {
  if [[ -z  $1 ]]
  then
    echo "Need commit message"
    return
  fi
  git add *
  git commit -m "$1"
  git push -u origin master
}

toc() {
  # Install from https://github.com/ekalinin/github-markdown-toc

  if [[ -z  $1 ]]
  then
    echo "Need file to toc"
    return
  fi

  content=$(cat $1) # no cat abuse this time
  toc=$(~/tools/gh-md-toc $1)
  echo -en "$toc\n$content" > $1
}

echo "loaded aedan.bashrc"