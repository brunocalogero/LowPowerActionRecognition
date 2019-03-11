# LowPowerActionRecognition
Low-power  action recognition by coupling Silicon Retina to a Smartphone

## Written by: BRUNO CALOGERO

This will eventually hold all the setup information for the user to leverage our overall system.


## To use this repo:

- First time you use this:
  - clone the repo: `git clone https://github.com/brunocalogero/LowPowerActionRecognition.git`

- Once you have cloned and you think your friends updated the codebase:
  - you need to pull their updates: `git pull --rebase`

- If you made changes to your local project that you 'cloned' and you want to 'push' those changes to github so that the rest of the team can see your progress:
  - you need to: `git status` to see what files you made changes on and that or not present on the github repo
  ```
  BCALOGER-M-J5VM:LowPowerActionRecognition bcaloger$ git status
  On branch master
  Your branch is up-to-date with 'origin/master'.

  Changes not staged for commit:
    (use "git add <file>..." to update what will be committed)
    (use "git checkout -- <file>..." to discard changes in working directory)

    modified:   .gitignore
  ```
  - you therefore need to 'add' the file you changed, which will be the file that is going to be 'pushed' to github
  ```
  BCALOGER-M-J5VM:LowPowerActionRecognition bcaloger$ git add .gitignore
  BCALOGER-M-J5VM:LowPowerActionRecognition bcaloger$ git status
  On branch master
  Your branch is up-to-date with 'origin/master'.

  Changes to be committed:
    (use "git reset HEAD <file>..." to unstage)

	modified:   .gitignore
  ```
  The file I staged for the 'commit' has now turned green, meaning i am ready to send it / 'push' it to github
  - now you need to `git commit`, in other words give a description to the changes you are about to push
  ```
  BCALOGER-M-J5VM:LowPowerActionRecognition bcaloger$ git commit -m "I modified the gitignore"
  [master ac347ba] I modified the gitignore
   1 file changed, 2 insertions(+), 2 deletions(-)
  ```
  - Now the last step is to simply `git push`
  ```
  BCALOGER-M-J5VM:LowPowerActionRecognition bcaloger$ git push
  Counting objects: 3, done.
  Delta compression using up to 8 threads.
  Compressing objects: 100% (2/2), done.
  Writing objects: 100% (3/3), 303 bytes | 303.00 KiB/s, done.
  Total 3 (delta 1), reused 0 (delta 0)
  remote: Resolving deltas: 100% (1/1), completed with 1 local object.
  To https://github.com/brunocalogero/LowPowerActionRecognition.git
     9271fa1..ac347ba  master -> master
  ```
  Look into: git merge conflicts

## File Structure
