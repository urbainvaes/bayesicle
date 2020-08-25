import git

# Directory of the data
git_repo = git.Repo('.', search_parent_directories=True)
data_root = "{}/data".format(git_repo.working_dir)
