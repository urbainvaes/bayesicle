import git

# Directory of the data
git_repo = git.Repo('.', search_parent_directories=True)
data_root = "{}/data".format(git_repo.working_dir)
fig_root = "{}/figures".format(git_repo.working_dir)

def direct_min(f, x0):
    argmin = opti.basinhopping(f, x0)
    if argmin.lowest_optimization_result.success:
        argmin, fmin = argmin.x, argmin.fun
    else:
        print("Warning: Could not locate minimum!")
        argmin, fmin = argmin.x, argmin.fun
    return argmin, fmin
