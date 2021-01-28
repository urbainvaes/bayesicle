functions = ["ackley", "rastrigin"]
bs = [0, 1, 2]
ds = [2, 10]

for f in functions
    for b in bs
        for d in ds
            run(`tmux new-window -n "f=$f,b=$b,d=$d" "julia test_cbs.jl $b $f $d > $f-$b-$d.txt"`)
        end
    end
end
