using ArgParse
using Random
using CUDAnative
using CuArrays
using Flux
using FileIO

include("DecisionStrain.jl")
using .DecisionStrain
using .DecisionStrain.Data
using .DecisionStrain.FluxFixes

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu"
            help = "Select GPU to use."
            arg_type = Int
            default = 0
        "--dat", "-i"
            help = "path to directory with data set"
            arg_type = String
            default = "dat/FB15K"
        "--binary_dat"
            help = "Data set is stored in binary files."
            action = :store_true
        "--model"
            help = "select model. must be either DecisionStrain or ComplEx"
            arg_type = String
            default = "DecisionStrain"
        "--head_prediction"
            help = "Only used for --model DecisionStrain: train head prediction rather than tail prediction"
            action = :store_true
        "--aux"
            help = "path to auxiliary model"
            arg_type = String
        "--epochs", "-E"
            help = "number of training epochs"
            arg_type = Int
            default = 100
        "--batchsize", "-B"
            help = "minibatch size"
            arg_type = Int
            default = 500
        "--eval_batchsize"
            help = "minibatch size for evaluation"
            arg_type = Int
            default = 500
        "--hidden_dim", "-k"
            help = "dimension of embedding space"
            arg_type = Int
            default = 20
        "--lmbda"
            help = "prefactor of regularizer"
            arg_type = Float64
            default = 0.01
        "--lr"
            help = "learning rate"
            arg_type = Float64
            default = 1.0
        "--neg_ratio"
            help = "Number of negative samples per positive sample"
            arg_type = Int
            default = 10
        "--rngseed"
            help = """
                seed for random number generator; if not provided, a seed will
                automatically be generated from system entropy and written to
                the output file."""
            arg_type = UInt32
        "--population", "-n"
            help = "number of simultaneously trained decision trees"
            arg_type = Int
            default = 1
        "--out", "-o"
            help = "output directory; must not exist"
            arg_type = String
        "--continue", "-c"
            help = "provide file name to continue training from."
            arg_type = String
        "--eval_every"
            help = "evaluate model every n epochs on validation set (set to zero to evaluate only at end)"
            arg_type = Int
            default = 10
        "--save_every"
            help = "save checkpoing every n epochs (set to zero to save only at end)"
            arg_type = Int
            default = 100
        "--copy_subtrees"
            help = "copy branching vectors of winning subtrees"
            action = :store_true
        "--precision"
            help = """
                floating point precision;  allowed values are "half", "single",
                and "double".  Default is "single"."""
            arg_type = Union{Type{Float16}, Type{Float32}, Type{Float64}}
            default = Float32
    end

    return parse_args(s, as_symbols = true)
end

function ArgParse.parse_item(
        ::Type{Union{Type{Float16}, Type{Float32}, Type{Float64}}},
        s::AbstractString)
    if s == "half"
        return Float16
    elseif s == "single"
        return Float32
    elseif s == "double"
        return Float64
    else
        throw("""Unknown float type "$s".""")
    end
end

function dicttostr(dict::Dict)
    stream = IOBuffer()
    show(stream, MIME("text/plain"), dict)
    lines = split(String(take!(stream)), "\n")
    "Dict(\n" * join("$line,\n" for line in lines[2:end]) * ")"
end

function main()
    args = parse_commandline()

    device!(args[:gpu])

    # initialize random number generator
    if args[:rngseed] === nothing
        args[:rngseed] = rand(RandomDevice(), UInt32)
    end
    rng = MersenneTwister(args[:rngseed]);

    if args[:out] === nothing
        args[:out] = joinpath("out", repr(hash(args))[end - 5 : end])
    end

    # create output directory and open log file
    print("Output directory: $(args[:out])\n")
    mkdir(args[:out])
    try
        open(joinpath(args[:out], "log"), write = true) do log
            # TODO: print date, time, hostname, git commit, and $CUDA_VISIBLE_DEVICES
            print(log, """
            args = $(dicttostr(args))

            # Running on $(Threads.nthreads()) threads.

            """)

            dat = Dataset(args[:dat], binary_files = args[:binary_dat], log = log,
                          reverse = args[:head_prediction])
            if args[:continue] === nothing
                if args[:model] == "DecisionStrain"
                    model = DecisionStrainModel(rng, dat, args, log = log)
                elseif args[:model] == "ComplEx"
                    aux_model = nothing
                    if args[:aux] !== nothing
                        aux_model = gpu(load(args[:aux], "model"), args, aux_model=true)
                    end
                    model = ComplExModel(rng, dat, args, aux_model)
                else
                    throw("Unknown model $(args[:model]).")
                end
                print(log, "# Initialized new model.\n\n")
            else
                model = gpu(load(args[:continue], "model"), args)
                print(log, "# Loaded model from file $(args[:continue])\n\n")
            end
            flush(log)

            # Run the training loop in a separate function to allow the compiler to generate
            # specialized code based on floattype(model).
            training_loop!(
                model, dat, rng, args[:epochs], args[:batchsize], args[:eval_every],
                args[:save_every], args[:out], log = log)
        end
    catch e
        open(joinpath(args[:out], "err"), write = true) do err
            trace = catch_backtrace()
            showerror(err, e, trace)
            print(err, "\n")
            rethrow()
        end
    end
end

function training_loop!(
        model::AbstractKnowledgGraphModel,
        dat::Dataset,
        rng::AbstractRNG,
        nepochs::Integer,
        batchsize::Integer,
        eval_every::Integer,
        save_every::Integer,
        out_path::AbstractString;
        log::IO = stdout)

    trainlength = size(dat.train_unsorted, 2)
    num_batches = trainlength ÷ batchsize

    # print(log, "initial_performance = ", evaluate(model, dat.val, dat), "\n\n")
    print(log, """
    col_headers = ["epoch", "avg_duration_continuous", "avg_duration_discrete",
        "mrr", "hits_at", "log_likelihood_per_data_point", "eval_fail_cnt"]

    data = [
    """)
    flush(log)

    sum_duration_cont = 0.0
    sum_duration_disc = 0.0
    duration_disc = 0.0
    last_print = 0

    for epoch in 1 : nepochs
        # print_stats(model)
        _, duration_cont = @timed begin
            shuffled = CuArray(shuffle_train(rng, dat))
            for batchstart = 1 : batchsize : batchsize * num_batches
                train_minibatch!(
                    rng, model, shuffled, batchstart, batchsize, epoch, log = log)
            end
        @sync_cuda("continuous updates done")
        end
        sum_duration_cont += duration_cont

        if needs_discrete_optimization(model)
            _, duration_disc = @timed begin
                train_epoch!(rng, model, dat, epoch, batchsize, log = log)
                @sync_cuda("discrete updates done")
            end
            sum_duration_disc += duration_disc
        else
            duration_disc = 0.0
        end

        GC.gc()

        print(log, "# Epoch $epoch took $duration_cont + $duration_disc seconds.\n")
        flush(log)

        if save_every != 0 && epoch % save_every == 0
            save_checkpoint(model, out_path, epoch)
            GC.gc()
        end

        # if eval_every != 0 && epoch % eval_every == 0
        #     print(log, (
        #         epoch,
        #         sum_duration_cont / (epoch - last_print),
        #         sum_duration_disc / (epoch - last_print),
        #         evaluate(model, dat.val, dat)...
        #         ), ",\n")
        #     flush(log)
        #     sum_duration_cont = sum_duration_disc = 0.0
        #     last_print = epoch
        #     GC.gc()
        # end
    end

    if save_every == 0 || nepochs % save_every != 0
        save_checkpoint(model, out_path, nepochs)
        GC.gc()
    end

    # if eval_every == 0 || nepochs % eval_every != 0
    #     print(
    #         log,
    #         (nepochs, nothing, nothing, evaluate(model, dat.val, dat)...),
    #         ",\n"
    #     )
    # end
    print(log, "]\n")
    flush(log)
end

@noinline function save_checkpoint(model, directory, epoch)
    save(joinpath(directory, "checkpoint-$epoch.jld2"), "model", cpu(model))
end

@noinline function evaluate(
            model::AbstractKnowledgGraphModel,
            heldout::AbstractArray{<:Integer, 2},
            fulldat::Dataset;
            batchsize = 100,
            windows=[1, 3, 10])
    hits_cnt_at_t = zeros(Int, length(windows), Threads.nthreads())
    hits_cnt_at_h = zeros(Int, length(windows), Threads.nthreads())
    sum_rr_t = zeros(Float64, Threads.nthreads())
    sum_rr_h = zeros(Float64, Threads.nthreads())
    cnt = zeros(Int, Threads.nthreads())
    total_ll = 0.0

    for batchstart = 1 : batchsize : size(heldout, 2)
        batchend = min(batchstart + batchsize - 1, size(heldout, 2))
        batch = batchstart : batchend
        hs = heldout[1, batch]
        ts = heldout[2, batch]
        rs = heldout[3, batch]
        all_ll, predicted_lls, raw_ranks, sum_ll = log_likelihoods_and_raw_ranks(model, hs, ts, rs)
        total_ll += sum_ll

        Threads.@threads for i = 1 : length(batch)
            predicted_ll = predicted_lls[i]
            raw_rank = raw_ranks[i]
            other_correct_ts = view(fulldat.all_sorted, fulldat.all_ranges[(hs[i], rs[i])])
            other_correct_lls = all_ll[other_correct_ts, i]
            filtered_rank_t = raw_rank - sum(other_correct_lls .> predicted_ll)

            threadid = Threads.threadid()

            for (ii, w) = enumerate(windows)
                if filtered_rank_t <= w
                    hits_cnt_at_t[ii, threadid] += 1
                end
            end

            sum_rr_t[threadid] += 1 / filtered_rank_t
            cnt[threadid] += 1
        end
    end

    success_cnt = sum(cnt)
    hits_cnt_at_t = sum(hits_cnt_at_t, dims = 2)[:, 1]
    hits_at_t = Dict(w => hits / success_cnt for (w, hits) = zip(windows, hits_cnt_at_t))
    mrr_t = sum(sum_rr_t) / success_cnt
    return mrr_t, hits_at_t, total_ll / size(heldout, 2), size(heldout, 2) - success_cnt
end

main()
