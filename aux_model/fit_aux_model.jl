using Statistics
using LinearAlgebra
using ProgressMeter
using FileIO
using ArgParse

mutable struct SortedDat{F,I}
    num_categories::I
    num_bits::I
    features::Array{F,2}
    labels::Array{I,1}
    category_ranges::Array{UnitRange{I}}
end

mutable struct DecisionTreeModel{F,I}
    num_categories::I
    num_bits::I
    category_to_leaf::Array{I,1}
    leaf_to_category::Array{I,1}
    weights::Array{F,2}
    biases::Array{F,1}
    avg_lls::Array{F,1}
end


function SortedDat(dir_path; dimensions = nothing, subset = "train")
    if dimensions === nothing
        dimensions = "full"
    else
        dimensions = "k$dimensions"
    end

    features = load_features(joinpath(dir_path, "$subset-features-$dimensions.np"))
    labels, num_categories = load_labels(joinpath(dir_path, "$subset-labels-first.np"))
    @assert size(features, 2) == size(labels, 1)

    num_bits = Int32(ceil(log2(num_categories)))

    # Sort data set by label.
    permutation = sortperm(labels)
    labels = labels[permutation]
    features = features[:, permutation]

    # Remember where data points for each label begin and end.
    category_ranges = Array{UnitRange{Int32}}(undef, 2^num_bits)
    data_i = 1
    for cat = 1:2^num_bits
        start = data_i
        while data_i <= length(labels) && labels[data_i] == cat
            data_i += 1
        end
        category_ranges[cat] = start:(data_i - 1)
    end

    SortedDat{Float32,Int32}(num_categories, num_bits, features, labels, category_ranges)
end

function load_features(path)
    println("Loading features from \"$path\" ...")
    file = open(path, "r")
    n = read(file, Int32)
    k = read(file, Int32)
    println("=> dimensions: $k x $n")
    features = reinterpret(Float32, read(file, 4 * k * n))
    @assert eof(file)
    close(file)

    reshape(features, Int64(k), Int64(n))
end

function load_labels(path)
    println("Loading labels from \"$path\" ...")
    file = open(path, "r")
    num_dat = read(file, Int32)
    num_categories = read(file, Int32)
    println("=> $num_dat data points over $num_categories categories")
    labels = reinterpret(Int32, read(file, 4 * num_dat)) .+ 1
    @assert eof(file)
    @assert length(labels) == num_dat
    close(file)
    labels, num_categories
end

function leaves_for_node(node, num_bits)
    level = Int32(floor(log2(node)))
    start = (node << (num_bits - level)) - (1 << num_bits) + 1
    stop = start + (1 << (num_bits - level)) - 1
    start:stop
end

softplus(x) = log(1 + exp(x))

function expand_dims(a, i)
    s = size(a)
    reshape(a, s[1:i - 1]..., 1, s[i:end]...)
end

struct Yes end
struct No end

function grad_log_joint(weight_and_bias, features, label_index, is_left, regularizer; hess = No())
    k = length(weight_and_bias) - 1
    scores = features' * view(weight_and_bias, 1:k) .+ weight_and_bias[k + 1]
    for i = 1:length(scores)
        if is_left[label_index[i]]
            scores[i] = - scores[i]
        end
    end

    signed_sigmoid_neg_scores = 1 ./ (1 .+ exp.(scores))
    signed_sigmoid_pos_scores = if hess isa Yes
        1 .- signed_sigmoid_neg_scores
    else
        nothing
    end

    for i = 1:length(signed_sigmoid_neg_scores)
        if is_left[label_index[i]]
            signed_sigmoid_neg_scores[i] = - signed_sigmoid_neg_scores[i]
            if hess isa Yes
                signed_sigmoid_pos_scores[i] = - signed_sigmoid_pos_scores[i]
            end
        end
    end

    features_signed_sigmoid_neg_scores = features .* expand_dims(signed_sigmoid_neg_scores, 1)

    grad = -regularizer * weight_and_bias  # Gradient of log prior.
    grad[1:k] += reshape(sum(features_signed_sigmoid_neg_scores, dims = 2), k)
    grad[k + 1] += sum(signed_sigmoid_neg_scores)

    if hess isa Yes
        features_signed_sigmoid_pos_scores = features .* expand_dims(signed_sigmoid_pos_scores, 1)
        hess_mat = Matrix{Float32}(regularizer * LinearAlgebra.I, k + 1, k + 1)
        hess_mat[1:k, 1:k] += sum(expand_dims(features_signed_sigmoid_pos_scores, 1) .* expand_dims(features_signed_sigmoid_neg_scores, 2),
            dims = 3)
        hess_mat[end, 1:k] = sum(features_signed_sigmoid_pos_scores .* expand_dims(signed_sigmoid_neg_scores, 1), dims = 2)
        hess_mat[1:k, end] = view(hess_mat, k + 1, 1:k)'
        hess_mat[end, end] += sum(signed_sigmoid_pos_scores .* signed_sigmoid_neg_scores)
        grad, hess_mat
    else
        grad
    end
end

function split_in_half(dat, categories, cat_emb, max_half, regularizer)
    k = size(cat_emb, 1)

    if isempty(categories)
        return zeros(Float32, k), Float32(0), Int32[], Int32[]
    elseif length(categories) == 1
        # Trivial classifier where all except one leaf are padding. Put the one relevant leaf into right subtree and
        # return a zero weight vector and a very large bias so that the left subtree has almost zero probability.
        return zeros(Float32, k), Float32(1000), Int32[], Int32[1]
    end

    max_half = min(max_half, length(categories))
    min_half = length(categories) - max_half
    @assert 0 <= min_half <= max_half

    # Make initial split at center.
    left_count = (min_half + max_half) ÷ 2

    # Initialize weight vector with PCA of category embeddings
    cur_cat_emb = cat_emb[:, categories]
    centered_cat_emb = cur_cat_emb .- Statistics.mean(cur_cat_emb, dims = 2)
    cov = centered_cat_emb * centered_cat_emb'  # shape (emb_dim, emb_dim)
    centered_cat_emb = nothing

    weight = reshape(LinearAlgebra.eigen(LinearAlgebra.Symmetric(cov), k:k).vectors, size(cov, 1))

    # Sort scores and initialize partition.
    # We use the relation `log(sigmoid(-x)) = log(sigmoid(x)) - x`.
    cat_scores = cur_cat_emb' * weight
    permutation = sortperm(cat_scores)
    is_left = BitArray{1}(undef, length(categories))
    is_left[view(permutation, 1:left_count)] .= true
    is_left[view(permutation, left_count + 1:length(permutation))] .= false

    # Extract relevant points from data set.
    num_data_points = Int32(0)
    for cat = categories
        num_data_points += length(dat.category_ranges[cat])
    end
    features = Array{Float32}(undef, size(cat_emb, 1), num_data_points)
    label_index = Array{Int32}(undef, num_data_points)  # Index into `is_left`.
    i = 1
    for (idx, cat) = enumerate(categories)
        r = dat.category_ranges[cat]
        features[:,i:i + length(r) - 1] = dat.features[:,r]
        label_index[i:i + length(r) - 1] .= idx
        i += length(r)
    end
    @assert i == num_data_points + 1

    # Initialize bias such that scores have zero mean.
    # A roughly correct initial bias is important for Newton's descent to work because
    # the objective function has a very low curvature far away from the optimum.
    bias = -Statistics.mean(features' * weight)

    weight_and_bias = vcat(weight, Float32[bias])
    steps_since_change = 0
    steps_until_hess = 0
    hess = Array{Float32}(undef, 1, 1)
    total_step = 0

    while steps_since_change < 10
        # Continuous optimization: optimize until convergence but break if change is large enough to warrant discrete update
        path_length = Float32(0)
        while steps_since_change < 10 && (total_step < 10 || path_length < 0.01)
            if steps_until_hess == 0
                grad, hess = grad_log_joint(weight_and_bias, features, label_index, is_left, regularizer, hess = Yes())
                steps_until_hess = 8
            else
                grad = grad_log_joint(weight_and_bias, features, label_index, is_left, regularizer)
                steps_until_hess -= 1
            end

            step = hess \ grad
            step_norm = sqrt(sum(step.^2))
            path_length += step_norm
            if step_norm > 1.0
                steps_until_hess = 0
                step *= 1 / step_norm
            end
            weight_and_bias += step

            if step_norm > 1e-10
                steps_since_change = 0
            else
                steps_since_change += 1
            end
            total_step += 1
        end

        # Discrete optimization.
        # We use the relation `log(sigmoid(-x)) = log(sigmoid(x)) - x`.
        cat_scores = cur_cat_emb' * weight
        permutation = sortperm(cat_scores)
        sorted_scores = cat_scores[permutation]
        first_found_pos = findfirst(x->x >= 0, sorted_scores)
        last_negative = if first_found_pos === nothing
            length(sorted_scores)
        else
            first_found_pos - 1
        end
        left_count = max(min_half, min(max_half, last_negative))

        if (!all(is_left[view(permutation, 1:left_count)]) || any(is_left[view(permutation, left_count + 1:length(permutation))]))
            is_left[view(permutation, 1:left_count)] .= true
            is_left[view(permutation, left_count + 1:length(permutation))] .= false
            steps_since_change = 0
            steps_until_hess = 0
            if left_count == 0 || left_count == length(permutation)
                # Classifier became trivial as one child tree only contains paddings. Once we make this decision we go all in.
                return zeros(Float32, k), Float32(1000), Int32[], collect(Int32, 1:length(permutation))
            end
        end
    end

    view(weight_and_bias, 1:k), weight_and_bias[end], view(permutation, 1:left_count), view(permutation, left_count + 1:length(permutation))
end

function optimize_tree(dat, regularizer, scale_regularizer::Bool; normalize = false)
    cat_emb = zeros(Float64, size(dat.features, 1), 2^dat.num_bits)
    if normalize
        for (category, data_range) = enumerate(dat.category_ranges)
            if length(data_range) != 0
                cat_emb[:, category] = view(Statistics.mean(Array{Float64}(view(dat.features, :, data_range)), dims = 2), :, 1)
            end
        end
        cat_emb[:,1:dat.num_categories] .-= Statistics.mean(cat_emb[:,1:dat.num_categories], dims = 2)
    else
        for (category, data_range) = enumerate(dat.category_ranges)
            cat_emb[:,category] = view(sum(Array{Float64}(view(dat.features, :, data_range)), dims = 2), :, 1)
            # No need to center since feature matrix is already centered.
        end
    end

    @assert isapprox(Statistics.mean(cat_emb, dims = 2), zeros(size(dat.features, 1), 1), atol = 1e-8)

    leaf_to_category = Array(Int32(1):Int32(2^dat.num_bits))
    k = size(dat.features, 1)
    node_weights = Array{Float32}(undef, k, 2^dat.num_bits - 1)
    node_biases = Array{Float32}(undef, 2^dat.num_bits - 1)


    @showprogress for layer = 1:dat.num_bits
        for node = 2^(layer - 1):(2^layer - 1)
            cur_leaves = leaves_for_node(node, dat.num_bits)
            cur_categories_with_padding = leaf_to_category[cur_leaves]
            cur_categories = filter(x->x <= dat.num_categories, cur_categories_with_padding)
            cur_padding = filter(x->x > dat.num_categories, cur_categories_with_padding)
            reg = if scale_regularizer
                regularizer * length(cur_categories)
            else
                regularizer
            end
            node_weights[:, node], node_biases[node], indices_left, indices_right = split_in_half(
                dat, cur_categories, cat_emb, length(cur_leaves) ÷ 2, reg)
            leaf_to_category[cur_leaves] = vcat(cur_categories[indices_left], cur_padding, cur_categories[indices_right])
        end
    end

    # Check that `leaf_to_category` is a permutation.
    counts = zeros(Int32, (2^dat.num_bits,))
    for i = leaf_to_category
        counts[i] += 1
    end
    @assert all(counts .== 1)

    category_to_leaf = Array{Int32}(undef, dat.num_categories)
    for (leaf, category) = enumerate(leaf_to_category)
        if category <= dat.num_categories
            category_to_leaf[category] = leaf
        end
    end

    leaf_to_category, category_to_leaf, node_weights, node_biases
end

function average_category_log_likelihoods(dat::SortedDat, category_to_leaf, weights, biases)
    avg_lls = Array{Float32}(undef, dat.num_categories)
    bits = Int32[1<<i for i = (dat.num_bits - 1):-1:0]
    for (cat, (leaf, r)) in enumerate(zip(category_to_leaf, dat.category_ranges))
        decisions = ((leaf - 1) .& bits) .!= 0
        nodes = Int32[(leaf - 1 + 2^dat.num_bits) >> i for i = dat.num_bits:-1:1]
        cur_weights = weights[:, nodes]
        cur_biases = biases[nodes]
        for i in 1:dat.num_bits
            if decisions[i]
                cur_weights[:, i] .*= -1
                cur_biases[i] *= -1
            end
        end
        scores = view(dat.features, :, r)' * cur_weights .+ expand_dims(cur_biases, 1)  # shape (num_dat, num_bits)
        avg_lls[cat] = - sum(log.(1 .+ exp.(scores))) / length(r)
    end

    avg_lls
end


arg_parser = ArgParseSettings()
@add_arg_table arg_parser begin
    "input_dir"
        help = "Path to indput directory."
    "output_file"
        help = "Path to output file."
    "--reg"
        help = "Regularizer strength."
        arg_type = Float32
        default = Float32(1.0)
    "--frequency_reg"
        help = "Multiply regularizer strength with number of data points under node."
        action = :store_true
    "-d", "--embedding_dim"
        help = "Embedding dimension."
        arg_type = Int
        default = 16
end

args = parse_args(ARGS, arg_parser, as_symbols = true)

dat = SortedDat(args[:input_dir], dimensions = args[:embedding_dim])
println("Starting fit ...")
(leaf_to_category, category_to_leaf, weights, biases), elapsed, _ = @timed optimize_tree(dat, args[:reg], args[:frequency_reg])
println("done ($elapsed seconds).")

avg_lls = average_category_log_likelihoods(dat, category_to_leaf, weights, biases)
println("Average log likelihood per category on training set: $(Statistics.mean(avg_lls))")

save(args[:output_file], "model",
DecisionTreeModel(dat.num_categories, dat.num_bits, category_to_leaf, leaf_to_category, weights, biases, avg_lls))
