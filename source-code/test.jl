include("obj.jl")
include("ogi.jl")

obj = load_obj_file("../data/cube_binary.obj")

convert_obj_to_gi(obj, 500, 500)
